# Fork modification (he1016060110, 2026-10-06): verified company extraction and local embedding pipeline.
"""
爬虫任务 — 使用 Playwright 爬取目标网站
在独立的 crawler 容器中执行
"""
import uuid
import logging
import enum
import re
from urllib.parse import unquote, urlparse
from datetime import datetime, timezone

from celery import shared_task
from app.core.logging_utils import log_event
from app.services.company_ingest import (
    build_candidate_links,
    fallback_select_company_pages,
    normalize_company_url,
)
from app.tasks.runtime import run_async as _run

logger = logging.getLogger("georank.crawl")


def _normalize_update_values(values: dict) -> dict:
    """Convert Enum members to their database labels for Core updates."""
    normalized = {}
    for key, value in values.items():
        normalized[key] = value.value if isinstance(value, enum.Enum) else value
    return normalized


async def _update_company(company_id: str, **kwargs):
    from app.core.database import async_session
    from app.models.company import Company
    from sqlalchemy import update
    async with async_session() as db:
        await db.execute(
            update(Company)
            .where(Company.id == uuid.UUID(company_id))
            .values(**_normalize_update_values(kwargs))
        )
        await db.commit()


async def _admit_crawl_callback(company_id: str, task_id: str | None) -> bool:
    """DB-only owner check: historical callbacks cannot mutate or call AI."""
    from app.core.database import async_session
    from app.models.company import Company
    from sqlalchemy import select, update

    async with async_session() as db:
        result = await db.execute(
            select(Company).where(Company.id == uuid.UUID(company_id)).with_for_update()
        )
        company = result.scalar_one_or_none()
        if company is None:
            await db.rollback()
            return False
        details = dict(company.geo_details or {})
        dispatch = dict(details.get("pipeline_dispatch") or {})
        owner_task_id = dispatch.get("task_id")
        if owner_task_id and str(owner_task_id) != str(task_id):
            await db.rollback()
            return False
        # Legacy rows may enter once. Freeze their actual Celery ID so later
        # callbacks with another ID cannot reuse this paid crawl admission.
        if not owner_task_id and task_id:
            dispatch.update(task_id=str(task_id), state="running", legacy_claimed=True)
            details["pipeline_dispatch"] = dispatch
            await db.execute(
                update(Company).where(Company.id == company.id).values(geo_details=details)
            )
            await db.commit()
        else:
            await db.rollback()
        return True


async def _record_clean_dispatch(
    company_id: str, crawl_task_id: str | None, clean_task_id: str, state: str,
) -> bool:
    """Merge a fresh, owner-bound receipt without regressing worker progress."""
    from app.core.database import async_session
    from app.models.company import Company
    from sqlalchemy import select, update

    async with async_session() as db:
        result = await db.execute(
            select(Company).where(Company.id == uuid.UUID(company_id)).with_for_update()
        )
        company = result.scalar_one_or_none()
        if company is None:
            await db.rollback()
            return False
        details = dict(company.geo_details or {})
        dispatch = dict(details.get("pipeline_dispatch") or {})
        owner_task_id = dispatch.get("task_id")
        if owner_task_id and str(owner_task_id) != str(crawl_task_id):
            await db.rollback()
            return False
        dispatch.update(
            clean_task_id=clean_task_id, clean_state=state,
            clean_receipt_updated_at=datetime.now(timezone.utc).isoformat(),
        )
        values = {"geo_details": details}
        if state == "unknown":
            # A late broker observation must not erase downstream failure facts
            # or manufacture an error after verified downstream completion.
            from app.models.company import PipelineStatus
            if company.pipeline_status == PipelineStatus.CLEANING:
                dispatch["state"] = "unknown"
                values["pipeline_error"] = (
                    "COMPANY_CLEAN_DISPATCH_UNKNOWN: 清洗任务派发回执未知；"
                    "请只读观察，禁止自动重试官网分析或重复收费。"
                )
        details["pipeline_dispatch"] = dispatch
        await db.execute(update(Company).where(Company.id == company.id).values(**values))
        await db.commit()
        return True


async def _update_report(report_id: str, **kwargs):
    from app.core.database import async_session
    from app.models.diagnostic import DiagnosticReport
    from sqlalchemy import update
    async with async_session() as db:
        await db.execute(
            update(DiagnosticReport)
            .where(DiagnosticReport.id == uuid.UUID(report_id))
            .values(**_normalize_update_values(kwargs))
        )
        await db.commit()


def _crawl_page(url: str, timeout_ms: int = 30000) -> dict:
    """
    使用 Playwright 爬取单个页面，返回 {html, text, title, links}
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        try:
            page = browser.new_page(
                user_agent="Mozilla/5.0 (compatible; GEOrankBot/1.0; +https://georank.com/bot)"
            )
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response is not None and response.status >= 400:
                raise RuntimeError(f"页面抓取 HTTP {response.status}：{url}")
            # 等待主要内容渲染
            page.wait_for_timeout(2000)

            html = page.content()
            title = page.title()

            # 提取纯文本
            text = page.evaluate("() => document.body.innerText || ''")

            # 提取所有一级导航候选链接
            links = page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        url: a.href,
                        title: (a.textContent || a.getAttribute('aria-label') || a.getAttribute('title') || '').trim()
                    }))
                    .filter(item => item.url && /^https?:\/\//i.test(item.url));
            }""")
            # Inspect every real anchor before the bounded navigation budget. A
            # long product menu must not hide the company's identity page.
            links = _prioritized_candidate_links(url, links, limit=80)

            return {"html": html, "text": text[:50000], "title": title, "links": links}
        finally:
            browser.close()


def _slugify_page_key(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:40] or "page"


# Strong identity signals are intentionally scoped to the path or anchor text,
# never the hostname (e.g. product.company.example) or a bare "overview".
_ABOUT_PATH_PRIORITIES = {
    "company-profile": 500,
    "company-introduction": 480,
    "about-us": 450,
    "who-we-are": 450,
    "about": 420,
    "aboutus": 420,
}
_ABOUT_TITLE_ZH = (
    "公司简介", "公司介绍", "公司概况", "关于我们", "关于公司",
    "企业简介", "企业介绍", "企业概况",
)
_ABOUT_TITLE_EN = re.compile(
    r"\b(?:about us|who we are|company profile|company introduction|company overview)\b",
    re.IGNORECASE,
)
_EXTRA_NAV_ASSET_EXTENSIONS = (
    ".mp4", ".webm", ".mov", ".avi", ".mp3", ".wav", ".ogg",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".eot",
)


def _about_link_priority(candidate: dict) -> int:
    path = unquote(urlparse(candidate.get("url") or "").path).strip("/").lower()
    title = re.sub(r"\s+", " ", str(candidate.get("title") or "")).strip()
    title_match = title.lower() == "about" or any(term in title for term in _ABOUT_TITLE_ZH) or bool(_ABOUT_TITLE_EN.search(title))
    priority = _ABOUT_PATH_PRIORITIES.get(path, 0)
    return priority + (30 if title_match else 0) if priority else (200 if title_match else 0)


def _prioritized_candidate_links(base_url: str, links: list[dict], *, limit: int) -> list[dict]:
    # The full same-domain/shallow/asset filter must run before either the 80
    # DOM budget or the 12-item AI budget. Only discovered URLs are eligible.
    # Prefer the informative title when the same canonical URL also appears
    # earlier as an icon/empty menu anchor; this still scans the entire input.
    ranked_anchors = sorted(links, key=_about_link_priority, reverse=True)
    candidates = build_candidate_links(base_url, ranked_anchors, limit=max(len(links), 1))
    candidates = [candidate for candidate in candidates
                  if not urlparse(candidate["url"]).path.lower().endswith(_EXTRA_NAV_ASSET_EXTENSIONS)]
    return sorted(candidates, key=_about_link_priority, reverse=True)[:limit]


async def _plan_company_pages(base_url: str, homepage_title: str, links: list[dict]) -> tuple[list[dict], list[dict]]:
    candidate_links = _prioritized_candidate_links(base_url, links, limit=12)
    try:
        from app.services.ai_client import ai_client

        selected_pages = await ai_client.select_company_pages(base_url, homepage_title, candidate_links)
        if not isinstance(selected_pages, list):
            raise ValueError("页面选择结果必须为列表")
    except Exception:
        selected_pages = fallback_select_company_pages(base_url, homepage_title, candidate_links, limit=3)

    normalized_base = normalize_company_url(base_url)
    planned_pages = [{
        "url": normalized_base,
        "title": homepage_title or "主页",
        "role": "homepage",
        "reason": "主页通常包含公司定位、产品摘要与核心导航，是企业知识库的主入口。",
    }]
    used = {normalized_base}
    candidates_by_url = {candidate["url"]: candidate for candidate in candidate_links}

    # Preserve the strongest verified identity page even if AI spends its
    # selection on products. This is an explicit rule, not a claimed AI choice.
    identity_page = next((candidate for candidate in candidate_links if _about_link_priority(candidate)), None)
    if identity_page is not None:
        planned_pages.append({
            "url": identity_page["url"],
            "title": identity_page["title"],
            "role": "about",
            "reason": "必需身份页已核实为首页发现的同域一级真实链接，优先用于公司简介与身份核验。",
        })
        used.add(identity_page["url"])

    for page in selected_pages:
        if len(planned_pages) >= 3:
            break
        if not isinstance(page, dict):
            continue
        if not isinstance(page.get("url"), str):
            continue
        try:
            page_url = normalize_company_url(page.get("url") or "")
        except (ValueError, TypeError):
            continue
        candidate = candidates_by_url.get(page_url)
        if candidate is None or page_url in used:
            continue
        page_role = page.get("role") or "supporting"
        if not isinstance(page_role, str) or page_role not in {"about", "team", "product", "supporting"}:
            page_role = "supporting"
        if page_role == "about" and not _about_link_priority(candidate):
            # A generic company hostname is not identity-page evidence.
            page_role = "supporting"
        planned_pages.append({
            "url": page_url,
            "title": candidate["title"],
            "role": page_role,
            "reason": page.get("reason") or "该页面被选入企业分析，URL已核实来自首页真实链接。",
        })
        used.add(page_url)

    return candidate_links, planned_pages


def _store_captured_html(storage, key: str, html: str) -> None:
    # Process workers are separate processes: the storage service's in-memory
    # fallback cannot transfer captured evidence to the next task.
    if not html or not storage.put(key, html.encode("utf-8", errors="replace")):
        raise RuntimeError(f"抓取页面未成功持久化，不能进入后续分析：{key}")


@shared_task(name="app.tasks.crawl.crawl_company_website", bind=True, max_retries=3)
def crawl_company_website(self, company_id: str, url: str):
    """
    爬取公司官网:
    1. Playwright 加载页面
    2. 尝试发现并爬取「关于我们」页面
    3. 将原始 HTML 持久化到 MinIO，失败则阻断后续阶段
    4. 更新 Company.pipeline_status → 'cleaning'
    5. 链式触发: clean_company_data
    """
    from app.models.company import PipelineStatus

    # Admission/read failures are not permission to run paid page selection.
    # No stale callback may change even the company's initial status.
    try:
        admitted = _run(_admit_crawl_callback(company_id, self.request.id))
    except Exception as exc:
        log_event(
            logger, logging.ERROR, "task.crawl_company.admission_unknown",
            task_id=self.request.id, company_id=company_id,
            error_type=type(exc).__name__, observe_only=True,
        )
        return {"state": "admission_unknown", "company_id": company_id}
    if not admitted:
        log_event(
            logger, logging.INFO, "task.crawl_company.stale_callback_skipped",
            task_id=self.request.id, company_id=company_id,
        )
        return {"state": "stale_callback_skipped", "company_id": company_id}

    try:
        log_event(
            logger,
            logging.INFO,
            "task.crawl_company.started",
            task_id=self.request.id,
            company_id=company_id,
            url=url,
            retries=self.request.retries,
        )
        _run(_update_company(company_id, pipeline_status=PipelineStatus.CRAWLING))

        normalized_url = normalize_company_url(url)

        # 爬取主页
        result = _crawl_page(normalized_url)
        html = result["html"]
        title = result["title"]
        links = result["links"]
        candidate_links, selected_pages = _run(_plan_company_pages(normalized_url, title, links))

        # 上传到 MinIO
        from app.services.storage import storage
        raw_key = f"companies/{company_id}/raw.html"
        _store_captured_html(storage, raw_key, html)
        crawl_pages = []
        about_key = None

        for index, page in enumerate(selected_pages):
            page_url = page.get("url") or normalized_url
            page_title = page.get("title") or title or "主页"
            page_role = page.get("role") or ("homepage" if index == 0 else "supporting")
            page_reason = page.get("reason") or "该页面被选入企业知识库分析流程。"
            page_key = raw_key if page_url == normalized_url else f"companies/{company_id}/{_slugify_page_key(page_role or page_title)}-{index + 1}.html"

            page_html = html
            status = "captured"
            if page_url != normalized_url:
                try:
                    page_result = _crawl_page(page_url)
                    page_html = page_result["html"]
                    if not page_title or page_title == "主页":
                        page_title = page_result["title"] or page_title
                except Exception as exc:
                    status = "failed"
                    page_reason = f"{page_reason} 页面抓取失败：{str(exc)[:120]}"
                    page_html = ""

            if page_html:
                if page_key != raw_key:
                    _store_captured_html(storage, page_key, page_html)
                if page_role == "about" and about_key is None and page_url != normalized_url:
                    about_key = page_key

            crawl_pages.append(
                {
                    "url": page_url,
                    "title": page_title,
                    "role": page_role,
                    "reason": page_reason,
                    "key": page_key if page_html else None,
                    "status": status,
                }
            )

        _run(
            _update_company(
                company_id,
                pipeline_status=PipelineStatus.CRAWLING,
                crawl_candidates=candidate_links,
                crawl_pages=crawl_pages,
            )
        )

        if any(page.get("role") == "about" and page.get("url") != normalized_url for page in selected_pages) and about_key is None:
            raise RuntimeError("必需公司身份页抓取失败，禁止跳过身份资料进入清洗阶段")

        # 更新状态，进入清洗阶段
        _run(_update_company(
            company_id,
            pipeline_status=PipelineStatus.CLEANING,
            raw_html_key=raw_key,
            about_html_key=about_key,
            crawl_candidates=candidate_links,
            crawl_pages=crawl_pages,
        ))

    except Exception as exc:
        logger.exception("crawl_company_website failed: %s", company_id)
        log_event(
            logger,
            logging.ERROR,
            "task.crawl_company.failed",
            task_id=self.request.id,
            company_id=company_id,
            url=url,
            retries=self.request.retries,
            error=str(exc)[:500],
        )
        try:
            _run(_update_company(
                company_id,
                pipeline_status=PipelineStatus.FAILED,
                pipeline_error=str(exc)[:500],
            ))
        except Exception:
            pass
        # Never retry the whole crawl automatically: page selection may have
        # already reached a paid model before a later capture/storage failure.
        return {"state": "failed", "company_id": company_id}

    # Downstream publish acknowledgement is a separate boundary. A broker
    # timeout does not prove the cleaning task was not queued, and therefore
    # must never replay the paid crawl/page-selection stage.
    clean_task_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"georank:company-clean:{company_id}:{self.request.id}",
    ))
    try:
        current = _run(_record_clean_dispatch(
            company_id, self.request.id, clean_task_id, "dispatching",
        ))
        if not current:
            return {"state": "stale_callback_skipped", "company_id": company_id}
        from app.core.celery_app import celery_app
        celery_app.send_task(
            "app.tasks.process.clean_company_data", args=[company_id], task_id=clean_task_id,
        )
        current = _run(_record_clean_dispatch(
            company_id, self.request.id, clean_task_id, "queued",
        ))
        if not current:
            return {"state": "stale_callback_skipped", "company_id": company_id}
    except Exception as exc:
        log_event(
            logger, logging.ERROR, "task.crawl_company.clean_dispatch_unknown",
            task_id=self.request.id, company_id=company_id,
            clean_task_id=clean_task_id, error_type=type(exc).__name__, observe_only=True,
        )
        try:
            _run(_record_clean_dispatch(company_id, self.request.id, clean_task_id, "unknown"))
        except Exception as receipt_exc:
            # Keep the already-persisted cleaning/dispatching state for read-only
            # reconciliation. Do not turn unknown delivery into FAILED/retry.
            log_event(
                logger, logging.ERROR, "task.crawl_company.clean_receipt_unknown",
                task_id=self.request.id, company_id=company_id,
                clean_task_id=clean_task_id, error_type=type(receipt_exc).__name__, observe_only=True,
            )
        return {"state": "dispatch_unknown", "company_id": company_id, "clean_task_id": clean_task_id}

    log_event(
        logger, logging.INFO, "task.crawl_company.completed",
        task_id=self.request.id, company_id=company_id, url=normalized_url,
        selected_pages=len(crawl_pages), about_page_found=bool(about_key),
        clean_task_id=clean_task_id,
    )
    return {"state": "clean_queued", "company_id": company_id, "clean_task_id": clean_task_id}


@shared_task(name="app.tasks.crawl.crawl_diagnostic_page", bind=True, max_retries=3)
def crawl_diagnostic_page(self, report_id: str, url: str):
    """
    诊断用爬虫 — 爬取单个页面:
    1. 获取完整 HTML 源码
    2. 上传到 MinIO
    3. 更新 DiagnosticReport.status → 'analyzing'
    4. 链式触发: analyze_page
    """
    from app.models.diagnostic import DiagnosticStatus

    try:
        log_event(
            logger,
            logging.INFO,
            "task.crawl_diagnostic.started",
            task_id=self.request.id,
            report_id=report_id,
            url=url,
            retries=self.request.retries,
        )
        _run(_update_report(report_id, status=DiagnosticStatus.CRAWLING))

        result = _crawl_page(url)
        html = result["html"]

        # 上传到 MinIO
        from app.services.storage import storage
        raw_key = f"diagnostics/{report_id}/raw.html"
        _store_captured_html(storage, raw_key, html)

        _run(_update_report(
            report_id,
            status=DiagnosticStatus.ANALYZING,
            raw_html_key=raw_key,
        ))

        # 链式触发分析任务
        from app.core.celery_app import celery_app
        celery_app.send_task("app.tasks.diagnose.analyze_page", args=[report_id])
        log_event(
            logger,
            logging.INFO,
            "task.crawl_diagnostic.completed",
            task_id=self.request.id,
            report_id=report_id,
            url=url,
        )

    except Exception as exc:
        logger.exception("crawl_diagnostic_page failed: %s", report_id)
        log_event(
            logger,
            logging.ERROR,
            "task.crawl_diagnostic.failed",
            task_id=self.request.id,
            report_id=report_id,
            url=url,
            retries=self.request.retries,
            error=str(exc)[:500],
        )
        try:
            _run(_update_report(
                report_id,
                status=DiagnosticStatus.FAILED,
                error_message=str(exc)[:500],
            ))
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=60)
