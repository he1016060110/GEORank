# Fork modification (he1016060110, 2026-10-06): verified company extraction and local embedding pipeline.
"""
公司 API — 提交 / 列表 / 详情 / 投票 / 进度 / 相似推荐
"""
import asyncio
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select, func, update, text
from sqlalchemy.exc import IntegrityError

from app.core.deps import DbSession, CurrentUser, OptionalUser
from app.models.company import Company, PublishStatus, PipelineStatus
from app.models.vote import CompanyVote
from app.services.company_lookup import get_company_by_identifier
from app.services.company_profile import company_profile_needs_hydration, ensure_company_profile
from app.services.company_ingest import normalize_company_url
from app.services.ai_usage import resolve_async_ai_access
from app.schemas.company import (
    SubmitCompanyRequest, SubmitCompanyResponse, CompanyBrief, CompanyDetail,
    PaginatedCompanies, PipelineStatusResponse, VoteResponse, SimilarCompanyItem,
)

router = APIRouter()

PUBLIC_ACTIVE_PIPELINE_STATUSES = {
    PipelineStatus.PENDING, PipelineStatus.CRAWLING, PipelineStatus.CLEANING,
    PipelineStatus.GRAPH_BUILDING, PipelineStatus.VECTORIZING,
}
PUBLIC_DISPATCH_TIMEOUT_SECONDS = 8.0


def _public_dispatch_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_dispatch_metadata(company) -> dict:
    details = company.geo_details if isinstance(company.geo_details, dict) else {}
    dispatch = details.get("pipeline_dispatch")
    return dispatch if isinstance(dispatch, dict) else {}


def _public_url_lock_key(url: str) -> int:
    # Transaction-scoped Postgres lock also serializes the absent-row/create case.
    # Stable signed bigint; no process-local hash or database schema migration.
    return int.from_bytes(hashlib.sha256(("georank-company-submit:" + url).encode("utf-8")).digest()[:8], "big", signed=True)


async def _record_public_pipeline_dispatch(db: DbSession, company_id, task_id: str, dispatch_state: str) -> None:
    """Reread under row lock so a late broker receipt cannot erase worker progress."""
    result = await db.execute(
        select(Company).where(Company.id == company_id).with_for_update()
        .execution_options(populate_existing=True)
    )
    company = result.scalar_one_or_none()
    if not company:
        await db.rollback()
        return
    details = dict(company.geo_details) if isinstance(company.geo_details, dict) else {}
    dispatch = dict(_public_dispatch_metadata(company))
    if dispatch.get("task_id") != task_id:
        await db.rollback()
        return
    dispatch.update(state=dispatch_state, receipt_updated_at=_public_dispatch_time())
    details["pipeline_dispatch"] = dispatch
    values = {"geo_details": details}
    if dispatch_state == "unknown" and company.pipeline_status == PipelineStatus.PENDING:
        values["pipeline_error"] = "任务派发结果尚未确认，请观察当前任务状态，不要重复发起分析。"
    await db.execute(update(Company).where(Company.id == company_id).values(**values))
    await db.commit()


async def _dispatch_public_company(db: DbSession, company_id, company_url: str, task_id: str) -> str:
    try:
        from app.core.celery_app import celery_app
        await asyncio.wait_for(asyncio.to_thread(
            celery_app.send_task, "app.tasks.crawl.crawl_company_website",
            args=[str(company_id), company_url], task_id=task_id, retry=False,
        ), timeout=PUBLIC_DISPATCH_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        # Cancelling a wait does not cancel the publishing thread or prove non-dispatch.
        await asyncio.shield(_record_public_pipeline_dispatch(db, company_id, task_id, "unknown"))
        raise
    except Exception:
        await _record_public_pipeline_dispatch(db, company_id, task_id, "unknown")
        return "unknown"
    await _record_public_pipeline_dispatch(db, company_id, task_id, "submitted")
    return "submitted"


@router.post("/submit", response_model=SubmitCompanyResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_company(data: SubmitCompanyRequest, db: DbSession, current_user: OptionalUser):
    """Admit at most one company run; uncertain broker outcomes are observe-only."""
    try:
        normalized_url = normalize_company_url(data.url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Row locks alone cannot protect a URL whose company has not been created yet.
    await db.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {
        "lock_key": _public_url_lock_key(normalized_url),
    })
    result = await db.execute(
        select(Company).where(Company.url == normalized_url).with_for_update()
        .execution_options(populate_existing=True)
    )
    existing = result.scalar_one_or_none()
    if existing:
        company_id = existing.id
        publication = existing.publish_status.value
        if existing.publish_status == PublishStatus.PUBLISHED:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"该 URL 已存在，company_id: {company_id}",
            )
        dispatch = _public_dispatch_metadata(existing)
        if (existing.pipeline_status in PUBLIC_ACTIVE_PIPELINE_STATUSES
                or dispatch.get("state") in {"admitted", "pending", "unknown"}
                or existing.pipeline_status != PipelineStatus.FAILED):
            saved_status = existing.pipeline_status.value
            await db.rollback()
            unknown = dispatch.get("state") == "unknown"
            return SubmitCompanyResponse(
                company_id=str(company_id), status="dispatch_unknown" if unknown else saved_status,
                message="派发回执尚未确认，只恢复同一任务观察，请勿重复创建。" if unknown
                    else "已存在同域名分析任务，正在恢复分析进度。",
                normalized_url=normalized_url, publish_status=publication, resumed=True,
                task_id=dispatch.get("task_id"), dispatch_state=dispatch.get("state"), observe_only=True,
            )

        await resolve_async_ai_access(db=db, current_user=current_user, module="companies", prompt_text=normalized_url)
        task_id = str(uuid.uuid4())
        details = dict(existing.geo_details) if isinstance(existing.geo_details, dict) else {}
        if details.get("pipeline_quality"):
            details["pipeline_previous_quality"] = details["pipeline_quality"]
        details["pipeline_quality"] = None
        details["pipeline_dispatch"] = {"task_id": task_id, "state": "pending", "admitted_at": _public_dispatch_time()}
        await db.execute(update(Company).where(Company.id == company_id).values(
            pipeline_status=PipelineStatus.PENDING, pipeline_error=None, geo_details=details,
            submitted_by=current_user.id if current_user else existing.submitted_by,
        ))
        # Preserve saved source artifacts and scores; the new worker owns replacing its results.
        await db.commit()
        resumed = True
    else:
        await resolve_async_ai_access(db=db, current_user=current_user, module="companies", prompt_text=normalized_url)
        task_id = str(uuid.uuid4())
        company_id = uuid.uuid4()
        publication = PublishStatus.DRAFT.value
        company = Company(
            id=company_id, name=normalized_url.split("//")[-1].split("/")[0], url=normalized_url,
            pipeline_status=PipelineStatus.PENDING, publish_status=PublishStatus.DRAFT,
            submitted_by=current_user.id if current_user else None,
            geo_details={"pipeline_dispatch": {"task_id": task_id, "state": "pending", "admitted_at": _public_dispatch_time()}},
        )
        db.add(company)
        await db.commit()
        resumed = False

    dispatch_state = await _dispatch_public_company(db, company_id, normalized_url, task_id)
    return SubmitCompanyResponse(
        company_id=str(company_id), status="dispatch_unknown" if dispatch_state == "unknown" else "pending",
        message="派发回执尚未确认，任务可能已进入队列；请观察同一任务，不要重试创建。" if dispatch_state == "unknown"
            else "已重新加入处理队列" if resumed else "已加入处理队列",
        normalized_url=normalized_url, publish_status=publication, resumed=resumed,
        task_id=task_id, dispatch_state=dispatch_state, observe_only=True,
    )


@router.get("/", response_model=PaginatedCompanies)
async def list_companies(
    db: DbSession,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    sort: str = Query("newest", pattern="^(newest|geo_score|upvotes)$"),
    q: Optional[str] = None,
):
    """公司列表 — 支持分类筛选、排序、全文搜索"""
    query = select(Company).where(Company.publish_status == PublishStatus.PUBLISHED)

    if category:
        query = query.where(Company.category == category)
    if q:
        query = query.where(
            Company.name.ilike(f"%{q}%") | Company.short_description.ilike(f"%{q}%")
        )

    # 排序
    if sort == "geo_score":
        query = query.order_by(Company.geo_score.desc().nullslast())
    elif sort == "upvotes":
        query = query.order_by(Company.upvotes.desc())
    else:
        query = query.order_by(Company.created_at.desc())

    # 总数
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    # 分页
    result = await db.execute(query.offset((page - 1) * size).limit(size))
    companies = result.scalars().all()

    items = []
    for c in companies:
        items.append(CompanyBrief(
            id=str(c.id),
            path_key=c.path_key,
            name=c.name,
            url=c.url,
            logo_url=c.logo_url,
            short_description=c.short_description,
            category=c.category,
            tags=c.tags if isinstance(c.tags, list) else [],
            geo_score=c.geo_score,
            is_geo_certified=c.is_geo_certified,
            tech_level=c.tech_level,
            funding_stage=c.funding_stage,
            headquarters=c.headquarters,
            pipeline_status=c.pipeline_status.value,
            publish_status=c.publish_status.value,
            upvotes=c.upvotes,
        ))

    return PaginatedCompanies(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size if total > 0 else 1,
    )


@router.get("/{company_id}", response_model=CompanyDetail)
async def get_company(company_id: str, db: DbSession):
    """公司详情 — 含完整知识库信息"""
    company = await get_company_by_identifier(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="公司不存在")

    if company.pipeline_status == PipelineStatus.COMPLETED and company_profile_needs_hydration(company):
        try:
            await ensure_company_profile(db, company)
        except Exception:
            pass

    return CompanyDetail(
        id=str(company.id),
        path_key=company.path_key,
        name=company.name,
        url=company.url,
        logo_url=company.logo_url,
        short_description=company.short_description,
        description=company.description,
        category=company.category,
        tags=company.tags if isinstance(company.tags, list) else [],
        geo_score=company.geo_score,
        geo_details=company.geo_details,
        is_geo_certified=company.is_geo_certified,
        tech_level=company.tech_level,
        tech_stack=company.tech_stack if isinstance(company.tech_stack, list) else [],
        team_members=company.team_members if isinstance(company.team_members, list) else [],
        funding_stage=company.funding_stage,
        headquarters=company.headquarters,
        employee_count=company.employee_count,
        founded_date=str(company.founded_date) if company.founded_date else None,
        pipeline_status=company.pipeline_status.value,
        publish_status=company.publish_status.value,
        pipeline_error=company.pipeline_error,
        upvotes=company.upvotes,
    )


@router.post("/{company_id}/upvote", response_model=VoteResponse)
async def upvote_company(company_id: str, db: DbSession, current_user: CurrentUser):
    """
    投票（幂等）
    - 写入 company_votes（UNIQUE company_id+user_id）
    - 若已投票返回 HTTP 409
    - 成功则 companies.upvotes 原子递增
    """
    company = await get_company_by_identifier(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="公司不存在")
    cid = company.id

    vote = CompanyVote(company_id=cid, user_id=current_user.id)
    db.add(vote)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="已投过票")

    # 原子递增
    await db.execute(
        update(Company).where(Company.id == cid).values(upvotes=Company.upvotes + 1)
    )
    await db.commit()

    result = await db.execute(select(Company.upvotes).where(Company.id == cid))
    upvotes = result.scalar_one()
    return VoteResponse(upvotes=upvotes)


def _saved_pipeline_quality(company) -> dict | None:
    """Return only the saved receipt; polling must never hydrate or call providers."""
    details = company.geo_details if isinstance(company.geo_details, dict) else {}
    quality = details.get("pipeline_quality")
    return quality if isinstance(quality, dict) else None


def _pipeline_stage_verified(name: str, stage) -> bool:
    if not isinstance(stage, dict) or stage.get("status") not in {"complete", "passed"}:
        return False
    if name == "crawl":
        return isinstance(stage.get("document_count"), (int, float)) and stage["document_count"] > 0
    if stage.get("verified") is not True:
        return False
    if name == "clean":
        return True
    count_key = {"graph": "entity_count", "vector": "vector_count"}.get(name)
    return bool(count_key and isinstance(stage.get(count_key), (int, float)) and stage[count_key] > 0)


def _pipeline_quality_passed(quality: dict | None) -> bool:
    if not quality or quality.get("status") not in {"complete", "passed"}:
        return False
    stages = quality.get("stages")
    if not isinstance(stages, dict):
        return False
    return all(_pipeline_stage_verified(name, stages.get(name) or (
        stages.get("profile") if name == "clean" else None
    )) for name in ("crawl", "clean", "graph", "vector"))


@router.get("/{company_id}/pipeline-status", response_model=PipelineStatusResponse)
async def get_pipeline_status(company_id: str, db: DbSession):
    """查询入库流水线当前进度（前端轮询）"""
    company = await get_company_by_identifier(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="公司不存在")

    progress_map = {
        PipelineStatus.PENDING: 0,
        PipelineStatus.CRAWLING: 20,
        PipelineStatus.CLEANING: 40,
        PipelineStatus.GRAPH_BUILDING: 65,
        PipelineStatus.VECTORIZING: 85,
        PipelineStatus.COMPLETED: 100,
        PipelineStatus.FAILED: 0,
    }

    crawl_pages = []
    for page in company.crawl_pages or []:
        crawl_pages.append(
            {
                "url": page.get("url"),
                "title": page.get("title"),
                "role": page.get("role"),
                "reason": page.get("reason"),
                "status": page.get("status"),
            }
        )

    current_activity = {
        PipelineStatus.PENDING: "已创建任务，等待进入官网解析队列。",
        PipelineStatus.CRAWLING: (
            "已解析首页一级目录，正在抓取 AI 选出的关键页面。"
            if crawl_pages
            else "正在抓取官网首页，并从一级目录里识别最值得深入的页面。"
        ),
        PipelineStatus.CLEANING: "已完成关键页面抓取，正在提取企业介绍、产品与团队信息。",
        PipelineStatus.GRAPH_BUILDING: "正在梳理实体关系并构建企业知识图谱。",
        PipelineStatus.VECTORIZING: "正在将企业知识写入语义检索索引。",
        PipelineStatus.COMPLETED: "企业知识库构建完成，已进入审核队列。",
        PipelineStatus.FAILED: company.pipeline_error or "本次知识库构建未成功完成。",
    }.get(company.pipeline_status)

    quality = _saved_pipeline_quality(company)
    quality_passed = _pipeline_quality_passed(quality) and not company.pipeline_error
    progress = progress_map.get(company.pipeline_status, 0)
    if company.pipeline_status == PipelineStatus.COMPLETED:
        if not quality_passed:
            progress = 0
            stages = quality.get("stages", {}) if quality else {}
            quality_failed = bool(company.pipeline_error) or bool(quality and (
                quality.get("status") in {"failed", "degraded"}
                or isinstance(stages, dict) and any(
                    isinstance(stage, dict) and stage.get("status") in {"failed", "degraded"}
                    for stage in stages.values()
                )
            ))
            current_activity = (
                company.pipeline_error or "企业知识库质量核验未通过，请在后台检查失败阶段。"
                if quality_failed else "历史记录标为完成，但企业资料、图谱和向量成果尚无完整核验回执。"
            )
        elif company.publish_status == PublishStatus.PENDING_REVIEW:
            current_activity = "企业资料、图谱和向量成果已核验，资料已提交后台审核。"
        elif company.publish_status == PublishStatus.PUBLISHED:
            current_activity = "企业知识库成果已核验，该公司已审核发布。"
        else:
            current_activity = "企业知识库成果已核验，当前仍为草稿，等待确认提交审核。"

    dispatch = _public_dispatch_metadata(company)
    if company.pipeline_status == PipelineStatus.PENDING and dispatch.get("state") == "unknown":
        current_activity = "任务派发结果尚未确认，仅观察同一任务，不要重复提交或创建新任务。"

    return PipelineStatusResponse(
        company_id=str(company.id),
        status=company.pipeline_status.value,
        progress=progress,
        error=company.pipeline_error,
        current_activity=current_activity,
        publish_status=company.publish_status.value,
        company_name=company.name,
        company_summary=company.short_description,
        company_url=company.url,
        pipeline_quality=quality,
        pipeline_dispatch=dispatch or None,
        selected_pages=crawl_pages,
    )


@router.post("/{company_id}/submit-review")
async def submit_company_for_review(company_id: str, db: DbSession, current_user: OptionalUser):
    """用户确认分析结果后，提交后台审核。"""
    company = await get_company_by_identifier(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="公司不存在")
    cid = company.id

    if company.publish_status == PublishStatus.PUBLISHED:
        return {
            "status": "published",
            "company_id": company_id,
            "message": "该公司已发布。",
        }

    if company.pipeline_status != PipelineStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="请等待分析完成后再提交审核。",
        )

    if company.pipeline_error or not _pipeline_quality_passed(_saved_pipeline_quality(company)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="企业资料、图谱和向量成果尚未通过质量核验，请先重新分析。",
        )

    # Review is an explicit state transition, not an implicit paid hydration path.
    if company_profile_needs_hydration(company):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="企业资料尚未抽取完整，请重新运行分析。",
        )

    update_values = {
        "publish_status": PublishStatus.PENDING_REVIEW,
    }
    if current_user and not company.submitted_by:
        update_values["submitted_by"] = current_user.id

    await db.execute(update(Company).where(Company.id == cid).values(**update_values))
    await db.commit()

    return {
        "status": "pending_review",
        "company_id": company_id,
        "message": "公司资料已提交审核，审核通过后将在前台展示。",
    }


@router.get("/{company_id}/similar", response_model=list[SimilarCompanyItem])
async def get_similar_companies(company_id: str, db: DbSession, top_k: int = 3):
    """
    相似公司推荐 — 先尝试 Qdrant 向量检索，
    向量库为空时降级为同类别随机推荐（保证接口始终有数据）
    """
    company = await get_company_by_identifier(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="公司不存在")
    cid = company.id

    # 尝试向量检索
    similar_ids = []
    try:
        from app.services.vector_store import vector_store
        similar_ids = await vector_store.get_similar_company_ids(str(cid), top_k=top_k + 1)
        similar_ids = [sid for sid in similar_ids if sid != str(cid)][:top_k]
    except Exception:
        pass

    if similar_ids:
        uuids = [uuid.UUID(sid) for sid in similar_ids]
        result = await db.execute(
            select(Company).where(Company.id.in_(uuids), Company.publish_status == PublishStatus.PUBLISHED)
        )
        companies = result.scalars().all()
    else:
        from app.services.company_retrieval import fallback_similar_companies

        companies = await fallback_similar_companies(db, company, limit=top_k)

    return [
        SimilarCompanyItem(
            id=str(c.id),
            path_key=c.path_key,
            name=c.name,
            short_description=c.short_description,
            logo_url=c.logo_url,
            geo_score=c.geo_score,
            category=c.category,
        )
        for c in companies
    ]
