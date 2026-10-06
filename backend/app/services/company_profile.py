# Fork modification (he1016060110, 2026-10-06): verified company extraction and local embedding pipeline.
"""
公司资料抽取与修复服务
"""
from __future__ import annotations

from datetime import date
import hashlib
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from sqlalchemy import update

from app.models.company import Company
from app.services.ai_client import ai_client
from app.services.storage import storage
from app.services.company_source import (
    build_company_evidence,
    clean_source_text,
    prepare_company_documents,
)
from app.tasks.diagnose import (
    _check_citations,
    _check_content,
    _check_meta,
    _check_schema,
    _calculate_overall_score,
)

_KNOWN_TECH_TERMS = [
    "OpenAI API",
    "Claude",
    "DeepSeek",
    "Firecrawl",
    "Ahrefs API",
    "Qdrant",
    "Neo4j",
    "Pinecone",
    "LangChain",
    "Next.js",
    "React",
    "Postgres",
    "Redis",
    "Playwright",
    "Python",
]

_COMPANY_NAME_SPLIT_PATTERN = re.compile(r"\s*[|｜丨]\s*|\s+[—–-]\s+")
_COMPANY_NAME_NOISE_SUFFIXES = (
    "官方网站",
    "官网首页",
    "官网",
    "首页",
    "主页",
)
_COMPANY_NAME_NOISE_EXACT = {
    "官网",
    "官方网站",
    "首页",
    "主页",
    "关于我们",
    "about us",
}


def company_profile_needs_hydration(company: Company) -> bool:
    return (
        not bool((company.short_description or "").strip() or (company.description or "").strip())
        or not bool(company.tags)
        or not bool(company.tech_stack)
        or company.geo_details is None
        or company.geo_score is None
    )


def load_company_source_pages(company: Company) -> list[dict]:
    """Read stored pages with their authoritative crawl URL/role, not naked HTML."""
    page_refs = [page for page in company.crawl_pages or [] if isinstance(page, dict) and page.get("key")]
    if not page_refs:
        page_refs = [
            {"key": key, "url": company.url if role == "homepage" else None, "role": role}
            for key, role in ((company.raw_html_key, "homepage"), (company.about_html_key, "about"))
            if key
        ]
    pages, seen = [], set()
    for page in page_refs:
        key = page["key"]
        if key in seen:
            continue
        seen.add(key)
        raw = storage.get(key)
        if raw:
            pages.append({
                "html": raw.decode("utf-8", errors="replace"), "url": page.get("url") or (company.url if page.get("role") == "homepage" else None),
                "role": page.get("role"), "key": key,
            })
    return pages


def load_company_source_html(company: Company) -> str:
    """Legacy HTML callers; new extraction/graph/vector paths use source pages."""
    return "\n".join(page["html"] for page in load_company_source_pages(company))


def load_company_homepage_html(company: Company) -> str:
    homepage_key = None
    for page in company.crawl_pages or []:
        if page.get("role") == "homepage" and page.get("key"):
            homepage_key = page["key"]
            break
    homepage_key = homepage_key or company.raw_html_key
    if not homepage_key:
        return ""
    raw = storage.get(homepage_key)
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _clean_text(value: str | None, limit: int | None = None) -> str | None:
    text = re.sub(r"\s+", " ", (value or "").strip())
    if not text:
        return None
    if limit is not None:
        return text[:limit]
    return text


def _pick_first(*values: str | None, limit: int | None = None) -> str | None:
    for value in values:
        cleaned = _clean_text(value, limit=limit)
        if cleaned:
            return cleaned
    return None


def normalize_company_name(value: str | None, fallback_name: str | None = None) -> str | None:
    primary = _clean_text(value, limit=200)
    fallback = _clean_text(fallback_name, limit=200)
    if not primary:
        return fallback

    candidates = [segment.strip(" -_|｜丨—–·•:：") for segment in _COMPANY_NAME_SPLIT_PATTERN.split(primary)]
    if not candidates:
        candidates = [primary]

    normalized_candidates: list[str] = []
    for candidate in candidates:
        cleaned = _clean_text(candidate, limit=200)
        if not cleaned:
            continue
        for suffix in _COMPANY_NAME_NOISE_SUFFIXES:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
        cleaned = cleaned.strip(" -_|｜丨—–·•:：")
        if not cleaned:
            continue
        if cleaned.lower() in _COMPANY_NAME_NOISE_EXACT:
            continue
        normalized_candidates.append(cleaned)

    for candidate in normalized_candidates:
        if 1 < len(candidate) <= 60:
            return candidate

    return fallback or primary


def _identity_evidence(documents: list[dict]) -> tuple[list[dict], list[str]]:
    candidates = []
    for document in documents:
        for candidate in document["name_candidates"]:
            if candidate not in candidates:
                candidates.append(candidate)
    # legalName/body-introduction conflicts are not resolved by majority vote or
    # by knowing a customer's preferred identity. Brand/alternateName != legalName
    # is normal and is retained as evidence rather than declared a contradiction.
    legal_names = {
        candidate["value"] for candidate in candidates
        if candidate["kind"] in {"json_ld.legalName", "body.company_intro"}
    }
    warnings = ["source_company_name_conflict"] if len(legal_names) > 1 else []
    return candidates, warnings


def fallback_company_profile_from_html(
    html: str,
    fallback_name: str | None = None,
    *,
    source_pages: list[dict] | None = None,
    source_url: str | None = None,
) -> dict:
    """Deterministic fallback is explicitly degraded, never an AI success."""
    documents = prepare_company_documents(source_pages if source_pages is not None else html, source_url=source_url)
    name_candidates, warnings = _identity_evidence(documents)
    intro_candidates = []
    names = []
    tags = []
    for document in documents:
        for organization in document["organizations"]:
            name = _pick_first(organization.get("legalName"), organization.get("name"), limit=200)
            if name:
                names.append((5, name))
            description = clean_source_text(organization.get("description"))
            if description:
                intro_candidates.append((5, description, document["url"], "json_ld.description"))
        for candidate in document["name_candidates"]:
            names.append((4 if candidate["kind"] == "body.company_intro" else 3, candidate["value"]))
        if document["role"] in {None, "homepage", "about"} and document["title"]:
            names.append((1, document["title"]))
        description = _pick_first(document["meta"].get("description"), document["meta"].get("og:description"))
        if description:
            intro_candidates.append((3, description, document["url"], "meta.description"))
        for paragraph in document["paragraphs"]:
            text, context = paragraph["text"], paragraph["context"].lower()
            priority = 1
            if re.search(r"about|company|profile|intro", context) or document["role"] == "about":
                priority = 4
            if re.search(r"(?:是一家|专门从事|专注于|speciali[sz]|company.*(?:develop|design|manufactur))", text, re.I):
                priority += 1
            intro_candidates.append((priority, text, document["url"], "main.paragraph"))
        keywords = document["meta"].get("keywords") or ""
        for item in re.split(r"[,，/|]+", keywords):
            value = item.strip()[:24]
            if value and value not in tags:
                tags.append(value)
    names.sort(key=lambda item: item[0], reverse=True)
    name = normalize_company_name(names[0][1] if names else None, fallback_name=fallback_name)
    intro_candidates.sort(key=lambda item: item[0], reverse=True)
    snippets, intro_evidence = [], []
    for priority, text, url, kind in intro_candidates:
        if any(text in previous for previous in snippets):
            continue
        snippets.append(text)
        intro_evidence.append({"url": url, "kind": kind, "quote": text[:500]})
        if len("\n".join(snippets)) >= 1200 or len(snippets) >= 5:
            break
    description = _pick_first("\n".join(snippets), limit=1500)
    short_description = _pick_first(snippets[0] if snippets else None, limit=140)
    body_text = " ".join(document["text"] for document in documents)
    tech_stack = [term for term in _KNOWN_TECH_TERMS if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", body_text, re.I)]
    warnings.extend(warning for document in documents for warning in document["warnings"])
    return {
        "extraction_source": "html_fallback",
        "name": name, "description": description, "short_description": short_description,
        "category": None, "headquarters": None, "funding_stage": None,
        "employee_count": None, "founded_date": None, "tags": tags[:6],
        "tech_stack": tech_stack[:8], "team_members": [],
        "name_evidence": name_candidates,
        "field_evidence": {"description": intro_evidence},
        "extraction_quality": {
            "status": "degraded", "source": "html_fallback", "extraction_source": "html_fallback", "errors": [],
            "warnings": sorted(set(warnings + ["fallback_not_ai_verified"])),
            "source_urls": list(dict.fromkeys(document["url"] for document in documents if document["url"])),
            "document_count": len(documents), "text_chars": sum(document["text_chars"] for document in documents),
            "text_sha256": hashlib.sha256("\n\n".join(document["text"] for document in documents).encode("utf-8")).hexdigest(),
            "requires_review": True,
        },
    }


class CompanyProfileExtractionError(ValueError):
    """A fallback/invalid extraction cannot silently advance the clean stage."""

    def __init__(self, quality: dict):
        self.quality = quality
        super().__init__("Company profile extraction incomplete: " + "; ".join(quality["errors"]))


def _safe_extraction_error(exc: Exception) -> str:
    # Provider exception text may contain URLs/query credentials, headers or keys.
    # Preserve the diagnosable failure category, never persist arbitrary secrets.
    message = str(exc).lower()
    if "api base url" in message and ("不能为空" in message or "required" in message or "missing" in message):
        return "company_extraction_api_base_missing"
    if "api key" in message and ("未配置" in message or "required" in message or "missing" in message):
        return "company_extraction_api_key_missing"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "company_extraction_invalid_response"
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "company_extraction_timeout"
    return "company_extraction_provider_error"


async def extract_company_profile(
    html: str,
    fallback_name: str | None = None,
    *,
    source_pages: list[dict] | None = None,
    source_url: str | None = None,
    strict: bool = False,
) -> dict:
    documents = prepare_company_documents(source_pages if source_pages is not None else html, source_url=source_url)
    profile = fallback_company_profile_from_html(html, fallback_name=fallback_name, source_pages=documents)
    quality = profile["extraction_quality"]
    profile["quality"] = quality  # compatibility alias for pipeline receipt consumers
    errors, extracted = [], {}
    if not documents or not any(document["text"] for document in documents):
        errors.append("company_source_text_empty")
    else:
        evidence = build_company_evidence(documents)
        try:
            extracted = await ai_client.extract_company_info(evidence)
            if not isinstance(extracted, dict) or not extracted:
                errors.append("company_extraction_empty_response")
                extracted = {}
        except Exception as exc:
            errors.append(_safe_extraction_error(exc))

    text_fields = ("name", "description", "short_description", "category", "headquarters", "funding_stage", "employee_count", "founded_date")
    unknown_values = {"未知", "unknown", "n/a", "null", "none", "未披露", "not disclosed"}
    for key in text_fields:
        value = extracted.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            errors.append(f"company_extraction_invalid_{key}")
            continue
        value = clean_source_text(value)
        if value and value.lower() not in unknown_values:
            profile[key] = value
    for key, limit in (("tags", 6), ("tech_stack", 8)):
        value = extracted.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(f"company_extraction_invalid_{key}")
            continue
        if any(not isinstance(item, str) for item in value):
            errors.append(f"company_extraction_invalid_{key}")
        profile[key] = list(dict.fromkeys(clean_source_text(item, 80) for item in value if isinstance(item, str) and clean_source_text(item)))[:limit]
    if extracted.get("team_members") is not None:
        if not isinstance(extracted["team_members"], list):
            errors.append("company_extraction_invalid_team_members")
        else:
            profile["team_members"] = [
                {key: clean_source_text(item.get(key), 300) for key in ("name", "role", "bg")}
                for item in extracted["team_members"] if isinstance(item, dict) and clean_source_text(item.get("name"))
            ][:6]
    if isinstance(extracted.get("field_evidence"), dict):
        # This is model-reported attribution, not proof that a fact was verified.
        known_urls = set(quality["source_urls"])
        for field, items in extracted["field_evidence"].items():
            if field not in text_fields or not isinstance(items, list):
                continue
            validated = []
            for item in items[:5]:
                if not isinstance(item, dict) or item.get("url") not in known_urls:
                    continue
                quote = clean_source_text(item.get("quote"), 800)
                if quote and any(document["url"] == item["url"] and (quote in document["text"] or quote in str(document["structured_data"]) + str(document["meta"])) for document in documents):
                    validated.append({"url": item["url"], "quote": quote, "kind": "llm_attribution_in_source"})
            if validated:
                profile["field_evidence"][field] = validated

    profile["name"] = normalize_company_name(profile.get("name"), fallback_name=fallback_name)
    llm_description = clean_source_text(extracted.get("description"))
    if extracted and len(llm_description) < 45:
        errors.append("company_extraction_description_insufficient")
    warning_values = extracted.get("warnings") if isinstance(extracted.get("warnings"), list) else []
    warnings = [value for value in quality["warnings"] if value != "fallback_not_ai_verified"]
    for warning in warning_values[:10]:
        if isinstance(warning, str):
            warnings.append(clean_source_text(warning, 250))
    conflicts = extracted.get("source_conflicts")
    if isinstance(conflicts, list) and conflicts:
        warnings.append("model_reported_source_conflicts")
        profile["source_conflicts"] = [item for item in conflicts[:8] if isinstance(item, (str, dict))]
    if extracted.get("name"):
        proposed_name = normalize_company_name(extracted.get("name")) or ""
        source_identity_text = " ".join(
            document["title"] + " " + document["text"] + " " + str(document["meta"]) + " " + str(document["organizations"])
            for document in documents
        ).lower()
        normalized_source = re.sub(r"[\W_]", "", source_identity_text)
        normalized_proposed = re.sub(r"[\W_]", "", proposed_name.lower())
        if normalized_proposed and normalized_proposed not in normalized_source:
            warnings.append("model_company_name_not_grounded_in_sources")
            # Unsupported transliterations/brand guesses are not verified legal
            # identities. Keep the last source-derived/Owner-supplied name.
            fallback_profile = fallback_company_profile_from_html(html, fallback_name=fallback_name, source_pages=documents)
            profile["name"] = fallback_profile["name"]
    legal_candidates = {item["value"] for item in profile["name_evidence"] if item["kind"] in {"json_ld.legalName", "body.company_intro"}}
    if "source_company_name_conflict" in warnings:
        # A conflicting website is not enough authority to rename an existing
        # company. Its candidate identities remain visible for Owner review.
        profile["name"] = normalize_company_name(fallback_name) or profile["name"]
    elif extracted.get("name") and legal_candidates and profile["name"] not in legal_candidates:
        warnings.append("model_company_name_not_in_legal_source_candidates")
    profile["extraction_source"] = "llm" if extracted else "html_fallback"
    quality.update({
        "status": "failed" if errors else "passed", "source": profile["extraction_source"],
        "extraction_source": profile["extraction_source"],
        "evidence_sha256": hashlib.sha256(build_company_evidence(documents).encode("utf-8")).hexdigest(),
        "errors": list(dict.fromkeys(errors)), "warnings": sorted(set(filter(None, warnings))),
        "requires_review": bool(warnings or errors),
    })
    if errors:
        quality["warnings"] = sorted(set(quality["warnings"] + ["fallback_not_ai_verified"]))
    if strict and errors:
        raise CompanyProfileExtractionError(quality)
    return profile


def calculate_company_geo_profile(company: Company, homepage_html: str) -> dict:
    if not homepage_html:
        return {}
    soup = BeautifulSoup(homepage_html, "lxml")
    base_domain = urlparse(company.url).netloc.lower()
    schema = _check_schema(soup)
    meta = _check_meta(soup)
    content = _check_content(soup)
    citation = _check_citations(soup, base_domain)
    score = _calculate_overall_score(
        schema["score"],
        content["score"],
        meta["score"],
        citation["score"],
    )
    return {
        "geo_score": score,
        "geo_details": {
            "schema": schema["score"],
            "content": content["score"],
            "meta": meta["score"],
            "citation": citation["score"],
        },
    }


def build_company_profile_values(company: Company, profile: dict) -> dict:
    values: dict = {}
    normalized_name = normalize_company_name(profile.get("name"), fallback_name=company.name)

    if normalized_name and normalized_name != company.name:
        values["name"] = normalized_name[:200]
    if profile.get("description"):
        values["description"] = profile["description"]
    if profile.get("short_description"):
        values["short_description"] = profile["short_description"][:300]
    if profile.get("category"):
        values["category"] = profile["category"][:50]
    if profile.get("headquarters"):
        values["headquarters"] = profile["headquarters"][:200]
    if profile.get("funding_stage"):
        values["funding_stage"] = profile["funding_stage"][:50]
    if profile.get("employee_count"):
        values["employee_count"] = profile["employee_count"][:50]
    if profile.get("founded_date"):
        try:
            founded_date = str(profile["founded_date"]).strip()
            if len(founded_date) == 7:
                founded_date += "-01"
            values["founded_date"] = date.fromisoformat(founded_date)
        except Exception:
            pass
    if profile.get("tags") is not None:
        values["tags"] = list(profile["tags"])[:6]
    if profile.get("tech_stack") is not None:
        values["tech_stack"] = list(profile["tech_stack"])[:8]
    if profile.get("team_members") is not None:
        values["team_members"] = list(profile["team_members"])[:6]
    if profile.get("geo_details"):
        values["geo_details"] = profile["geo_details"]
    if isinstance(profile.get("extraction_quality"), dict):
        details = dict(values.get("geo_details") or company.geo_details or {})
        details["extraction_quality"] = {
            **profile["extraction_quality"],
            "name_evidence": profile.get("name_evidence", []),
            "field_evidence": profile.get("field_evidence", {}),
            "source_conflicts": profile.get("source_conflicts", []),
        }
        values["geo_details"] = details
    if profile.get("geo_score") is not None:
        values["geo_score"] = float(profile["geo_score"])

    return values


async def ensure_company_profile(db, company: Company, *, force: bool = False) -> dict:
    if not force and not company_profile_needs_hydration(company):
        return {}

    pages = load_company_source_pages(company)
    html = "\n".join(page["html"] for page in pages)
    if not html:
        return {}

    profile = await extract_company_profile(html, fallback_name=company.name, source_pages=pages, source_url=company.url)
    profile.update(calculate_company_geo_profile(company, load_company_homepage_html(company)))
    values = build_company_profile_values(company, profile)
    if not values:
        return {}

    await db.execute(
        update(Company)
        .where(Company.id == company.id)
        .values(**values)
    )
    await db.commit()
    await db.refresh(company)
    return values
