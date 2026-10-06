# Fork modification (he1016060110, 2026-10-06): verified company extraction and local embedding pipeline.
"""Company processing. Every required artifact must be durably verified."""
import enum
import hashlib
import json
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone

from celery import shared_task
from app.core.logging_utils import log_event
from app.tasks.runtime import run_async as _run

logger = logging.getLogger("georank.process")
POINT_NAMESPACE = uuid.UUID("12ae5a2f-2c3e-5ba4-96ed-a6b7423be97e")


class PipelineArtifactError(ValueError):
    pass


def _normalize_update_values(values: dict) -> dict:
    return {key: value.value if isinstance(value, enum.Enum) else value for key, value in values.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Bound by characters, including unspaced Chinese, for the 512-token local model."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size/overlap")
    text = str(text or "").strip()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap) if text[i:i + chunk_size].strip()]


def _quality(company, *, reset: bool = False) -> dict:
    details = company.geo_details if isinstance(company.geo_details, dict) else {}
    old = details.get("pipeline_quality", {})
    if reset or not isinstance(old, dict) or not old.get("run_id"):
        return {
            "run_id": str(uuid.uuid4()), "status": "running", "producedAt": _now(),
            "stages": {}, "source_urls": [], "entity_count": 0, "vector_count": 0,
        }
    return {**old, "stages": dict(old.get("stages") or {})}


async def _write_stage(db, company, quality: dict, stage: str, status: str, *, values=None, **facts):
    from sqlalchemy import update
    from app.models.company import Company
    quality["stages"][stage] = {"status": status, "producedAt": _now(), **facts}
    quality["producedAt"] = _now()
    details = dict(company.geo_details) if isinstance(company.geo_details, dict) else {}
    payload = dict(values or {})
    if isinstance(payload.get("geo_details"), dict):
        details.update(payload.pop("geo_details"))
    details["pipeline_quality"] = quality
    payload["geo_details"] = details
    await db.execute(update(Company).where(Company.id == company.id).values(**_normalize_update_values(payload)))
    await db.commit()
    # Core updates do not reliably refresh already-loaded JSONB across all test/session modes.
    if hasattr(company, "_sa_instance_state"):
        from sqlalchemy.orm.attributes import set_committed_value
        set_committed_value(company, "geo_details", details)
    else:
        company.geo_details = details


def _failure_code(stage: str, exc: Exception) -> str:
    # Raw provider exceptions can contain query credentials or headers. Persist a category,
    # not the exception string. Our own gate messages contain no secrets.
    if isinstance(exc, PipelineArtifactError):
        return str(exc)[:240]
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return f"{stage}_timeout"
    return f"{stage}_{type(exc).__name__}"


async def _fail_stage(db, company, quality, stage, exc, **facts):
    from app.models.company import PipelineStatus
    await db.rollback()
    await db.refresh(company)
    code = _failure_code(stage, exc)
    quality["status"] = "failed"
    quality["failure"] = {"stage": stage, "code": code, "producedAt": _now()}
    await _write_stage(
        db, company, quality, stage, "failed",
        values={"pipeline_status": PipelineStatus.FAILED, "pipeline_error": f"{stage}: {code}"},
        error=code, **facts,
    )


def _read_sources(company):
    from app.services.company_profile import load_company_source_pages
    from app.services.company_source import prepare_company_documents
    pages = load_company_source_pages(company)
    expected_keys = {page["key"] for page in company.crawl_pages or [] if isinstance(page, dict) and page.get("key")}
    if not expected_keys:
        expected_keys = {key for key in (company.raw_html_key, company.about_html_key) if key}
    if not expected_keys or expected_keys != {page["key"] for page in pages}:
        raise PipelineArtifactError("source_saved_html_missing")
    documents = prepare_company_documents(pages)
    if not documents or not any(len(document.get("text") or "") >= 45 for document in documents):
        raise PipelineArtifactError("source_main_text_empty")
    return pages, documents


def _source_facts(documents):
    return {
        "source_urls": list(dict.fromkeys(doc.get("url") for doc in documents if doc.get("url"))),
        "document_count": len(documents),
        "sources": [{"url": doc.get("url"), "html_sha256": doc.get("html_sha256"), "text_chars": len(doc.get("text") or "")} for doc in documents],
    }


def _require_source_identity(quality, documents):
    previous = {item.get("url"): item.get("html_sha256") for item in quality.get("sources") or []}
    current = {item.get("url"): item.get("html_sha256") for item in documents}
    if not previous or previous != current:
        raise PipelineArtifactError("source_changed_after_extraction")


def _require_stage(quality, stage):
    if quality.get("stages", {}).get(stage, {}).get("status") != "complete":
        raise PipelineArtifactError(f"required_stage_{stage}_not_verified")


async def _run_clean(company_id: str):
    from app.core.database import async_session
    from app.models.company import Company, PipelineStatus
    from app.services.company_profile import build_company_profile_values, extract_company_profile
    from sqlalchemy import select
    async with async_session() as db:
        company = (await db.execute(select(Company).where(Company.id == uuid.UUID(company_id)))).scalar_one_or_none()
        if not company:
            raise PipelineArtifactError("company_not_found")
        quality = _quality(company, reset=True)
        try:
            pages, documents = _read_sources(company)
            source_facts = _source_facts(documents)
            quality.update(source_facts)
            await _write_stage(db, company, quality, "crawl", "complete", verified=True, **source_facts)
            profile = await extract_company_profile(
                "\n".join(page["html"] for page in pages), fallback_name=company.name,
                source_pages=pages, source_url=company.url, strict=True,
            )
            extraction = profile.get("extraction_quality") or {}
            if extraction.get("source") != "llm" or extraction.get("status") != "passed":
                raise PipelineArtifactError("profile_not_llm_verified")
            if len((profile.get("description") or "").strip()) < 45:
                raise PipelineArtifactError("profile_description_insufficient")
            values = build_company_profile_values(company, profile)
            values.update(pipeline_status=PipelineStatus.GRAPH_BUILDING, pipeline_error=None)
            await _write_stage(
                db, company, quality, "clean", "complete", values=values,
                extraction_quality=extraction, name_evidence=profile.get("name_evidence") or [],
                field_evidence=profile.get("field_evidence") or {},
                source_conflicts=profile.get("source_conflicts") or [],
                verified=True,
            )
            from app.core.celery_app import celery_app
            celery_app.send_task("app.tasks.process.build_knowledge_graph", args=[company_id])
            return quality
        except Exception as exc:
            await _fail_stage(db, company, quality, "clean", exc,
                              extraction_quality=getattr(exc, "quality", {}))
            raise


def _match_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or "")).casefold()


def _prepare_graph_data(data: dict, source_text: str, company_name: str, *, documents=None) -> tuple[list, list, dict]:
    """Literal source membership is entity evidence; a semantic edge also needs a quote.

    No global name-based MATCH, invented name aliases or shared entity ownership.
    Ambiguous duplicate names and unsupported/unquoted relationships are not asserted.
    """
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise PipelineArtifactError("graph_response_invalid")
    entities, by_name, ambiguous = [], {}, set()
    source = _match_text(source_text)
    separate_sources = [_match_text(doc["text"]) for doc in documents] if documents else [source]
    for node in data["nodes"][:100]:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name") or "").strip()[:200]
        kind = str(node.get("type") or "Entity").strip()
        norm = _match_text(name)
        if len(norm) < 2 or norm not in source or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", kind):
            continue
        if norm in by_name:
            if by_name[norm]["type"] != kind:
                ambiguous.add(norm)
            continue
        quote = str(node.get("evidence") or node.get("source_quote") or "").strip()[:800]
        if not quote or _match_text(quote) not in source:
            quote = name
        entity = {"name": name, "type": kind, "source_quote": quote}
        by_name[norm] = entity
        entities.append(entity)
    entities = [entity for entity in entities if _match_text(entity["name"]) not in ambiguous]
    by_name = {_match_text(entity["name"]): entity for entity in entities}
    if not entities:
        raise PipelineArtifactError("graph_no_source_grounded_entities")
    company_norm = _match_text(company_name)
    if company_norm and company_norm in source and company_norm not in ambiguous:
        existing = by_name.get(company_norm)
        if existing is None:
            by_name[company_norm] = {"name": company_name, "type": "Company", "root": True}
        elif existing["type"].casefold() != "company":
            # A same-name Product/Person is not authority to infer a Company alias.
            by_name.pop(company_norm, None)
    relations, seen = [], set()
    raw_relations = data.get("relationships") or []
    if not isinstance(raw_relations, list):
        raise PipelineArtifactError("graph_relationships_invalid")
    for rel in raw_relations[:200]:
        if not isinstance(rel, dict):
            continue
        from_key, to_key = _match_text(str(rel.get("from") or "")), _match_text(str(rel.get("to") or ""))
        kind = str(rel.get("type") or "").upper()
        quote = rel.get("evidence") or rel.get("source_quote") or ""
        if isinstance(quote, dict):
            quote = quote.get("quote") or ""
        quote = str(quote).strip()[:800]
        norm_quote = _match_text(quote)
        identity = (from_key, to_key, kind)
        if (from_key not in by_name or to_key not in by_name or identity in seen
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", kind)
            or len(norm_quote) < 12 or not any(norm_quote in text for text in separate_sources)
            or from_key not in norm_quote or to_key not in norm_quote):
            continue
        relations.append({"from": by_name[from_key], "to": by_name[to_key], "type": kind, "source_quote": quote})
        seen.add(identity)
    return entities, relations, {"dropped_relationships": len(raw_relations) - len(relations), "ambiguous_entity_names": sorted(ambiguous)}


async def _persist_graph(company, entities: list, relations: list, run_id: str, *, documents: list) -> dict:
    from app.services.graph_store import upsert_company_entities, get_company_graph
    facts = _source_facts(documents)
    source_hashes = [{"url": row["url"], "sha256": row["html_sha256"]}
                     for row in facts["sources"] if row["url"] and row["html_sha256"]]
    by_name = {_match_text(entity["name"]): dict(entity) for entity in entities}
    for relation in relations:
        for endpoint in (relation["from"], relation["to"]):
            if endpoint.get("root") and _match_text(endpoint["name"]) not in by_name:
                by_name[_match_text(endpoint["name"])] = {
                    "name": endpoint["name"], "type": "Company", "source_quote": relation["source_quote"],
                }
    def claim_sources(quote):
        matching = [doc["url"] for doc in documents if doc.get("url") and _match_text(quote) in _match_text(doc["text"])]
        return matching
    store_entities = []
    for entity in by_name.values():
        urls = claim_sources(entity["source_quote"])
        store_entities.append({
            "name": entity["name"], "type": entity["type"],
            "props": {"evidence": entity["source_quote"]}, "source_urls": urls,
            "source_hashes": [row for row in source_hashes if row["url"] in urls],
        })
    store_relations = []
    for relation in relations:
        urls = claim_sources(relation["source_quote"])
        store_relations.append({
            "from": {key: relation["from"][key] for key in ("name", "type")},
            "to": {key: relation["to"][key] for key in ("name", "type")},
            "type": relation["type"], "evidence": relation["source_quote"],
            "source_urls": urls, "source_hashes": [row for row in source_hashes if row["url"] in urls],
        })
    receipt = await upsert_company_entities(
        str(company.id), store_entities, store_relations,
        properties={"name": company.name, "url": company.url, "category": company.category or ""},
        source_urls=facts["source_urls"], run_id=run_id, source_hashes=source_hashes,
    )
    if not receipt.get("verified") or not receipt.get("persisted"):
        raise PipelineArtifactError("graph_artifacts_not_verified")
    # Managed writer verifies inside its transaction; read again after commit so an
    # unknown commit or later active snapshot cannot be mistaken for this run.
    observed = await get_company_graph(str(company.id))
    if (observed.get("graph_version") != receipt.get("graph_version")
        or observed.get("run_id") != run_id
        or observed.get("node_count") != len(store_entities)
        or observed.get("relation_count") != len(store_relations)):
        raise PipelineArtifactError("graph_committed_readback_mismatch")
    return {
        "verified": True, "entity_count": observed["node_count"],
        "relationship_count": observed["relation_count"], "run_id": run_id,
        "graph_version": receipt["graph_version"], "archived_bindings": receipt.get("archived_bindings", 0),
    }


async def _run_graph(company_id: str):
    from app.core.database import async_session
    from app.models.company import Company, PipelineStatus
    from app.services.ai_client import ai_client
    from app.services.company_source import build_company_evidence
    from sqlalchemy import select
    async with async_session() as db:
        company = (await db.execute(select(Company).where(Company.id == uuid.UUID(company_id)))).scalar_one_or_none()
        if not company:
            raise PipelineArtifactError("company_not_found")
        quality = _quality(company)
        try:
            _require_stage(quality, "clean")
            _, documents = _read_sources(company)
            _require_source_identity(quality, documents)
            data = await ai_client.extract_entities(build_company_evidence(documents))
            source_text = "\n".join(doc["text"] for doc in documents)
            entities, relations, review = _prepare_graph_data(data, source_text, company.name, documents=documents)
            receipt = await _persist_graph(company, entities, relations, quality["run_id"], documents=documents)
            if not receipt.get("verified") or receipt.get("entity_count", 0) <= 0:
                raise PipelineArtifactError("graph_artifacts_not_verified")
            quality["entity_count"] = receipt["entity_count"]
            await _write_stage(
                db, company, quality, "graph", "complete",
                values={"pipeline_status": PipelineStatus.VECTORIZING, "pipeline_error": None},
                **receipt, **review,
            )
            from app.core.celery_app import celery_app
            celery_app.send_task("app.tasks.process.vectorize_knowledge_base", args=[company_id])
            return quality
        except Exception as exc:
            await _fail_stage(db, company, quality, "graph", exc)
            raise


def _vector_chunks(company, documents: list[dict], identity: str) -> list[dict]:
    sources = [{"text": company.description or "", "url": company.url, "role": "company_profile", "html_sha256": ""}]
    share = max(400, 18000 // max(1, len(documents)))
    sources.extend({**doc, "text": doc["text"][:share]} for doc in documents)
    points = []
    for source in sources:
        for source_index, text in enumerate(_chunk_text(source["text"])):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            logical_id = f"vector|{company.id}|{identity}|{source.get('url')}|{source.get('role')}|{source_index}|{digest}"
            points.append({
                "id": str(uuid.uuid5(POINT_NAMESPACE, logical_id)), "text": text,
                "metadata": {"chunk_index": len(points), "source_chunk_index": source_index,
                             "source_url": source.get("url"), "source_role": source.get("role"),
                             "source_html_sha256": source.get("html_sha256"), "category": company.category or ""},
            })
    if not points:
        raise PipelineArtifactError("vector_source_chunks_empty")
    return points


async def _require_current_graph_artifacts(company_id: str, quality: dict):
    from app.services.graph_store import get_company_graph
    stage = quality.get("stages", {}).get("graph", {})
    graph = await get_company_graph(company_id)
    if (not stage.get("graph_version")
        or graph.get("graph_version") != stage["graph_version"]
        or graph.get("run_id") != quality.get("run_id")
        or graph.get("node_count", 0) != stage.get("entity_count")
        or graph.get("node_count", 0) <= 0
        or graph.get("relation_count") != stage.get("relationship_count")):
        raise PipelineArtifactError("graph_current_artifacts_no_longer_verified")


async def _run_vectorize(company_id: str):
    from app.core.database import async_session
    from app.models.company import Company, PipelineStatus, PublishStatus
    from app.services.ai_client import ai_client
    from app.services.runtime_settings import get_ai_runtime_config
    from app.services.vector_store import vector_store, resolve_vector_contract, validate_vector
    from sqlalchemy import select
    async with async_session() as db:
        company = (await db.execute(select(Company).where(Company.id == uuid.UUID(company_id)))).scalar_one_or_none()
        if not company:
            raise PipelineArtifactError("company_not_found")
        quality = _quality(company)
        try:
            _require_stage(quality, "clean")
            _require_stage(quality, "graph")
            await _require_current_graph_artifacts(company_id, quality)
            _, documents = _read_sources(company)
            _require_source_identity(quality, documents)
            config = await get_ai_runtime_config()
            contract = resolve_vector_contract(config)
            vector_store.ensure_collection(config)
            points = _vector_chunks(company, documents, contract.identity)
            for offset in range(0, len(points), 20):
                batch = points[offset:offset + 20]
                vectors = await ai_client.embed_batch([point["text"] for point in batch], runtime_config=config)
                if not isinstance(vectors, list) or len(vectors) != len(batch):
                    raise PipelineArtifactError("embedding_response_count_mismatch")
                for point, vector in zip(batch, vectors):
                    point["vector"] = validate_vector(vector, contract.dimensions)
            # A settings edit during embedding cannot mix identities within one upsert.
            fresh = resolve_vector_contract(await get_ai_runtime_config())
            if fresh != contract:
                raise PipelineArtifactError("embedding_runtime_changed_during_run")
            receipt = vector_store.upsert_company_vectors(company_id, points, runtime_config=config)
            if not receipt.get("verified") or receipt.get("vector_count") != len(points):
                raise PipelineArtifactError("vector_artifacts_not_verified")
            await _require_current_graph_artifacts(company_id, quality)
            quality["vector_count"] = receipt["vector_count"]
            quality["status"] = "complete"
            quality.pop("failure", None)
            await _write_stage(
                db, company, quality, "vector", "complete",
                values={"pipeline_status": PipelineStatus.COMPLETED, "pipeline_error": None,
                        "publish_status": PublishStatus.PENDING_REVIEW},
                **receipt,
            )
            # This is an artifact receipt, not fabricated provider token/cost usage.
            # AI request usage belongs to actual provider responses, not chunk counts.
            return quality
        except Exception as exc:
            await _fail_stage(db, company, quality, "vector", exc)
            raise


def _execute_stage(task, company_id, stage, operation):
    log_event(logger, logging.INFO, f"task.{stage}.started", task_id=task.request.id,
              company_id=company_id, retries=task.request.retries)
    try:
        _run(operation(company_id))
    except Exception as exc:
        code = _failure_code(stage, exc)
        log_event(logger, logging.ERROR, f"task.{stage}.failed", task_id=task.request.id,
                  company_id=company_id, error=code)
        # Never auto-repeat a paid LLM stage or enqueue vectorization after graph failure.
        raise
    log_event(logger, logging.INFO, f"task.{stage}.completed", task_id=task.request.id, company_id=company_id)


@shared_task(name="app.tasks.process.clean_company_data", bind=True)
def clean_company_data(self, company_id: str):
    return _execute_stage(self, company_id, "clean_company", _run_clean)


@shared_task(name="app.tasks.process.build_knowledge_graph", bind=True)
def build_knowledge_graph(self, company_id: str):
    return _execute_stage(self, company_id, "build_graph", _run_graph)


@shared_task(name="app.tasks.process.vectorize_knowledge_base", bind=True)
def vectorize_knowledge_base(self, company_id: str):
    return _execute_stage(self, company_id, "vectorize_company", _run_vectorize)

@shared_task(name="app.tasks.process.re_diagnose_all")
def re_diagnose_all():
    """定时任务: 每 30 天重新诊断所有已收录公司"""
    async def _run_all():
        from app.core.database import async_session
        from app.models.company import Company, PublishStatus, PipelineStatus
        from sqlalchemy import select
        async with async_session() as db:
            result = await db.execute(
                select(Company).where(Company.publish_status == PublishStatus.PUBLISHED)
            )
            companies = result.scalars().all()

        from app.core.celery_app import celery_app
        for c in companies:
            celery_app.send_task("app.tasks.crawl.crawl_company_website", args=[str(c.id), c.url])

    try:
        _run(_run_all())
    except Exception as e:
        logger.exception("re_diagnose_all failed: %s", e)
        log_event(logger, logging.ERROR, "task.re_diagnose_all.failed", error=str(e)[:500])
