# Fork modification (he1016060110, 2026-10-06): verified company extraction and local embedding pipeline.
"""Company-scoped, versioned Neo4j graph storage with retained history.

An entity name is not a global identity.  Each current snapshot gets company- and
version-scoped entity keys.  Replacing a company snapshot archives its bindings,
not the old entities or semantic edges, and cannot change another company graph.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlsplit
from uuid import UUID

from neo4j import AsyncGraphDatabase
from app.core.config import settings

_driver = AsyncGraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
)

_SCHEMA_VERSION = 2
_INTERNAL_RELATIONS = {"HAS_ENTITY", "HAS_ENTITY_HISTORY"}
_RESERVED_PROPERTIES = {
    "entity_key", "identity_key", "company_id", "graph_version", "schema_version",
    "normalized_name", "normalized_type", "entity_type", "name", "run_id",
    "source_urls", "source_hashes", "source_hashes_json", "source_sha256",
    "provenance_scope", "content_sha256", "updated_at", "created_at",
    "relation_key", "active", "binding_id",
}


class GraphPersistenceError(RuntimeError):
    """The graph readback does not match the requested snapshot; rollback it."""


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _company_uuid(company_id: str) -> str:
    try:
        return str(UUID(str(company_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Graph company_id must be a UUID") from exc


def _text(value, field: str, *, max_length: int = 1024) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Graph {field} must be text")
    value = " ".join(unicodedata.normalize("NFKC", value).split())
    if not value or len(value) > max_length:
        raise ValueError(f"Invalid graph {field}")
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in value):
        raise ValueError(f"Invalid graph {field} encoding")
    return value


def _entity_type(value) -> tuple[str, str]:
    value = _text(value or "Entity", "entity type", max_length=64)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError("Invalid graph entity type")
    normalized = value.casefold()
    standard = {name.casefold(): name for name in (
        "Company", "Product", "Technology", "Person", "Organization", "Entity",
        "Application", "Industry", "Location", "Certification", "Material", "Service",
    )}
    return standard.get(normalized, value), normalized


def _props(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Graph props must be an object")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or key in _RESERVED_PROPERTIES:
            continue
        if item is None or isinstance(item, (str, bool, int)):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, (list, tuple)) and all(
            isinstance(part, (str, bool, int, float))
            and not (isinstance(part, float) and not math.isfinite(part))
            for part in item
        ):
            # Neo4j property lists must be homogeneous; other values are retained
            # as explicit JSON text instead of inventing an unsupported property.
            if len({type(part) for part in item}) <= 1:
                result[key] = list(item)
            else:
                result[key + "_json"] = _json(list(item))
        elif isinstance(item, (dict, list, tuple)):
            result[key + "_json"] = _json(item)
        else:
            raise ValueError(f"Unsupported graph property: {key}")
    return result


def _source_metadata(source_urls=None, source_hashes=None, source_sha256=None) -> dict:
    if source_urls is None:
        source_urls = []
    if not isinstance(source_urls, (list, tuple)):
        raise ValueError("Graph source_urls must be a list")
    urls = set()
    for url in source_urls:
        if not isinstance(url, str):
            raise ValueError("Graph source URL must be text")
        url = url.strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Graph source URL must be an unauthenticated HTTP(S) URL")
        urls.add(url)
    if source_hashes is None:
        hash_rows = []
    elif isinstance(source_hashes, dict):
        hash_rows = [{"url": url, "sha256": digest} for url, digest in source_hashes.items()]
    elif isinstance(source_hashes, (list, tuple)):
        hash_rows = list(source_hashes)
    else:
        raise ValueError("Graph source_hashes must be a URL/hash mapping or list")
    hashes = {}
    for row in hash_rows:
        if not isinstance(row, dict):
            raise ValueError("Graph source hash must contain url and sha256")
        url, digest = row.get("url"), row.get("sha256")
        if url not in urls:
            raise ValueError("Graph source hash URL is not in the supplied source_urls")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            raise ValueError("Graph source hash must be a real SHA-256 value")
        digest = digest.lower()
        if url in hashes and hashes[url] != digest:
            raise ValueError("Conflicting graph source hashes")
        hashes[url] = digest
    if source_sha256 is not None and (
        not isinstance(source_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", source_sha256)
    ):
        raise ValueError("Graph source_sha256 must be a SHA-256 value")
    # These are caller-supplied source references, not a claim that this service
    # fetched or verified original bytes. Never manufacture missing source hashes.
    return {
        "source_urls": sorted(urls),
        "source_hashes": sorted(set(hashes.values())),
        "source_hashes_json": _json([{"url": url, "sha256": hashes[url]} for url in sorted(hashes)]),
        "source_sha256": source_sha256.lower() if source_sha256 else "",
    }


def _claim_metadata(row: dict, corpus: dict) -> dict:
    """Preserve direct quote/URL aliases and match hashes to their real corpus page."""
    if "source_urls" in row:
        urls = row["source_urls"]
    elif row.get("source_url"):
        urls = [row["source_url"]]
    else:
        urls = corpus["source_urls"]
    # The crawler hashes were already normalized. A direct citation to a subset
    # must not inherit unrelated page hashes, or fabricate a new hash for a URL.
    default_hashes = [item for item in json.loads(corpus["source_hashes_json"])
                      if item["url"] in urls]
    metadata = _source_metadata(urls, row.get("source_hashes", default_hashes),
                                row.get("source_sha256", corpus["source_sha256"] or None))
    if corpus["source_urls"] and not set(metadata["source_urls"]).issubset(corpus["source_urls"]):
        raise ValueError("Graph claim source URL is outside the supplied analysis corpus")
    metadata["provenance_scope"] = ("claim_reference" if "source_urls" in row or row.get("source_url")
                                    else "analysis_corpus")
    return metadata


def _claim_properties(row: dict) -> dict:
    props = _props(row.get("props"))
    for field in ("description", "source_quote", "quote", "evidence", "evidence_quote", "confidence"):
        if field in row and field not in props:
            props.update(_props({field: row[field]}))
    return props

def _prepare_snapshot(company_id, entities, relations, source_urls, run_id, source_hashes, source_sha256):
    company_id = _company_uuid(company_id)
    if not isinstance(entities, list) or not entities:
        raise ValueError("A company graph requires non-empty entities")
    if not isinstance(relations, list):
        raise ValueError("Graph relations must be a list")
    if run_id is not None:
        run_id = _text(run_id, "run_id", max_length=200)
    metadata = _source_metadata(source_urls, source_hashes, source_sha256)
    by_identity = {}
    for row in entities:
        if not isinstance(row, dict):
            raise ValueError("Graph entity must be an object")
        name = _text(row.get("name"), "entity name")
        entity_type, normalized_type = _entity_type(row.get("type"))
        normalized_name = name.casefold()
        identity_key = f"{company_id}:{normalized_type}:{normalized_name}"
        props = _claim_properties(row)
        claim_sources = _claim_metadata(row, metadata)
        entity = {
            "identity_key": identity_key, "name": name, "entity_type": entity_type,
            "normalized_name": normalized_name, "normalized_type": normalized_type,
            "props": props, **claim_sources,
        }
        previous = by_identity.get(identity_key)
        if previous:
            # Same normalized type/name is one node, not one node per LLM row.
            previous["props"].update({key: value for key, value in props.items() if value not in (None, "")})
        else:
            by_identity[identity_key] = entity
    prepared_entities = [by_identity[key] for key in sorted(by_identity)]
    names = {}
    for entity in prepared_entities:
        names.setdefault(entity["normalized_name"], []).append(entity)

    def endpoint(row, side):
        value = row.get(side)
        endpoint_type = row.get(side + "_type")
        if isinstance(value, dict):
            endpoint_type = value.get("type", endpoint_type)
            value = value.get("name")
        candidates = names.get(_text(value, f"relation {side}").casefold(), [])
        if endpoint_type:
            _, normalized_type = _entity_type(endpoint_type)
            candidates = [item for item in candidates if item["normalized_type"] == normalized_type]
        if len(candidates) != 1:
            raise ValueError(f"Graph relation {side} endpoint is absent or ambiguous in this company snapshot")
        return candidates[0]["identity_key"]

    prepared_relations = {}
    for row in relations:
        if not isinstance(row, dict):
            raise ValueError("Graph relation must be an object")
        relation_type = _text(row.get("type"), "relation type", max_length=64).upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", relation_type) or relation_type in _INTERNAL_RELATIONS:
            raise ValueError("Invalid or reserved graph relationship type")
        relation = {
            "from_identity": endpoint(row, "from"), "to_identity": endpoint(row, "to"),
            "type": relation_type, "props": _claim_properties(row),
            **_claim_metadata(row, metadata),
        }

        identity = (relation["from_identity"], relation_type, relation["to_identity"])
        prepared_relations[identity] = relation
    prepared_relations = [prepared_relations[key] for key in sorted(prepared_relations)]
    version = "v2-" + _sha({
        "company_id": company_id, "run_id": run_id or "", "sources": metadata,
        "entities": prepared_entities, "relations": prepared_relations,
    })
    for entity in prepared_entities:
        entity["entity_key"] = f"{entity['identity_key']}:{version}"
        entity["content_sha256"] = _sha({key: value for key, value in entity.items() if key != "entity_key"})
    entity_keys = {row["identity_key"]: row["entity_key"] for row in prepared_entities}
    for relation in prepared_relations:
        relation["from_key"] = entity_keys[relation.pop("from_identity")]
        relation["to_key"] = entity_keys[relation.pop("to_identity")]
        relation["relation_key"] = f"{company_id}:{version}:" + _sha({
            "from": relation["from_key"], "type": relation["type"], "to": relation["to_key"],
        })
        relation["content_sha256"] = _sha(relation)
    return company_id, version, run_id or "", metadata, prepared_entities, prepared_relations


_READ_CURRENT = """
// graph-store:read-current
MATCH (c:Company {id: $company_id})
OPTIONAL MATCH (c)-[binding:HAS_ENTITY]->(n:GraphEntity)
WHERE binding.active = true AND binding.company_id = $company_id
  AND binding.graph_version = c.graph_version
  AND n.company_id = $company_id AND n.graph_version = c.graph_version
WITH c, collect(DISTINCT n) AS entity_nodes
CALL {
    WITH c, entity_nodes
    UNWIND entity_nodes AS a
    OPTIONAL MATCH (a)-[r]->(b:GraphEntity)
    WHERE b IN entity_nodes AND r.company_id = c.id AND r.graph_version = c.graph_version
    RETURN collect(DISTINCT CASE WHEN r IS NOT NULL THEN {
        relation_key: r.relation_key, type: type(r), from_key: a.entity_key,
        to_key: b.entity_key, properties: properties(r)
    } END) AS relationships
}
RETURN properties(c) AS company,
       [node IN entity_nodes | properties(node)] AS entities, relationships
"""


async def _read_current(tx, company_id: str) -> dict:
    result = await tx.run(_READ_CURRENT, company_id=company_id)
    record = await result.single()
    company = dict(record["company"] or {}) if record else {}
    # Also deduplicate at the API boundary. DB record/path multiplicity cannot
    # inflate node counts, and a malformed or foreign entry is never accepted.
    entities = {}
    for row in (record["entities"] or []) if record else []:
        item = dict(row)
        if (item.get("company_id") != company_id or item.get("graph_version") != company.get("graph_version")
                or not item.get("entity_key")):
            continue
        entities[item["entity_key"]] = item
    relationships = {}
    for row in (record["relationships"] or []) if record else []:
        item = dict(row)
        props = dict(item.get("properties") or {})
        if (item.get("from_key") not in entities or item.get("to_key") not in entities
                or props.get("company_id") != company_id or props.get("graph_version") != company.get("graph_version")):
            continue
        key = item.get("relation_key") or props.get("relation_key")
        if key:
            relationships[key] = item
    entity_list = [entities[key] for key in sorted(entities)]
    relation_list = [relationships[key] for key in sorted(relationships)]
    return {
        "company_id": company_id, "graph_version": company.get("graph_version"),
        "run_id": company.get("graph_run_id", ""), "schema_version": company.get("graph_schema_version"),
        "nodes": len(entity_list), "node_count": len(entity_list), "relation_count": len(relation_list),
        "entities": entity_list, "relationships": relation_list,
        "source_urls": company.get("graph_source_urls", []),
        "source_hashes": company.get("graph_source_hashes", []),
        "source_hashes_json": company.get("graph_source_hashes_json", "[]"),
        "source_sha256": company.get("graph_source_sha256", ""),
    }


def _company_properties(properties) -> dict:
    safe = {key: value for key, value in _props(properties).items()
            if key != "id" and not key.startswith("graph_")}
    # `name` is reserved on an entity so it cannot override its normalized
    # identity, but is a legitimate display property on the Company root.
    if properties and "name" in properties:
        safe["name"] = _text(properties["name"], "company name")
    return safe

async def create_company_node(company_id: str, properties: dict):
    """Legacy helper for company display properties; never modifies graph history."""
    company_id = _company_uuid(company_id)
    safe_properties = _company_properties(properties)

    async def write(tx):
        result = await tx.run(
            "MERGE (c:Company {id: $id}) SET c += $props",
            id=company_id, props=safe_properties,
        )
        await result.consume()

    async with _driver.session() as session:
        await session.execute_write(write)


async def _write_snapshot(tx, company_id, version, run_id, metadata, entities, relations, updated_at, properties):
    # A write lock on the one Company node serializes this company's snapshots.
    # Every statement, including readback, is in this one managed transaction.
    result = await tx.run(
        """
        // graph-store:company-lock
        MERGE (c:Company {id: $company_id})
        SET c.graph_write_revision = coalesce(c.graph_write_revision, 0) + 1
        SET c += $props
        RETURN c.id AS company_id
        """,
        company_id=company_id, props=properties,
    )
    await result.consume()
    result = await tx.run(
        """
        // graph-store:archive-bindings
        MATCH (c:Company {id: $company_id})-[binding:HAS_ENTITY]->(old)
        WHERE coalesce(binding.graph_version, '') <> $graph_version
        MERGE (c)-[history:HAS_ENTITY_HISTORY {
            binding_id: coalesce(binding.binding_id, $company_id + ':legacy:' + elementId(binding))
        }]->(old)
        SET history += properties(binding), history.active = false,
            history.archived_at = $updated_at, history.archived_by_run_id = $run_id,
            history.replaced_by_graph_version = $graph_version
        DELETE binding
        RETURN count(history) AS archived_bindings
        """,
        company_id=company_id, graph_version=version, updated_at=updated_at, run_id=run_id,
    )
    record = await result.single()
    archived = record["archived_bindings"] if record else 0
    for entity in entities:
        result = await tx.run(
            """
            // graph-store:upsert-entity
            MATCH (c:Company {id: $company_id})
            MERGE (e:GraphEntity {entity_key: $entity_key})
            SET e += $props
            SET e.identity_key = $identity_key, e.name = $name, e.entity_type = $entity_type,
                e.normalized_name = $normalized_name, e.normalized_type = $normalized_type,
                e.company_id = $company_id, e.graph_version = $graph_version,
                e.run_id = $run_id, e.schema_version = $schema_version,
                e.source_urls = $source_urls, e.source_hashes = $source_hashes,
                e.source_hashes_json = $source_hashes_json, e.source_sha256 = $source_sha256,
                e.provenance_scope = $provenance_scope, e.content_sha256 = $content_sha256,
                e.updated_at = $updated_at
            MERGE (c)-[binding:HAS_ENTITY {binding_id: $entity_key}]->(e)
            SET binding.company_id = $company_id, binding.graph_version = $graph_version,
                binding.run_id = $run_id, binding.active = true, binding.updated_at = $updated_at
            """,
            **entity, company_id=company_id, graph_version=version, run_id=run_id,
            schema_version=_SCHEMA_VERSION, updated_at=updated_at,
        )
        await result.consume()
    for relation in relations:
        # Only the validated relationship identifier is interpolated. Endpoints
        # are full keys scoped to this snapshot; never MATCH by a global name.
        result = await tx.run(
            f"""
            // graph-store:upsert-relation
            MATCH (a:GraphEntity {{entity_key: $from_key, company_id: $company_id, graph_version: $graph_version}})
            MATCH (b:GraphEntity {{entity_key: $to_key, company_id: $company_id, graph_version: $graph_version}})
            MERGE (a)-[r:{relation['type']} {{relation_key: $relation_key}}]->(b)
            SET r += $props
            SET r.company_id = $company_id, r.graph_version = $graph_version, r.run_id = $run_id,
                r.schema_version = $schema_version, r.source_urls = $source_urls,
                r.source_hashes = $source_hashes, r.source_hashes_json = $source_hashes_json,
                r.source_sha256 = $source_sha256, r.provenance_scope = $provenance_scope,
                r.content_sha256 = $content_sha256, r.updated_at = $updated_at
            """,
            **relation, company_id=company_id, graph_version=version, run_id=run_id,
            schema_version=_SCHEMA_VERSION, updated_at=updated_at,
        )
        await result.consume()
    result = await tx.run(
        """
        // graph-store:activate-version
        MATCH (c:Company {id: $company_id})
        SET c.graph_version = $graph_version, c.graph_run_id = $run_id,
            c.graph_schema_version = $schema_version, c.graph_updated_at = $updated_at,
            c.graph_source_urls = $source_urls, c.graph_source_hashes = $source_hashes,
            c.graph_source_hashes_json = $source_hashes_json, c.graph_source_sha256 = $source_sha256
        """,
        company_id=company_id, graph_version=version, run_id=run_id,
        schema_version=_SCHEMA_VERSION, updated_at=updated_at, **metadata,
    )
    await result.consume()
    readback = await _read_current(tx, company_id)
    if (readback["graph_version"] != version or readback["node_count"] != len(entities)
            or readback["relation_count"] != len(relations)):
        raise GraphPersistenceError("Company graph readback does not match the requested snapshot")
    return {
        "company_id": company_id, "graph_version": version, "run_id": run_id,
        "node_count": readback["node_count"], "relation_count": readback["relation_count"],
        "archived_bindings": archived, "persisted": True, "verified": True,
        "readback": readback, **metadata,
    }


async def upsert_company_entities(company_id: str, entities: list[dict], relations: list[dict],
                                  source_urls=None, run_id=None, source_hashes=None,
                                  source_sha256=None, properties=None) -> dict:
    """Atomically replace this company's current graph, retain old snapshots.

    Optional provenance must be passed by the real crawler/process; empty values
    stay empty rather than being presented as verified source/run information.
    Replaying the same run and content gives the same keys, without duplication.
    """
    company_id, version, run_id, metadata, entities, relations = _prepare_snapshot(
        company_id, entities, relations, source_urls, run_id, source_hashes, source_sha256,
    )
    safe_properties = _company_properties(properties)
    updated_at = datetime.now(timezone.utc).isoformat()
    async with _driver.session() as session:
        return await session.execute_write(
            _write_snapshot, company_id, version, run_id, metadata,
            entities, relations, updated_at, safe_properties,
        )


async def add_entities_and_relations(company_id: str, entities: list[dict], relations: list[dict],
                                     source_urls=None, run_id=None, source_hashes=None,
                                     source_sha256=None) -> dict:
    """Compatible old entry point; new writes obey the same namespace/history contract."""
    return await upsert_company_entities(
        company_id, entities, relations, source_urls, run_id, source_hashes, source_sha256,
    )


async def get_company_graph(company_id: str) -> dict:
    """Return only this company's active snapshot and true distinct entity counts.

    ``nodes`` remains the legacy integer field. ``entities`` and ``relationships``
    contain bounded-to-this-company records, with stable counts and provenance.
    Legacy global-name entities/relationships are retained but never traversed.
    """
    company_id = _company_uuid(company_id)
    async with _driver.session() as session:
        return await session.execute_read(_read_current, company_id)
