# Fork modification (he1016060110, 2026-10-06): verified company extraction and local embedding pipeline.
"""Qdrant storage with model-isolated collections and verified company replacement."""
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Optional

from app.core.config import settings

COLLECTION = settings.QDRANT_COLLECTION
DIM = settings.EMBEDDING_DIMENSIONS


@dataclass(frozen=True)
class VectorContract:
    collection: str
    dimensions: int
    model: str
    provider: str

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            f"{self.provider}|{self.model}|{self.dimensions}".encode("utf-8")
        ).hexdigest()


def resolve_vector_contract(runtime_config: Optional[dict] = None) -> VectorContract:
    config = runtime_config or {}
    model = str(config.get("embedding_model") or settings.EMBEDDING_MODEL).strip()
    provider = str(config.get("embedding_provider") or "remote").strip()
    dimensions = config.get("embedding_dimensions", settings.EMBEDDING_DIMENSIONS)
    if isinstance(dimensions, bool) or (isinstance(dimensions, float) and not dimensions.is_integer()):
        raise ValueError("Embedding dimensions must be a positive integer")
    try:
        dimensions = int(dimensions)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding dimensions must be a positive integer") from exc
    if dimensions <= 0 or dimensions > 65536:
        raise ValueError("Embedding dimensions must be in 1..65536")
    collection = str(config.get("embedding_collection") or "").strip()
    if not collection:
        if model == settings.EMBEDDING_MODEL and dimensions == settings.EMBEDDING_DIMENSIONS:
            collection = settings.QDRANT_COLLECTION
        elif model == "intfloat/multilingual-e5-small" and dimensions == 384:
            collection = "companies_e5_small_v1"
        else:
            # A new model/size never mutates or recreates the legacy collection.
            suffix = hashlib.sha256(f"{provider}|{model}".encode()).hexdigest()[:12]
            collection = f"companies_embedding_{suffix}_{dimensions}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", collection):
        raise ValueError("Invalid embedding collection name")
    if collection == settings.QDRANT_COLLECTION and (
        model != settings.EMBEDDING_MODEL or dimensions != settings.EMBEDDING_DIMENSIONS
    ):
        raise ValueError("A different embedding model requires a separate collection")
    if collection == "companies_e5_small_v1" and (
        model != "intfloat/multilingual-e5-small" or dimensions != 384
    ):
        raise ValueError("companies_e5_small_v1 is reserved for the 384-dimensional multilingual-e5-small model")
    return VectorContract(collection, dimensions, model, provider)


def validate_vector(vector, dimensions: int) -> list[float]:
    if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
        raise ValueError(f"Embedding dimension mismatch: expected {dimensions}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise ValueError("Embedding values must be finite numbers")
    result = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Embedding values must be finite numbers")
    if not any(value != 0 for value in result):
        raise ValueError("Embedding vector must not be all zero")
    return result


class VectorStore:
    """Lazy client; every operation binds an explicit model/collection/size contract."""

    def __init__(self):
        self._client = None
        self._client_signature = None

    def _get_client(self, runtime_config=None):
        contract = resolve_vector_contract(runtime_config)
        signature = (settings.QDRANT_HOST, settings.QDRANT_PORT, contract.collection, contract.identity)
        if self._client is None or self._client_signature != signature:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=signature[0], port=signature[1], timeout=5)
            self._client_signature = signature
        return self._client

    def ensure_collection(self, runtime_config: Optional[dict] = None) -> VectorContract:
        """Create if absent, then read back the actual size; never recreate existing data."""
        from qdrant_client.models import Distance, VectorParams
        from qdrant_client.http.exceptions import UnexpectedResponse
        contract = resolve_vector_contract(runtime_config)
        client = self._get_client(runtime_config)
        collections = [c.name for c in client.get_collections().collections]
        if contract.collection not in collections:
            try:
                client.create_collection(
                    collection_name=contract.collection,
                    vectors_config=VectorParams(size=contract.dimensions, distance=Distance.COSINE),
                )
            except UnexpectedResponse as exc:
                # A race is only successful after the same collection is verified below.
                if exc.status_code != 409 or "already exists" not in str(exc).lower():
                    raise
        actual = client.get_collection(collection_name=contract.collection)
        vectors = actual.config.params.vectors
        if isinstance(vectors, dict) or getattr(vectors, "size", None) != contract.dimensions:
            raise ValueError(
                f"Qdrant collection {contract.collection} dimension mismatch: "
                f"expected {contract.dimensions}, observed {getattr(vectors, 'size', 'named/unknown')}"
            )
        if str(getattr(vectors, "distance", "")).lower() not in ("cosine", "distance.cosine"):
            raise ValueError("Qdrant collection distance must be Cosine")
        return contract

    @staticmethod
    def _company_filter(company_id: str):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        return Filter(must=[FieldCondition(key="company_id", match=MatchValue(value=company_id))])

    def upsert_company_vectors(self, company_id: str, chunks: list[dict], *, runtime_config=None) -> dict:
        """Upsert+read back first, then delete only this company's obsolete point IDs."""
        from qdrant_client.models import Filter, HasIdCondition, PointStruct
        if not company_id or not chunks:
            raise ValueError("Company vector replacement requires nonempty company and chunks")
        contract = self.ensure_collection(runtime_config)
        client = self._get_client(runtime_config)
        points = []
        expected = {}
        for chunk in chunks:
            point_id = chunk["id"]
            text = str(chunk.get("text") or "").strip()
            if not text or str(point_id) in expected:
                raise ValueError("Company vectors require nonempty text and unique stable IDs")
            content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            payload = {
                **chunk.get("metadata", {}),
                "company_id": company_id,
                "text": text,
                "text_sha256": content_sha256,
                "embedding_model": contract.model,
                "embedding_dimensions": contract.dimensions,
                "embedding_identity": contract.identity,
            }
            points.append(PointStruct(
                id=point_id, vector=validate_vector(chunk["vector"], contract.dimensions), payload=payload,
            ))
            expected[str(point_id)] = payload
        update = client.upsert(collection_name=contract.collection, points=points, wait=True)
        if str(getattr(update, "status", "")).lower() not in ("completed", "updatestatus.completed"):
            raise RuntimeError("Qdrant upsert completion is unknown; do not declare success")
        stored = client.retrieve(
            collection_name=contract.collection,
            ids=[point.id for point in points], with_payload=True, with_vectors=True,
        )
        if {str(point.id) for point in stored} != set(expected) or len(stored) != len(expected):
            raise RuntimeError("Qdrant readback did not return every newly written company vector")
        for point in stored:
            wanted = expected[str(point.id)]
            payload = point.payload or {}
            if any(payload.get(key) != wanted[key] for key in (
                "company_id", "text", "text_sha256", "embedding_identity", "embedding_dimensions"
            )):
                raise RuntimeError("Qdrant company vector readback content/ownership mismatch")
            validate_vector(point.vector, contract.dimensions)
        # Cleanup happens only after every new point is durable and verified. Filter ownership
        # is enforced by Qdrant, not by an untrusted/stale list of IDs from another company.
        selector = Filter(
            must=self._company_filter(company_id).must,
            must_not=[HasIdCondition(has_id=[point.id for point in points])],
        )
        deleted = client.delete(collection_name=contract.collection, points_selector=selector, wait=True)
        if str(getattr(deleted, "status", "")).lower() not in ("completed", "updatestatus.completed"):
            raise RuntimeError("Qdrant obsolete company vector cleanup is unknown")
        count = self.count_company_vectors(company_id, runtime_config=runtime_config)
        if count != len(points):
            raise RuntimeError(f"Qdrant company vector count mismatch: expected {len(points)}, observed {count}")
        return {
            "verified": True, "vector_count": count, "collection": contract.collection,
            "dimensions": contract.dimensions, "model": contract.model,
            "embedding_identity": contract.identity, "point_ids": list(expected),
        }

    def count_company_vectors(self, company_id: str, *, runtime_config=None) -> int:
        contract = resolve_vector_contract(runtime_config)
        return self._get_client(runtime_config).count(
            collection_name=contract.collection, count_filter=self._company_filter(company_id), exact=True,
        ).count

    def search_companies(self, query_vector: list[float], top_k: int = 5,
                         category: Optional[str] = None, *, runtime_config=None) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        contract = self.ensure_collection(runtime_config)
        query_vector = validate_vector(query_vector, contract.dimensions)
        filters = [FieldCondition(key="embedding_identity", match=MatchValue(value=contract.identity))]
        if category:
            filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
        results = self._get_client(runtime_config).search(
            collection_name=contract.collection, query_vector=query_vector,
            limit=top_k, query_filter=Filter(must=filters),
        )
        return [{
            "company_id": r.payload.get("company_id"), "text": r.payload.get("text"),
            "score": r.score,
            "metadata": {k: v for k, v in r.payload.items() if k not in ("company_id", "text")},
        } for r in results]

    async def get_similar_company_ids(self, company_id: str, top_k: int = 3) -> list[str]:
        from app.services.runtime_settings import get_ai_runtime_config
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        config = await get_ai_runtime_config()
        contract = self.ensure_collection(config)
        client = self._get_client(config)
        filters = self._company_filter(company_id).must + [
            FieldCondition(key="embedding_identity", match=MatchValue(value=contract.identity))
        ]
        points, _ = client.scroll(
            collection_name=contract.collection, scroll_filter=Filter(must=filters),
            with_vectors=True, limit=100,
        )
        if not points:
            return []
        vectors = [validate_vector(p.vector, contract.dimensions) for p in points]
        centroid = [sum(column) / len(vectors) for column in zip(*vectors)]
        hits = self.search_companies(centroid, top_k=top_k + 20, runtime_config=config)
        similar = []
        for hit in hits:
            cid = hit.get("company_id")
            if cid and cid != company_id and cid not in similar:
                similar.append(cid)
                if len(similar) >= top_k:
                    break
        return similar

    def delete_company_vectors(self, company_id: str, *, runtime_config=None):
        if not company_id:
            raise ValueError("An exact company ID is required")
        contract = resolve_vector_contract(runtime_config)
        return self._get_client(runtime_config).delete(
            collection_name=contract.collection,
            points_selector=self._company_filter(company_id), wait=True,
        )


vector_store = VectorStore()
