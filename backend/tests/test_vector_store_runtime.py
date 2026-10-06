"""No real Qdrant/AI/DB calls: strict runtime and durable replacement fixtures."""
import asyncio
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from qdrant_client.http.exceptions import UnexpectedResponse
from app.services.vector_store import VectorStore, resolve_vector_contract, validate_vector

LOCAL = {
    "embedding_provider": "local_tei", "embedding_model": "intfloat/multilingual-e5-small",
    "embedding_dimensions": 384, "embedding_collection": "companies_e5_small_v1",
}


def _matches(point, selector):
    payload = point.payload
    def check(condition):
        if hasattr(condition, "has_id"):
            return str(point.id) in {str(item) for item in condition.has_id}
        return payload.get(condition.key) == condition.match.value
    return all(check(condition) for condition in selector.must or []) and not any(
        check(condition) for condition in selector.must_not or []
    )


class MemoryQdrant:
    def __init__(self, size=384):
        self.size = size
        self.points = {}
        self.events = []
        self.present = True
        self.upsert_error = None
        self.readback_error = None
        self.readback_empty = False
        self.readback_foreign = False
        self.delete_unknown = False
        self.operation_unknown = False
        self.race = False

    def get_collections(self):
        return NS(collections=[NS(name=LOCAL["embedding_collection"])] if self.present else [])

    def create_collection(self, **kwargs):
        self.events.append(("create", kwargs))
        self.present = True
        if self.race:
            raise UnexpectedResponse(409, "Conflict", b'{"status":{"error":"Collection already exists"}}', None)
        self.size = kwargs["vectors_config"].size

    def get_collection(self, **kwargs):
        self.events.append(("get_collection", kwargs))
        return NS(config=NS(params=NS(vectors=NS(size=self.size, distance="Cosine"))))

    def upsert(self, **kwargs):
        self.events.append(("upsert", kwargs))
        if self.upsert_error:
            raise self.upsert_error
        for point in kwargs["points"]:
            self.points[str(point.id)] = NS(id=point.id, vector=point.vector, payload=copy.deepcopy(point.payload))
        return NS(status="acknowledged" if self.operation_unknown else "completed")

    def retrieve(self, **kwargs):
        self.events.append(("retrieve", kwargs))
        if self.readback_error:
            raise self.readback_error
        if self.readback_empty:
            return []
        result = [copy.deepcopy(self.points[str(point_id)]) for point_id in kwargs["ids"] if str(point_id) in self.points]
        if self.readback_foreign and result:
            result[0].payload["company_id"] = "foreign"
        return result

    def delete(self, **kwargs):
        self.events.append(("delete", kwargs))
        for key, point in list(self.points.items()):
            if _matches(point, kwargs["points_selector"]):
                del self.points[key]
        return NS(status="acknowledged" if self.delete_unknown else "completed")

    def count(self, **kwargs):
        return NS(count=sum(_matches(point, kwargs["count_filter"]) for point in self.points.values()))

    def scroll(self, **kwargs):
        self.events.append(("scroll", kwargs))
        return ([point for point in self.points.values() if _matches(point, kwargs["scroll_filter"])], None)

    def search(self, **kwargs):
        self.events.append(("search", kwargs))
        return [NS(payload=point.payload, score=0.98) for point in self.points.values() if _matches(point, kwargs["query_filter"])]


class VectorRuntimeTests(unittest.TestCase):
    def points(self):
        return [{"id": "013fd010-64c4-5864-b1aa-6ea16e75c34e", "text": "传感器产品", "vector": [0.1] * 384, "metadata": {"company_id": "forged"}}]

    def test_local_model_uses_explicit_384_namespace_and_preserves_legacy(self):
        contract = resolve_vector_contract(LOCAL)
        self.assertEqual(contract.dimensions, 384)
        self.assertEqual(contract.collection, "companies_e5_small_v1")
        self.assertNotEqual(contract.collection, "companies")
        with self.assertRaises(ValueError):
            resolve_vector_contract({**LOCAL, "embedding_collection": "companies"})

    def test_new_models_get_distinct_derived_collections(self):
        a = resolve_vector_contract({"embedding_model": "vendor/a", "embedding_dimensions": 384})
        b = resolve_vector_contract({"embedding_model": "vendor/b", "embedding_dimensions": 384})
        self.assertNotEqual(a.collection, b.collection)
        self.assertNotEqual(a.identity, b.identity)

    def test_collection_dimension_mismatch_never_recreates(self):
        client = MemoryQdrant(size=1536)
        store = VectorStore()
        with patch.object(store, "_get_client", return_value=client), self.assertRaises(ValueError):
            store.ensure_collection(LOCAL)
        self.assertNotIn("create", [name for name, _ in client.events])

    def test_409_race_still_checks_dimensions(self):
        for size, expected_success in [(384, True), (1536, False)]:
            client = MemoryQdrant(size=size)
            client.present, client.race = False, True
            store = VectorStore()
            with patch.object(store, "_get_client", return_value=client):
                if expected_success:
                    self.assertEqual(store.ensure_collection(LOCAL).dimensions, 384)
                else:
                    with self.assertRaises(ValueError):
                        store.ensure_collection(LOCAL)
            self.assertIn("get_collection", [name for name, _ in client.events])

    def test_upsert_is_read_back_before_company_scoped_cleanup(self):
        client = MemoryQdrant()
        client.points = {
            "11": NS(id=11, payload={"company_id": "target"}, vector=[0.1] * 384),
            "12": NS(id=12, payload={"company_id": "foreign"}, vector=[0.1] * 384),
        }
        store = VectorStore()
        with patch.object(store, "_get_client", return_value=client):
            receipt = store.upsert_company_vectors("target", self.points(), runtime_config=LOCAL)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["vector_count"], 1)
        self.assertIn("12", client.points)
        self.assertNotIn("11", client.points)
        self.assertEqual(client.points[self.points()[0]["id"]].payload["company_id"], "target")
        events = [name for name, _ in client.events]
        self.assertLess(events.index("retrieve"), events.index("delete"))
        self.assertTrue(next(kwargs for name, kwargs in client.events if name == "upsert")["wait"])

    def test_failed_unknown_empty_or_foreign_readback_never_deletes_previous_points(self):
        failures = ["upsert_error", "operation_unknown", "readback_error", "readback_empty", "readback_foreign"]
        for failure in failures:
            with self.subTest(failure=failure):
                client = MemoryQdrant()
                client.points["old"] = NS(id=10, payload={"company_id": "target"}, vector=[0.1] * 384)
                setattr(client, failure, RuntimeError("fixture failure") if failure.endswith("error") else True)
                store = VectorStore()
                with patch.object(store, "_get_client", return_value=client), self.assertRaises(RuntimeError):
                    store.upsert_company_vectors("target", self.points(), runtime_config=LOCAL)
                self.assertIn("old", client.points)
                self.assertNotIn("delete", [name for name, _ in client.events])

    def test_repeat_identical_upsert_does_not_duplicate_points(self):
        client, store = MemoryQdrant(), VectorStore()
        with patch.object(store, "_get_client", return_value=client):
            first = store.upsert_company_vectors("target", self.points(), runtime_config=LOCAL)
            second = store.upsert_company_vectors("target", self.points(), runtime_config=LOCAL)
        self.assertEqual(first["point_ids"], second["point_ids"])
        self.assertEqual(len(client.points), 1)

    def test_vectors_reject_empty_wrong_zero_nan_inf_bool_or_nonnumeric(self):
        for vector in [[], [1, 2], [0, 0, 0], [1, float("nan"), 2], [1, float("inf"), 2], [1, True, 2], [1, "2", 3]]:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                validate_vector(vector, 3)
        self.assertEqual(validate_vector([0, 0.2, 0.3], 3), [0.0, 0.2, 0.3])

    def test_empty_vector_upsert_fails(self):
        with self.assertRaises(ValueError):
            VectorStore().upsert_company_vectors("target", [], runtime_config=LOCAL)

    def test_search_and_similar_share_write_runtime_model_contract(self):
        client, store = MemoryQdrant(), VectorStore()
        with patch.object(store, "_get_client", return_value=client):
            store.upsert_company_vectors("target", self.points(), runtime_config=LOCAL)
            hits = store.search_companies([0.1] * 384, runtime_config=LOCAL)
            with patch("app.services.runtime_settings.get_ai_runtime_config", AsyncMock(return_value=LOCAL)):
                asyncio.run(store.get_similar_company_ids("target"))
        self.assertEqual(hits[0]["company_id"], "target")
        collections = {kwargs["collection_name"] for name, kwargs in client.events if name in {"search", "scroll", "upsert"}}
        self.assertEqual(collections, {"companies_e5_small_v1"})


if __name__ == "__main__":
    unittest.main()
