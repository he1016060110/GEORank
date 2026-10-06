"""Company stage gates run with a fake SQL session and fake artifact services only."""
import copy
from types import SimpleNamespace as NS
import unittest
import uuid
from unittest.mock import AsyncMock, Mock, patch

from app.models.company import PipelineStatus, PublishStatus
from app.tasks import process

CID = "11111111-1111-4111-8111-111111111111"
LOCAL = {"embedding_provider": "local_tei", "embedding_model": "intfloat/multilingual-e5-small",
         "embedding_dimensions": 384, "embedding_collection": "companies_e5_small_v1"}
TEXT = "烨映提供SSG11DF42热电堆传感器用于非接触测温。该传感器适用于工业测温和人体温度检测。企业产品包括红外传感器和MEMS元件。"
DOCUMENTS = [{"url": "https://example.com/about", "role": "about", "text": TEXT,
              "html_sha256": "a" * 64, "meta": {}, "structured_data": [], "name_candidates": [],
              "warnings": [], "text_chars": len(TEXT), "title": "企业介绍"}]
PAGES = [{"key": "source.html", "html": "<main>" + TEXT + "</main>", "url": "https://example.com/about", "role": "about"}]


class FakeSession:
    def __init__(self, company):
        self.company = company
        self.writes = []
        self.commits = 0
        self.rollbacks = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def execute(self, statement):
        if getattr(statement, "is_select", False):
            return NS(scalar_one_or_none=lambda: self.company)
        params = statement.compile().params
        self.writes.append(copy.deepcopy(params))
        for key, value in params.items():
            if hasattr(self.company, key):
                setattr(self.company, key, copy.deepcopy(value))
        return NS()

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1

    async def refresh(self, company):
        return None


class PipelineGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.company = NS(
            id=uuid.UUID(CID), name="烨映", url="https://example.com", category="传感器",
            description=TEXT, short_description="红外传感器制造商", geo_details={},
            crawl_pages=[{"key": "source.html", "url": "https://example.com/about", "role": "about"}],
            raw_html_key="source.html", about_html_key=None, submitted_by=None,
            pipeline_status=PipelineStatus.CLEANING, pipeline_error=None,
            publish_status=PublishStatus.DRAFT, tags=[], tech_stack=[], team_members=[],
        )
        self.db = FakeSession(self.company)
        self.db_patch = patch("app.core.database.async_session", return_value=self.db)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.dispatch_patch = patch("app.core.celery_app.celery_app.send_task")
        self.dispatch = self.dispatch_patch.start()
        self.addCleanup(self.dispatch_patch.stop)
        self.sources_patch = patch.object(process, "_read_sources", return_value=(PAGES, DOCUMENTS))
        self.sources_patch.start()
        self.addCleanup(self.sources_patch.stop)

    def stage_quality(self, stages):
        self.company.geo_details = {"pipeline_quality": {
            "run_id": "fixture-run", "status": "running", "entity_count": 2,
            "sources": [{"url": doc["url"], "html_sha256": doc["html_sha256"]} for doc in DOCUMENTS],
            "stages": {stage: {"status": "complete", "graph_version": "fixture-graph", "entity_count": 2, "relationship_count": 0} for stage in stages},
        }}

    def assert_failed(self, stage):
        self.assertEqual(self.company.pipeline_status, "failed")
        quality = self.company.geo_details["pipeline_quality"]
        self.assertEqual(quality["status"], "failed")
        self.assertEqual(quality["stages"][stage]["status"], "failed")
        self.assertNotEqual(self.company.publish_status, PublishStatus.PENDING_REVIEW)
        self.assertFalse(any(item.get("pipeline_status") == "completed" for item in self.db.writes))
        self.dispatch.assert_not_called()

    async def test_clean_missing_saved_source_fails_without_http_or_ai_fallback(self):
        with patch.object(process, "_read_sources", side_effect=process.PipelineArtifactError("source_saved_html_missing")), patch("app.services.company_profile.extract_company_profile", AsyncMock()) as extract:
            with self.assertRaises(process.PipelineArtifactError):
                await process._run_clean(CID)
        self.assert_failed("clean")
        extract.assert_not_awaited()
        self.assertEqual(self.company.description, TEXT)

    async def test_clean_fallback_or_insufficient_profile_cannot_advance(self):
        profiles = [
            {"description": TEXT, "extraction_quality": {"source": "html_fallback", "status": "degraded"}},
            {"description": "too short", "extraction_quality": {"source": "llm", "status": "passed"}},
        ]
        for profile in profiles:
            with self.subTest(profile=profile), patch("app.services.company_profile.extract_company_profile", AsyncMock(return_value=profile)):
                with self.assertRaises(process.PipelineArtifactError):
                    await process._run_clean(CID)
            self.assert_failed("clean")

    async def test_clean_verified_llm_profile_records_evidence_and_advances_only_graph(self):
        profile = {"description": TEXT, "name": "烨映", "extraction_quality": {"source": "llm", "status": "passed"},
                   "name_evidence": [{"value": "烨映", "url": "https://example.com/about"}]}
        with patch("app.services.company_profile.extract_company_profile", AsyncMock(return_value=profile)) as extract:
            quality = await process._run_clean(CID)
        self.assertEqual(self.company.pipeline_status, "graph_building")
        self.assertEqual(quality["stages"]["clean"]["status"], "complete")
        self.assertIn("https://example.com/about", quality["source_urls"])
        self.assertTrue(quality["stages"]["crawl"]["verified"])
        self.assertTrue(quality["stages"]["clean"]["verified"])
        self.assertEqual(quality["document_count"], quality["stages"]["crawl"]["document_count"])
        self.dispatch.assert_called_once_with("app.tasks.process.build_knowledge_graph", args=[CID])
        self.assertTrue(extract.call_args.kwargs["strict"])

    async def test_graph_missing_clean_receipt_never_calls_llm_or_enqueues_vector(self):
        with patch("app.services.ai_client.ai_client.extract_entities", AsyncMock()) as extract:
            with self.assertRaises(process.PipelineArtifactError):
                await process._run_graph(CID)
        self.assert_failed("graph")
        extract.assert_not_awaited()

    async def test_graph_empty_or_failed_provider_cannot_be_completed(self):
        self.stage_quality(["clean"])
        for response in [{"nodes": []}, {"nodes": [{"name": "invented", "type": "Product"}]}]:
            with self.subTest(response=response), patch("app.services.ai_client.ai_client.extract_entities", AsyncMock(return_value=response)):
                with self.assertRaises(process.PipelineArtifactError):
                    await process._run_graph(CID)
            self.assert_failed("graph")
            self.stage_quality(["clean"])
        with patch("app.services.ai_client.ai_client.extract_entities", AsyncMock(side_effect=RuntimeError("secret=should-not-persist"))):
            with self.assertRaises(RuntimeError):
                await process._run_graph(CID)
        self.assert_failed("graph")
        self.assertNotIn("secret", self.company.pipeline_error)

    async def test_graph_readback_failure_never_advances(self):
        self.stage_quality(["clean"])
        data = {"nodes": [{"name": "SSG11DF42", "type": "Product"}], "relationships": []}
        with patch("app.services.ai_client.ai_client.extract_entities", AsyncMock(return_value=data)), patch.object(process, "_persist_graph", AsyncMock(side_effect=process.PipelineArtifactError("graph_readback_entity_count_mismatch"))):
            with self.assertRaises(process.PipelineArtifactError):
                await process._run_graph(CID)
        self.assert_failed("graph")

    async def test_graph_verified_artifacts_only_advance_vector(self):
        self.stage_quality(["clean"])
        data = {"nodes": [{"name": "SSG11DF42", "type": "Product"}], "relationships": []}
        receipt = {"verified": True, "entity_count": 1, "relationship_count": 0, "run_id": "fixture-run"}
        with patch("app.services.ai_client.ai_client.extract_entities", AsyncMock(return_value=data)), patch.object(process, "_persist_graph", AsyncMock(return_value=receipt)):
            quality = await process._run_graph(CID)
        self.assertEqual(self.company.pipeline_status, "vectorizing")
        self.assertEqual(quality["entity_count"], 1)
        self.dispatch.assert_called_once_with("app.tasks.process.vectorize_knowledge_base", args=[CID])

    def vector_patches(self, vector_results):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("app.services.runtime_settings.get_ai_runtime_config", AsyncMock(return_value=LOCAL)))
        stack.enter_context(patch("app.services.graph_store.get_company_graph", AsyncMock(return_value={"graph_version": "fixture-graph", "run_id": "fixture-run", "node_count": 2, "relation_count": 0})))
        stack.enter_context(patch("app.services.vector_store.vector_store.ensure_collection"))
        stack.enter_context(patch("app.services.ai_client.ai_client.embed_batch", vector_results))
        return stack

    async def test_vector_empty_partial_wrong_dims_nan_or_zero_fails(self):
        fixtures = [[], [[0.1] * 384], [[0.1] * 1536] * 2, [[float("nan")] * 384] * 2, [[0] * 384] * 2]
        for result in fixtures:
            self.stage_quality(["clean", "graph"])
            with self.subTest(size=len(result)), self.vector_patches(AsyncMock(return_value=result)), patch("app.services.vector_store.vector_store.upsert_company_vectors") as upsert:
                with self.assertRaises(ValueError):
                    await process._run_vectorize(CID)
            self.assert_failed("vector")
            upsert.assert_not_called()

    async def test_vector_upsert_or_readback_failure_cannot_complete(self):
        for failure in [RuntimeError("upsert fixture"), {"verified": False, "vector_count": 2}, {"verified": True, "vector_count": 0}]:
            self.stage_quality(["clean", "graph"])
            with self.subTest(failure=failure), self.vector_patches(AsyncMock(return_value=[[0.1] * 384] * 2)), patch("app.services.vector_store.vector_store.upsert_company_vectors", **({"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure})):
                with self.assertRaises((RuntimeError, ValueError)):
                    await process._run_vectorize(CID)
            self.assert_failed("vector")

    async def test_vector_verified_readback_only_completes_and_remains_unpublished(self):
        self.stage_quality(["clean", "graph"])
        receipt = {"verified": True, "vector_count": 2, "collection": LOCAL["embedding_collection"], "dimensions": 384, "model": LOCAL["embedding_model"]}
        with self.vector_patches(AsyncMock(return_value=[[0.1] * 384] * 2)), patch("app.services.vector_store.vector_store.upsert_company_vectors", return_value=receipt) as upsert, patch("app.services.ai_client.ai_client.embed_batch", AsyncMock(return_value=[[0.1] * 384] * 2)) as embed:
            quality = await process._run_vectorize(CID)
        self.assertEqual(embed.call_args.kwargs["runtime_config"], LOCAL)
        self.assertEqual(self.company.pipeline_status, "completed")
        self.assertEqual(self.company.publish_status, "pending_review")
        self.assertEqual(quality["vector_count"], 2)
        self.assertEqual(quality["stages"]["vector"]["collection"], "companies_e5_small_v1")
        self.assertEqual(upsert.call_args.kwargs["runtime_config"], LOCAL)
        self.dispatch.assert_not_called()

    async def test_embedding_settings_edit_during_run_prevents_mixed_upsert(self):
        self.stage_quality(["clean", "graph"])
        changed = {**LOCAL, "embedding_collection": "companies_another_e5_namespace"}
        with self.vector_patches(AsyncMock(return_value=[[0.1] * 384] * 2)), patch("app.services.runtime_settings.get_ai_runtime_config", AsyncMock(side_effect=[LOCAL, changed])), patch("app.services.vector_store.vector_store.upsert_company_vectors") as upsert, patch("app.services.ai_client.ai_client.embed_batch", AsyncMock(return_value=[[0.1] * 384] * 2)) as embed:
            with self.assertRaises(process.PipelineArtifactError):
                await process._run_vectorize(CID)
        self.assert_failed("vector")
        upsert.assert_not_called()
        self.assertEqual(embed.call_args.kwargs["runtime_config"], LOCAL)

    async def test_graph_adapter_uses_managed_snapshot_writer_and_reads_committed_result(self):
        entity = {"name": "SSG11DF42", "type": "Product", "source_quote": "SSG11DF42"}
        receipt = {"verified": True, "persisted": True, "graph_version": "fixture-graph", "node_count": 1, "relation_count": 0}
        observed = {"graph_version": "fixture-graph", "run_id": "fixture-run", "node_count": 1, "relation_count": 0}
        with patch("app.services.graph_store.upsert_company_entities", AsyncMock(return_value=receipt)) as writer, patch("app.services.graph_store.get_company_graph", AsyncMock(return_value=observed)):
            verified = await process._persist_graph(self.company, [entity], [], "fixture-run", documents=DOCUMENTS)
        self.assertTrue(verified["verified"])
        self.assertEqual(writer.call_args.kwargs["source_urls"], ["https://example.com/about"])
        self.assertEqual(writer.call_args.kwargs["source_hashes"][0]["sha256"], "a" * 64)
        with patch("app.services.graph_store.upsert_company_entities", AsyncMock(return_value=receipt)), patch("app.services.graph_store.get_company_graph", AsyncMock(return_value={**observed, "run_id": "old-run"})):
            with self.assertRaises(process.PipelineArtifactError):
                await process._persist_graph(self.company, [entity], [], "fixture-run", documents=DOCUMENTS)

    async def test_saved_sources_changed_after_clean_blocks_graph(self):
        self.stage_quality(["clean"])
        self.company.geo_details["pipeline_quality"]["sources"][0]["html_sha256"] = "b" * 64
        with patch("app.services.ai_client.ai_client.extract_entities", AsyncMock()) as extract:
            with self.assertRaises(process.PipelineArtifactError):
                await process._run_graph(CID)
        self.assert_failed("graph")
        extract.assert_not_awaited()

    async def test_current_graph_disappeared_or_changed_blocks_vector_without_embedding(self):
        self.stage_quality(["clean", "graph"])
        with self.vector_patches(AsyncMock()) as patches, patch("app.services.graph_store.get_company_graph", AsyncMock(return_value={"graph_version": "old", "run_id": "old", "node_count": 0})), patch("app.services.ai_client.ai_client.embed_batch", AsyncMock()) as embed:
            with self.assertRaises(process.PipelineArtifactError):
                await process._run_vectorize(CID)
        self.assert_failed("vector")
        embed.assert_not_awaited()

    def test_quote_from_two_different_pages_is_never_one_supported_relationship(self):
        docs = [{"text": "烨映提供"}, {"text": "SSG11DF42热电堆传感器用于测温。"}]
        source = "\n".join(doc["text"] for doc in docs)
        data = {"nodes": [{"name": "SSG11DF42", "type": "Product"}], "relationships": [
            {"from": "烨映", "to": "SSG11DF42", "type": "HAS_PRODUCT", "evidence": "烨映提供SSG11DF42热电堆传感器用于测温。"}
        ]}
        _, relations, _ = process._prepare_graph_data(data, source, "烨映", documents=docs)
        self.assertEqual(relations, [])

    async def test_full_fixture_chain_produces_review_gate_compatible_verified_quality(self):
        from app.api.routes.admin import _company_publish_quality_issues
        profile = {"description": TEXT, "name": "烨映", "extraction_quality": {"source": "llm", "status": "passed"}}
        data = {"nodes": [{"name": "SSG11DF42", "type": "Product"}], "relationships": []}
        graph_receipt = {"verified": True, "entity_count": 1, "relationship_count": 0, "graph_version": "fixture-graph"}
        vector_receipt = {"verified": True, "vector_count": 2, "collection": LOCAL["embedding_collection"], "dimensions": 384, "model": LOCAL["embedding_model"]}
        with patch("app.services.company_profile.extract_company_profile", AsyncMock(return_value=profile)):
            await process._run_clean(CID)
        run_id = self.company.geo_details["pipeline_quality"]["run_id"]
        with patch("app.services.ai_client.ai_client.extract_entities", AsyncMock(return_value=data)), patch.object(process, "_persist_graph", AsyncMock(return_value={**graph_receipt, "run_id": run_id})):
            await process._run_graph(CID)
        with self.vector_patches(AsyncMock(return_value=[[0.1] * 384] * 2)), patch("app.services.vector_store.vector_store.upsert_company_vectors", return_value=vector_receipt), patch("app.services.graph_store.get_company_graph", AsyncMock(return_value={"graph_version": "fixture-graph", "run_id": run_id, "node_count": 1, "relation_count": 0})):
            await process._run_vectorize(CID)
        self.assertEqual(_company_publish_quality_issues(self.company), [])
        self.assertEqual(self.company.publish_status, "pending_review")
        self.assertEqual(self.dispatch.call_count, 2)

    def test_cjk_chunks_are_bounded_and_ids_are_stable(self):
        text = "传感器" * 1000
        chunks = process._chunk_text(text)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))
        first = process._vector_chunks(self.company, DOCUMENTS, "identity")
        second = process._vector_chunks(self.company, DOCUMENTS, "identity")
        self.assertEqual([point["id"] for point in first], [point["id"] for point in second])
        self.assertTrue(all(uuid.UUID(point["id"]).version == 5 for point in first))

    def test_graph_filters_unsupported_edges_and_ambiguous_entity_names(self):
        data = {"nodes": [{"name": "SSG11DF42", "type": "Product"}, {"name": "SSG11DF42", "type": "Person"},
                          {"name": "热电堆", "type": "Technology"}, {"name": "Invented", "type": "Product"}],
                "relationships": [{"from": "烨映", "to": "热电堆", "type": "USES_TECH", "evidence": "烨映使用热电堆"}]}
        entities, relations, review = process._prepare_graph_data(data, TEXT, "烨映")
        self.assertEqual([entity["name"] for entity in entities], ["热电堆"])
        self.assertEqual(relations, [])
        self.assertEqual(review["dropped_relationships"], 1)

    def test_graph_accepts_only_current_source_quote_with_both_endpoints(self):
        data = {"nodes": [{"name": "SSG11DF42", "type": "Product"}],
                "relationships": [{"from": "烨映", "to": "SSG11DF42", "type": "HAS_PRODUCT", "evidence": TEXT.split("。")[0] + "。"}]}
        entities, relations, _ = process._prepare_graph_data(data, TEXT, "烨映")
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(relations), 1)
        self.assertTrue(relations[0]["from"]["root"])


if __name__ == "__main__":
    unittest.main()
