"""Network-free tests for the explicit trusted TEI embedding provider."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.services.ai_client import (
    AIClient, LOCAL_TEI_BASE_URL, LOCAL_TEI_DIMENSIONS,
    LOCAL_TEI_MAX_INPUT_CHARS, LOCAL_TEI_MODEL, validate_local_tei_config,
)


def local_config(**changes):
    return {
        "embedding_provider": "local_tei", "embedding_api_key": "",
        "embedding_base_url": LOCAL_TEI_BASE_URL, "embedding_model": LOCAL_TEI_MODEL,
        "embedding_dimensions": LOCAL_TEI_DIMENSIONS, **changes,
    }


def vector(value=0.5):
    return [value] * LOCAL_TEI_DIMENSIONS


class LocalTEIConfigTests(unittest.TestCase):
    def test_explicit_exact_local_endpoint_accepts_no_key(self):
        self.assertEqual(validate_local_tei_config(local_config()), LOCAL_TEI_BASE_URL)

    def test_local_exception_does_not_accept_arbitrary_private_endpoints(self):
        for url in (
            "http://127.0.0.1:45310/v1", "http://localhost:45310/v1",
            "http://192.168.1.2:45310/v1", "http://host.docker.internal:6333/v1",
            "http://host.docker.internal:45310/", "http://host.docker.internal:45310/v1?x=y",
            "http://host.docker.internal:45310/v1#fragment", "http://u:p@host.docker.internal:45310/v1",
            "http://host.docker.internal.evil.example:45310/v1", "https://host.docker.internal:45310/v1",
            "http://host.docker.internal:45310/v1/../admin", "http://host.docker.internal:wrong/v1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_local_tei_config(local_config(embedding_base_url=url))

    def test_remote_mode_cannot_enable_private_endpoint_exception(self):
        with self.assertRaises(ValueError):
            validate_local_tei_config(local_config(embedding_provider="remote"))

    def test_model_and_real_dimension_must_match_verified_local_service(self):
        for changes in (
            {"embedding_model": "deepseek-chat"}, {"embedding_dimensions": 1536},
            {"embedding_dimensions": True}, {"embedding_dimensions": "384"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_local_tei_config(local_config(**changes))

    def test_vector_cardinality_dimension_finiteness_and_nonzero_required(self):
        cases = ([], [vector(), vector()], [[0.5] * 1536], [[0.0] * 384],
                 [[math.nan] * 384], [[math.inf] * 384], [[True] * 384], [["0.5"] * 384])
        for vectors in cases:
            with self.subTest(case=str(vectors)[:40]), self.assertRaises(ValueError):
                AIClient._validate_embeddings(vectors, 1, 384)

    def test_valid_real_vector_not_padded_or_reduced(self):
        result = AIClient._validate_embeddings([vector()], 1, 384)
        self.assertEqual(len(result[0]), 384)
        self.assertEqual(result, [vector()])


class LocalTEIEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, config=None):
        client = AIClient()
        client._get_runtime_config = AsyncMock(return_value=config or local_config())
        return client

    def fake_http(self, side_effect=None):
        http = MagicMock()

        async def post(url, **kwargs):
            body = side_effect(url, kwargs) if side_effect else [vector() for _ in kwargs["json"]["inputs"]]
            return httpx.Response(200, json=body, request=httpx.Request("POST", url))

        http.post = AsyncMock(side_effect=post)
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=http)
        manager.__aexit__ = AsyncMock(return_value=False)
        return http, manager

    async def test_local_service_never_gets_legacy_remote_key_or_openai_sdk(self):
        client = self.make_client(local_config(embedding_api_key="legacy-remote-fixture-key"))
        http, manager = self.fake_http()
        with patch("app.services.ai_client.httpx.AsyncClient", return_value=manager) as factory, \
             patch("openai.AsyncOpenAI") as openai_client:
            result = await client.embed("上海烨映微电子热电堆传感器")
        openai_client.assert_not_called()
        factory.assert_called_once_with(timeout=60.0, follow_redirects=False, trust_env=False)
        request = http.post.await_args
        self.assertEqual(request.args[0], "http://host.docker.internal:45310/embed")
        self.assertEqual(request.kwargs, {"json": {"inputs": ["passage: 上海烨映微电子热电堆传感器"], "truncate": True}})
        self.assertEqual(len(result), 384)

    async def test_e5_queries_and_documents_have_separate_prefixes(self):
        client = self.make_client()
        http, manager = self.fake_http()
        with patch("app.services.ai_client.httpx.AsyncClient", return_value=manager):
            await client.embed("介绍热电堆")
            await client.embed_query("什么是热电堆")
            await client.embed_query("passage: 查询传感器")
            await client.embed("query: 文档介绍")
        actual = [call.kwargs["json"]["inputs"][0] for call in http.post.await_args_list]
        self.assertEqual(actual, ["passage: 介绍热电堆", "query: 什么是热电堆", "query: 查询传感器", "passage: 文档介绍"])

    async def test_batches_are_bounded_to_sixteen_and_preserve_order(self):
        def make_vectors(url, kwargs):
            return [vector(int(text.split("doc", 1)[1]) + 1) for text in kwargs["json"]["inputs"]]
        client = self.make_client()
        http, manager = self.fake_http(make_vectors)
        with patch("app.services.ai_client.httpx.AsyncClient", return_value=manager):
            result = await client.embed_batch([f"doc{i}" for i in range(35)])
        self.assertEqual([len(call.kwargs["json"]["inputs"]) for call in http.post.await_args_list], [16, 16, 3])
        self.assertEqual([item[0] for item in result], list(range(1, 36)))

    async def test_truncation_uses_real_tei_tokenizer_not_a_guessed_token_count(self):
        client = self.make_client()
        http, manager = self.fake_http()
        with patch("app.services.ai_client.httpx.AsyncClient", return_value=manager):
            await client.embed("热电堆" * 20000)
        payload = http.post.await_args.kwargs["json"]
        self.assertTrue(payload["truncate"])
        self.assertLessEqual(len(payload["inputs"][0]), LOCAL_TEI_MAX_INPUT_CHARS + len("passage: "))

    async def test_empty_batch_does_not_load_config_or_access_network(self):
        client = self.make_client()
        with patch("app.services.ai_client.httpx.AsyncClient") as factory:
            self.assertEqual(await client.embed_batch([]), [])
        factory.assert_not_called()
        client._get_runtime_config.assert_not_awaited()

    async def test_blank_and_non_text_inputs_fail_before_network(self):
        for text in ("", "   ", None, 12, "passage: "):
            client = self.make_client()
            with self.subTest(text=text), self.assertRaises(ValueError), \
                 patch("app.services.ai_client.httpx.AsyncClient") as factory:
                await client.embed(text)
            factory.assert_not_called()

    async def test_remote_provider_still_requires_its_own_key(self):
        client = self.make_client(local_config(embedding_provider="remote", embedding_base_url="https://provider.example/v1"))
        with self.assertRaisesRegex(ValueError, "API Key 未配置"), \
             patch.object(client, "_create_openai_client", new=AsyncMock()) as creator:
            await client.embed("文档")
        creator.assert_not_awaited()

    async def test_remote_key_does_not_bypass_existing_private_provider_policy(self):
        client = self.make_client(local_config(embedding_provider="remote", embedding_api_key="fixture-key"))
        with self.assertRaises(ValueError), patch("openai.AsyncOpenAI") as openai_client:
            await client.embed("文档")
        openai_client.assert_not_called()

    async def test_unknown_provider_is_not_treated_as_local(self):
        client = self.make_client(local_config(embedding_provider="unsafe_local"))
        with self.assertRaises(ValueError), patch("app.services.ai_client.httpx.AsyncClient") as factory:
            await client.embed("文档")
        factory.assert_not_called()

    async def test_remote_response_reorders_by_input_index_and_validates_dimension(self):
        config = local_config(embedding_provider="remote", embedding_api_key="fixture-key", embedding_base_url="https://embedding.example/v1")
        client = self.make_client(config)
        api = MagicMock()
        api.embeddings.create = AsyncMock(return_value=SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=vector(2)), SimpleNamespace(index=0, embedding=vector(1)),
        ]))
        with patch.object(client, "_create_openai_client", new=AsyncMock(return_value=api)):
            result = await client.embed_batch(["one", "two"])
        self.assertEqual([item[0] for item in result], [1.0, 2.0])
        api.embeddings.create.assert_awaited_once_with(model=LOCAL_TEI_MODEL, input=["one", "two"], dimensions=384)

    async def test_wrong_local_vector_dimensions_raise_instead_of_synthetic_padding(self):
        client = self.make_client()
        http, manager = self.fake_http(lambda url, kwargs: [[0.5] * 1536])
        with self.assertRaisesRegex(ValueError, "真实维数"), patch("app.services.ai_client.httpx.AsyncClient", return_value=manager):
            await client.embed("document")

    async def test_redirect_is_not_followed(self):
        client = self.make_client()
        http, manager = self.fake_http()
        http.post = AsyncMock(return_value=httpx.Response(302, headers={"location": "http://127.0.0.1:6333"}, request=httpx.Request("POST", "http://host.docker.internal:45310/embed")))
        with self.assertRaises(httpx.HTTPStatusError), patch("app.services.ai_client.httpx.AsyncClient", return_value=manager):
            await client.embed("document")
        http.post.assert_awaited_once()

    async def test_explicit_frozen_contract_does_not_reload_mutable_runtime(self):
        client = self.make_client(local_config(embedding_provider="remote", embedding_dimensions=1536))
        frozen = local_config(embedding_collection="companies_e5_small_v1")
        http, manager = self.fake_http()
        with patch("app.services.ai_client.httpx.AsyncClient", return_value=manager):
            result = await client.embed_query("查询传感器", runtime_config=frozen)
        client._get_runtime_config.assert_not_awaited()
        self.assertEqual(len(result), 384)
        self.assertEqual(http.post.await_args.kwargs["json"]["inputs"], ["query: 查询传感器"])

    async def _verify_rag_contract_chain(self, *, streaming):
        from app.services.vector_store import VectorStore, resolve_vector_contract

        original = local_config(embedding_collection="companies_e5_small_v1")
        client = self.make_client(original)
        client.complete = AsyncMock(return_value="fixture answer")
        client._resolve_solution_channel = AsyncMock(return_value={})

        async def fake_stream(*args, **kwargs):
            yield "fixture answer"

        client.stream_complete = fake_stream
        http, manager = self.fake_http()
        fake_qdrant = MagicMock()
        fake_qdrant.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="companies_e5_small_v1")])
        fake_qdrant.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=384, distance="Cosine"))))
        fake_qdrant.search.return_value = []
        store = VectorStore()
        store._get_client = MagicMock(return_value=fake_qdrant)
        real_search = store.search_companies
        observed_configs = []

        def observed_search(query_vector, top_k=5, *, runtime_config=None):
            observed_configs.append(runtime_config)
            # Mutating the source cache must not rewrite the request's frozen contract.
            self.assertIsNot(runtime_config, original)
            self.assertEqual(resolve_vector_contract(runtime_config).collection, "companies_e5_small_v1")
            self.assertEqual(len(query_vector), 384)
            return real_search(query_vector, top_k=top_k, runtime_config=runtime_config)

        original_post = http.post.side_effect

        async def mutate_after_embedding(url, **kwargs):
            response = await original_post(url, **kwargs)
            original.update({"embedding_provider":"remote", "embedding_model":"different-model", "embedding_dimensions":1536, "embedding_collection":"companies"})
            return response

        http.post.side_effect = mutate_after_embedding
        templates = {"system_prompt":"fixture system", "streaming_system_prompt":"fixture stream system", "response_instruction":"fixture instruction"}
        with patch("app.services.ai_client.httpx.AsyncClient", return_value=manager), \
             patch("app.services.vector_store.vector_store.search_companies", side_effect=observed_search), \
             patch("app.services.runtime_settings.get_solution_template_config", new=AsyncMock(return_value=templates)):
            if streaming:
                result = [event async for event in client.rag_recommend_stream("热电堆传感器", None, db=None)]
                self.assertEqual(result, [{"type":"text", "content":"fixture answer"}])
            else:
                result = await client.rag_recommend("热电堆传感器", None, db=None)
                self.assertEqual(result, ("fixture answer", []))
        client._get_runtime_config.assert_awaited_once()
        self.assertEqual(http.post.await_args.kwargs["json"]["inputs"], ["query: 热电堆传感器"])
        self.assertEqual(len(observed_configs), 1)
        search_call = fake_qdrant.search.call_args
        self.assertEqual(search_call.kwargs["collection_name"], "companies_e5_small_v1")
        self.assertEqual(len(search_call.kwargs["query_vector"]), 384)
        self.assertEqual(fake_qdrant.get_collection.call_args.kwargs["collection_name"], "companies_e5_small_v1")

    async def test_non_stream_rag_uses_frozen_e5_prefix_and_actual_new_collection(self):
        await self._verify_rag_contract_chain(streaming=False)

    async def test_stream_rag_uses_frozen_e5_prefix_and_actual_new_collection(self):
        await self._verify_rag_contract_chain(streaming=True)
