"""Bounded company evidence and graph grounding, without real model requests."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.services.ai_client import AIClient
from app.services.company_source import build_company_evidence


class CompanyAISourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_does_not_cut_off_body_after_large_style_or_navigation(self):
        html = '<html><head><style>' + ('padding:5px;' * 2000) + '</style></head><body><nav>新闻 导航 菜单</nav><main><p>上海烨映微电子有限公司专注于热电堆红外传感器和MEMS芯片。</p><p>产品用于测温和工业检测。</p></main></body></html>'
        client = AIClient()
        client.complete = AsyncMock(return_value=json.dumps({"name": "上海烨映微电子有限公司", "category": "传感器"}, ensure_ascii=False))
        result = await client.extract_company_info(html)
        system, user = client.complete.await_args.args
        self.assertEqual(result["category"], "传感器")
        self.assertIn("热电堆红外传感器", user)
        self.assertNotIn("padding:5px", user)
        self.assertNotIn("新闻 导航 菜单", user)
        self.assertIn("普通B2B", system)
        self.assertIn("不要把页面SEO", system)
        self.assertIn("source_evidence", system)

    async def test_source_labels_and_later_about_page_are_preserved(self):
        pages = [
            {"url":"https://example.test/", "role":"homepage", "html":"<main>首页产品摘要" + "示例" * 5000 + "</main>"},
            {"url":"https://example.test/about", "role":"about", "html":"<main><p>示例传感器有限公司是一家专注于红外传感器的企业。</p></main>"},
        ]
        evidence = build_company_evidence(pages)
        client = AIClient()
        client.complete = AsyncMock(return_value='{"name":"示例传感器有限公司"}')
        await client.extract_company_info(evidence)
        user = client.complete.await_args.args[1]
        self.assertIn("https://example.test/about", user)
        self.assertIn("示例传感器有限公司", user)
        self.assertLessEqual(len(user), 18150)

    async def test_invalid_profile_response_does_not_become_empty_success(self):
        for response in ("no JSON", "[]"):
            client = AIClient()
            client.complete = AsyncMock(return_value=response)
            with self.subTest(response=response), self.assertRaises(ValueError):
                await client.extract_company_info("<main>示例传感器公司生产红外传感器。</main>")

    async def test_graph_requires_exact_quotes_and_real_source_urls(self):
        pages = [{"url":"https://example.test/about", "role":"about", "html":"<main><p>示例传感器公司生产热电堆产品，采用MEMS工艺。</p><p>行业新闻提到另一家公司。</p></main>"}]
        evidence = build_company_evidence(pages)
        quote = "示例传感器公司生产热电堆产品，采用MEMS工艺。"
        client = AIClient()
        client.complete = AsyncMock(return_value=json.dumps({
            "nodes":[
                {"name":"示例传感器公司", "type":"Company", "evidence":quote, "source_url":"https://example.test/about"},
                {"name":"热电堆产品", "type":"Product", "evidence":quote, "source_url":"https://example.test/about"},
                {"name":"MEMS工艺", "type":"Technology", "evidence":quote, "source_url":"https://example.test/about"},
                {"name":"虚构客户", "type":"Company", "evidence":"虚构客户购买产品。", "source_url":"https://example.test/about"},
            ],
            "relationships":[
                {"from":"示例传感器公司", "to":"热电堆产品", "type":"HAS_PRODUCT", "evidence":quote, "source_url":"https://example.test/about"},
                {"from":"示例传感器公司", "to":"MEMS工艺", "type":"USES_TECH", "evidence":quote, "source_url":"https://example.test/about"},
                {"from":"示例传感器公司", "to":"热电堆产品", "type":"COMPETES_WITH", "evidence":"示例传感器公司和热电堆产品竞争。", "source_url":"https://example.test/about"},
                {"from":"示例传感器公司", "to":"MEMS工艺", "type":"USES_TECH", "evidence":quote, "source_url":"https://evil.example/"},
            ],
        }, ensure_ascii=False))
        result = await client.extract_entities(evidence)
        self.assertEqual(len(result["nodes"]), 3)
        self.assertEqual([item["type"] for item in result["relationships"]], ["HAS_PRODUCT", "USES_TECH"])
        self.assertTrue(all(item["evidence"] == quote for item in result["relationships"]))
        self.assertIn("新闻提到他家公司", client.complete.await_args.args[0])

    async def test_graph_raw_html_uses_same_clean_evidence_boundary(self):
        client = AIClient()
        client.complete = AsyncMock(return_value='{"nodes":[],"relationships":[]}')
        await client.extract_entities('<html><head><style>random_css</style></head><body><nav>导航垃圾</nav><main>红外传感器产品正文。</main></body></html>')
        sent = client.complete.await_args.args[1]
        self.assertIn("红外传感器产品正文", sent)
        self.assertNotIn("random_css", sent)
        self.assertNotIn("导航垃圾", sent)

    async def test_graph_invalid_json_is_a_visible_failure(self):
        client = AIClient()
        client.complete = AsyncMock(return_value="graph failed")
        with self.assertRaises(ValueError):
            await client.extract_entities("公司介绍")


class SupportedGraphResultTests(unittest.TestCase):
    def test_dynamic_labels_missing_evidence_and_unlisted_endpoints_rejected(self):
        evidence = "示例企业生产示例产品。"
        parsed = {
            "nodes":[
                {"name":"示例企业", "type":"Company", "evidence":evidence},
                {"name":"示例产品", "type":"Product", "evidence":evidence},
                {"name":"示例企业", "type":"Company) DETACH DELETE n //", "evidence":evidence},
                {"name":"未证明", "type":"Company"},
            ],
            "relationships":[
                {"from":"示例企业", "to":"示例产品", "type":"HAS_PRODUCT", "evidence":evidence},
                {"from":"示例企业", "to":"未证明", "type":"HAS_PRODUCT", "evidence":evidence},
                {"from":"示例企业", "to":"示例产品", "type":"HAS_PRODUCT] DELETE n //", "evidence":evidence},
                {"from":"示例企业", "to":"示例产品", "type":"COMPETES_WITH"},
            ],
        }
        result = AIClient._supported_graph_result(parsed, evidence)
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["relationships"][0]["type"], "HAS_PRODUCT")

    def test_cooccurrence_without_a_quote_is_not_relationship_evidence(self):
        parsed = {"nodes":[{"name":"主体公司", "type":"Company", "evidence":"主体公司介绍。"}, {"name":"他家公司", "type":"Company", "evidence":"他家公司新闻。"}], "relationships":[{"from":"主体公司", "to":"他家公司", "type":"COMPETES_WITH"}]}
        result = AIClient._supported_graph_result(parsed, "主体公司介绍。 他家公司新闻。")
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(result["relationships"], [])

    def test_relation_type_needs_an_explicit_predicate_not_merely_two_names(self):
        for quote in ("主体公司和他家公司出现在行业新闻中。", "主体公司与他家公司不构成竞争关系。"):
            parsed = {"nodes":[{"name":"主体公司", "type":"Company", "evidence":quote}, {"name":"他家公司", "type":"Company", "evidence":quote}], "relationships":[{"from":"主体公司", "to":"他家公司", "type":"COMPETES_WITH", "evidence":quote}]}
            result = AIClient._supported_graph_result(parsed, quote)
            self.assertEqual(result["relationships"], [])

    def test_competes_with_only_accepts_explicit_positive_competition_statement(self):
        quote = "主体公司与他家公司在传感器产品领域存在竞争。"
        parsed = {"nodes":[{"name":"主体公司", "type":"Company", "evidence":quote}, {"name":"他家公司", "type":"Company", "evidence":quote}], "relationships":[{"from":"主体公司", "to":"他家公司", "type":"COMPETES_WITH", "evidence":quote}]}
        result = AIClient._supported_graph_result(parsed, quote)
        self.assertEqual(len(result["relationships"]), 1)

    def test_quote_cannot_be_attributed_to_another_sources_url(self):
        quote = "主体公司生产热电堆产品。"
        evidence = "GEORANK_COMPANY_EVIDENCE_V1\n[SOURCE 1] URL=https://example.test/about; role=about\nMAIN_TEXT:\n" + quote + "\n[SOURCE 2] URL=https://example.test/news; role=news\nMAIN_TEXT:\n这是他家公司新闻。"
        parsed = {"nodes":[{"name":"主体公司", "type":"Company", "evidence":quote, "source_url":"https://example.test/news"}], "relationships":[]}
        result = AIClient._supported_graph_result(parsed, evidence)
        self.assertEqual(result["nodes"], [])

    def test_invalid_graph_list_shapes_are_visible_failure(self):
        for parsed in ({"nodes":{}}, {"relationships":"invalid"}):
            with self.subTest(parsed=parsed), self.assertRaises(ValueError):
                AIClient._supported_graph_result(parsed, "正文")


class DeepSeekStructuredCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def client_with_provider(self, *, url="https://api.deepseek.com/v1", model="deepseek-flash", content='{"result":"final-json"}'):
        client = AIClient()
        client._get_runtime_config = AsyncMock(return_value={"llm_providers":[{"id":"fixture", "name":"fixture DeepSeek", "base_url":url, "api_key":"fixture-key", "model":model, "enabled":True}]})
        api = MagicMock()
        api.chat.completions.create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content="DO_NOT_USE_REASONING_AS_FINAL_JSON"), finish_reason="stop")]))
        client._get_provider_client = AsyncMock(return_value=api)
        return client, api

    async def test_exact_official_flash_json_disables_thinking_and_preserves_final_content(self):
        client, api = self.client_with_provider()
        result = await client.complete("return JSON", "fixture evidence", json_mode=True, max_tokens=4096)
        payload = api.chat.completions.create.await_args.kwargs
        self.assertEqual(result, '{"result":"final-json"}')
        self.assertEqual(payload["extra_body"], {"thinking":{"type":"disabled"}})
        self.assertEqual(payload["response_format"], {"type":"json_object"})
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertNotIn("DO_NOT_USE_REASONING", result)

    async def test_only_exact_host_and_supported_version_get_vendor_options(self):
        for url, model in (
            ("https://api.deepseek.com.evil.example/v1", "deepseek-flash"),
            ("https://deepseek.proxy.example/v1", "deepseek-flash"),
            ("https://evil.example/api.deepseek.com", "deepseek-flash"),
            ("https://api.deepseek.com/v1", "deepseek-chat"),
            ("https://api.deepseek.com/v1", "deepseek-reasoner"),
            ("https://api.deepseek.com/v1", "deepseek-flash-custom"),
            ("https://embedding.example/v1", "deepseek-v4-pro"),
        ):
            client, api = self.client_with_provider(url=url, model=model)
            await client.complete("return JSON", "fixture evidence", json_mode=True)
            payload = api.chat.completions.create.await_args.kwargs
            with self.subTest(url=url, model=model):
                self.assertNotIn("extra_body", payload)
                self.assertNotIn("response_format", payload)

    async def test_explicit_v4_model_uses_same_narrow_json_options(self):
        client, api = self.client_with_provider(model="deepseek-v4-pro")
        await client.complete("return JSON", "fixture evidence", json_mode=True, max_tokens=6000)
        payload = api.chat.completions.create.await_args.kwargs
        self.assertEqual(payload["extra_body"], {"thinking":{"type":"disabled"}})
        self.assertEqual(payload["max_tokens"], 6000)

    async def test_general_chat_does_not_change_flash_default_thinking(self):
        client, api = self.client_with_provider()
        await client.complete("chat system", "ordinary question")
        payload = api.chat.completions.create.await_args.kwargs
        self.assertNotIn("extra_body", payload)
        self.assertNotIn("response_format", payload)

    async def test_general_stream_does_not_inject_flash_json_options(self):
        client, api = self.client_with_provider()

        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="fixture token"))])

        api.chat.completions.create = AsyncMock(return_value=stream())
        result = [token async for token in client.stream_complete("chat system", "ordinary question")]
        payload = api.chat.completions.create.await_args.kwargs
        self.assertEqual(result, ["fixture token"])
        self.assertTrue(payload["stream"])
        self.assertNotIn("extra_body", payload)
        self.assertNotIn("response_format", payload)

    async def test_empty_content_does_not_treat_reasoning_as_json_success(self):
        client, _ = self.client_with_provider(content="")
        with self.assertRaisesRegex(ValueError, "空内容"):
            await client.complete("return JSON", "fixture evidence", json_mode=True)

    async def test_raw_provider_override_flattens_only_the_known_deepseek_extension(self):
        client = AIClient()
        response = httpx.Response(200, request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions"), json={"choices":[{"message":{"content":'{"result":"fixture"}', "reasoning_content":"not final"}}]})
        http = MagicMock()
        http.post = AsyncMock(return_value=response)
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=http)
        manager.__aexit__ = AsyncMock(return_value=False)
        with patch("app.services.ai_client.validate_provider_base_url", new=AsyncMock()), patch("app.services.ai_client.build_provider_http_client", return_value=manager):
            result = await client._raw_chat_complete(api_key="fixture-key", base_url="https://api.deepseek.com/v1", model="deepseek-flash", messages=[{"role":"user","content":"return JSON"}], temperature=0.1, max_tokens=4096, json_mode=True)
        payload = http.post.await_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type":"disabled"})
        self.assertEqual(payload["response_format"], {"type":"json_object"})
        self.assertNotIn("extra_body", payload)
        self.assertEqual(result, '{"result":"fixture"}')

    async def test_only_company_structured_tasks_enable_json_with_bounded_budget(self):
        client = AIClient()
        client.complete = AsyncMock(return_value='{"name":"示例公司"}')
        await client.extract_company_info("<main>示例公司生产传感器。</main>")
        self.assertTrue(client.complete.await_args.kwargs["json_mode"])
        client.complete = AsyncMock(return_value='{"selected":[{"url":"https://example.test/","role":"homepage"}]}')
        await client.select_company_pages("https://example.test/", "示例公司", [{"url":"https://example.test/about","title":"关于公司","path":"/about"}])
        self.assertTrue(client.complete.await_args.kwargs["json_mode"])
        self.assertEqual(client.complete.await_args.kwargs["max_tokens"], 1536)
        client.complete = AsyncMock(return_value='{"nodes":[],"relationships":[]}')
        await client.extract_entities("示例公司生产传感器。")
        self.assertTrue(client.complete.await_args.kwargs["json_mode"])
        self.assertEqual(client.complete.await_args.kwargs["max_tokens"], 6000)
