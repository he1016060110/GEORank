import unittest
from unittest.mock import AsyncMock, patch

from app.services.company_profile import (
    CompanyProfileExtractionError,
    extract_company_profile,
    fallback_company_profile_from_html,
    build_company_profile_values,
    normalize_company_name,
)


class CompanyProfileTests(unittest.TestCase):
    def test_normalize_company_name_removes_marketing_suffixes(self):
        self.assertEqual(
            normalize_company_name("移山科技官网 | GEO优化服务 | AI搜索引擎优化"),
            "移山科技",
        )

    def test_normalize_company_name_preserves_clean_brand_name(self):
        self.assertEqual(
            normalize_company_name("BrightEdge"),
            "BrightEdge",
        )

    def test_normalize_company_name_falls_back_when_primary_empty(self):
        self.assertEqual(
            normalize_company_name("", fallback_name="Narrato"),
            "Narrato",
        )


_ABOUT_HTML = """<html><head><title>Example Sensor Ltd | Official Site</title><style>CSS</style></head>
<body><nav>NAV-ONLY Home Products Contact</nav><main><section class="company-profile">
<p>Example Sensor Ltd designs and develops infrared thermopile detectors, CMOS-MEMS chips and temperature compensation modules for medical instruments and industrial equipment.</p>
<p>Products include digital infrared temperature modules and NDIR gas sensing detector components.</p>
</section></main><footer>FOOTER-ONLY</footer></body></html>"""
_LLM_PROFILE = {
    "name": "Example Sensor Ltd",
    "description": "Example Sensor Ltd designs infrared thermopile detectors and CMOS-MEMS chips, supplying temperature compensation modules for medical instruments and industrial equipment.",
    "short_description": "Infrared thermopile and CMOS-MEMS detector developer.",
    "category": "Sensor components", "tags": ["infrared", "MEMS"],
    "tech_stack": ["CMOS-MEMS"], "team_members": [],
    "headquarters": None, "founded_date": None, "employee_count": None, "funding_stage": None,
}


class CompanyFallbackQualityTests(unittest.TestCase):
    def test_fallback_uses_about_paragraph_not_navigation(self):
        profile = fallback_company_profile_from_html(_ABOUT_HTML, fallback_name="example.test", source_url="https://example.test/about")
        self.assertIn("infrared thermopile", profile["description"])
        self.assertNotIn("NAV-ONLY", profile["description"])
        self.assertNotIn("FOOTER-ONLY", profile["description"])
        self.assertIsNone(profile["headquarters"])
        self.assertIsNone(profile["employee_count"])
        self.assertEqual(profile["extraction_quality"]["status"], "degraded")
        self.assertEqual(profile["extraction_quality"]["source"], "html_fallback")
        self.assertIn("https://example.test/about", profile["extraction_quality"]["source_urls"])

    def test_about_intro_wins_over_unrelated_homepage_slider(self):
        pages = [
            {"url": "https://example.test/", "role": "homepage", "html": "<main><div><p>A generic theme slider displays unrelated headline text about consumer travel deals.</p></div></main>"},
            {"url": "https://example.test/about", "role": "about", "html": _ABOUT_HTML},
        ]
        profile = fallback_company_profile_from_html("", source_pages=pages)
        self.assertTrue(profile["description"].startswith("Example Sensor Ltd designs"))


class CompanyProfileExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_success_uses_clean_source_labelled_evidence(self):
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(return_value=_LLM_PROFILE)) as extract:
            profile = await extract_company_profile(_ABOUT_HTML, source_url="https://example.test/about", strict=True)
        prompt = extract.await_args.args[0]
        self.assertTrue(prompt.startswith("GEORANK_COMPANY_EVIDENCE_V1"))
        self.assertIn("https://example.test/about", prompt)
        self.assertIn("CMOS-MEMS chips", prompt)
        self.assertNotIn("NAV-ONLY", prompt)
        self.assertEqual(profile["extraction_quality"]["status"], "passed")
        self.assertEqual(profile["extraction_quality"]["source"], "llm")
        self.assertEqual(profile["extraction_quality"]["errors"], [])
        self.assertIsNone(profile["headquarters"])

    async def test_provider_error_is_failed_fallback_not_success_or_secret(self):
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(side_effect=RuntimeError("key=fixture-secret never-persist"))):
            profile = await extract_company_profile(_ABOUT_HTML)
        quality = profile["extraction_quality"]
        self.assertEqual(quality["status"], "failed")
        self.assertEqual(quality["source"], "html_fallback")
        self.assertIn("company_extraction_provider_error", quality["errors"])
        self.assertNotIn("fixture-secret", str(quality))
        self.assertIn("infrared thermopile", profile["description"])

    async def test_strict_provider_failure_raises_with_quality(self):
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(side_effect=ValueError("API Base URL 不能为空"))):
            with self.assertRaises(CompanyProfileExtractionError) as context:
                await extract_company_profile(_ABOUT_HTML, strict=True)
        self.assertEqual(context.exception.quality["status"], "failed")
        self.assertIn("company_extraction_api_base_missing", context.exception.quality["errors"])

    async def test_empty_response_and_brief_description_fail_strict(self):
        for response, code in (({}, "company_extraction_empty_response"), ({"name": "Example", "description": "Sensors"}, "company_extraction_description_insufficient")):
            with self.subTest(code=code), patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(return_value=response)):
                with self.assertRaises(CompanyProfileExtractionError) as context:
                    await extract_company_profile(_ABOUT_HTML, strict=True)
                self.assertIn(code, context.exception.quality["errors"])

    async def test_navigation_only_no_model_request(self):
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock()) as extract:
            with self.assertRaises(CompanyProfileExtractionError) as context:
                await extract_company_profile("<nav>Home Product English</nav><footer>Copyright</footer>", strict=True)
        extract.assert_not_awaited()
        self.assertIn("company_source_text_empty", context.exception.quality["errors"])

    async def test_invalid_llm_types_fail_without_typeerror_in_database_builder(self):
        response = dict(_LLM_PROFILE, description={"text": "not a string"}, tags="not a list", employee_count=50)
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(return_value=response)):
            profile = await extract_company_profile(_ABOUT_HTML)
        self.assertIsInstance(profile["description"], str)
        self.assertIsInstance(profile["tags"], list)
        self.assertIsNone(profile["employee_count"])
        self.assertIn("company_extraction_invalid_description", profile["extraction_quality"]["errors"])

    async def test_company_name_conflict_preserved_and_not_auto_corrected(self):
        pages = [
            {"url": "https://example.test/zh/about", "role": "about", "html": "<main><p>示例传感有限公司是一家专注于温度测量模块与传感器研发的公司，为工业设备提供完整检测组件。</p></main>"},
            {"url": "https://example.test/legacy/about", "role": "about", "html": "<main><p>示例芯片有限公司是一家专注于集成电路与红外芯片研发的公司，为医疗仪器提供模块与元器件。</p></main>"},
        ]
        response = dict(_LLM_PROFILE, name="示例芯片有限公司")
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(return_value=response)):
            profile = await extract_company_profile("", fallback_name="Existing Brand", source_pages=pages, strict=True)
        self.assertEqual(profile["name"], "Existing Brand")
        self.assertIn("source_company_name_conflict", profile["extraction_quality"]["warnings"])
        self.assertEqual({item["value"] for item in profile["name_evidence"]}, {"示例传感有限公司", "示例芯片有限公司"})
        self.assertTrue(profile["extraction_quality"]["requires_review"])

    async def test_field_attribution_must_have_actual_quote_at_that_url(self):
        pages = [{"url": "https://example.test/about", "role": "about", "html": _ABOUT_HTML}]
        response = dict(_LLM_PROFILE, field_evidence={
            "name": [{"url": "https://example.test/about", "quote": "Example Sensor Ltd"}],
            "headquarters": [{"url": "https://wrong.test/", "quote": "London"}],
        })
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(return_value=response)):
            profile = await extract_company_profile("", source_pages=pages, strict=True)
        self.assertIn("name", profile["field_evidence"])
        self.assertNotIn("headquarters", profile["field_evidence"])


class CompanyProfilePersistenceTests(unittest.TestCase):
    def test_quality_receipt_preserved_alongside_geo_scores(self):
        from types import SimpleNamespace
        company = SimpleNamespace(name="Existing", geo_details={"meta": 55})
        profile = fallback_company_profile_from_html(_ABOUT_HTML, source_url="https://example.test/about")
        values = build_company_profile_values(company, profile)
        self.assertEqual(values["geo_details"]["meta"], 55)
        self.assertEqual(values["geo_details"]["extraction_quality"]["source"], "html_fallback")
        self.assertIn("description", values["geo_details"]["extraction_quality"]["field_evidence"])


class CompanyProfileIdentityGroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_cannot_invent_missing_legal_name(self):
        response = dict(_LLM_PROFILE, name="Unsupported Official Corp")
        with patch("app.services.company_profile.ai_client.extract_company_info", new=AsyncMock(return_value=response)):
            profile = await extract_company_profile(_ABOUT_HTML, fallback_name="Existing", strict=True)
        self.assertEqual(profile["name"], "Example Sensor Ltd")
        self.assertIn("model_company_name_not_grounded_in_sources", profile["extraction_quality"]["warnings"])
