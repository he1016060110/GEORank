"""Company navigation fixtures: no network, database, queue or paid AI calls."""
import asyncio
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.tasks import crawl


BASE = "https://fixture.example"
PROFILE = {"url": f"{BASE}/company-profile/", "title": "公司简介"}
HOME = {"url": BASE, "title": "Fixture", "role": "homepage", "reason": "Fixture homepage"}


def product_links(count=20, base=BASE):
    return [{"url": f"{base}/product-{i}", "title": f"Product {i}"} for i in range(count)]


def ai_product(url, role="product"):
    return {"url": url, "title": "AI title is not source evidence", "role": role, "reason": "Fixture AI selection"}


class AboutPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def plan(self, links, selected, base=BASE):
        selector = AsyncMock(return_value=selected)
        with patch("app.services.ai_client.ai_client.select_company_pages", selector):
            result = await crawl._plan_company_pages(base, "Fixture", links)
        selector.assert_awaited_once()
        candidates = selector.await_args.args[2]
        self.assertLessEqual(len(candidates), 12)
        self.assertLessEqual(len(result[1]), 3)
        return result, selector

    async def test_profile_after_twenty_products_survives_ai_candidate_budget(self):
        products = product_links()
        (candidates, pages), _ = await self.plan(
            products + [PROFILE], [ai_product(p["url"]) for p in products[:3]],
        )
        self.assertEqual(candidates[0]["url"], f"{BASE}/company-profile")
        self.assertEqual([page["url"] for page in pages], [BASE, f"{BASE}/company-profile", products[0]["url"]])
        self.assertEqual(pages[1]["role"], "about")
        self.assertIn("真实链接", pages[1]["reason"])
        self.assertNotIn("AI选择", pages[1]["reason"])
        self.assertEqual(pages[2]["title"], "Product 0")

    async def test_chinese_company_intro_title_on_real_arbitrary_path(self):
        actual = {"url": f"{BASE}/gsxx", "title": "公司介绍"}
        products = product_links()
        (_, pages), _ = await self.plan(products + [actual], [ai_product(products[0]["url"])])
        self.assertEqual(pages[1]["url"], actual["url"])
        self.assertEqual(pages[1]["role"], "about")

    async def test_all_supported_chinese_identity_titles(self):
        for title in ("公司简介", "公司介绍", "公司概况", "关于我们", "企业介绍"):
            with self.subTest(title=title):
                (_, pages), _ = await self.plan(
                    product_links() + [{"url": f"{BASE}/identity", "title": title}], [],
                )
                self.assertEqual(pages[1]["url"], f"{BASE}/identity")
                self.assertEqual(pages[1]["role"], "about")

    async def test_company_in_hostname_and_product_overview_are_not_identity_evidence(self):
        base = "https://company-products.example"
        links = [{"url": f"{base}/overview", "title": "Product Overview"}]
        (_, pages), _ = await self.plan(links, [ai_product(links[0]["url"], role="about")], base)
        self.assertEqual([page["url"] for page in pages], [base, links[0]["url"]])
        self.assertNotIn("about", [page["role"] for page in pages])
        self.assertEqual(crawl._about_link_priority(links[0]), 0)

    async def test_cross_domain_deep_assets_do_not_displace_real_profile(self):
        invalid = [
            {"url": "https://other.example/company-profile", "title": "公司简介"},
            {"url": f"{BASE}/product/company-profile", "title": "关于我们"},
            {"url": f"{BASE}/company-profile.pdf", "title": "公司简介"},
            {"url": f"{BASE}/company-profile.mp4", "title": "公司简介"},
            {"url": "mailto:about@fixture.example", "title": "About Us"},
        ]
        products = product_links()
        (candidates, pages), _ = await self.plan(invalid + products + [PROFILE], [ai_product(products[0]["url"])])
        urls = {item["url"] for item in candidates}
        for item in invalid:
            self.assertNotIn(item["url"], urls)
        self.assertEqual(pages[1]["url"], f"{BASE}/company-profile")

    async def test_actual_profile_beats_generic_about_and_is_not_duplicated(self):
        links = [{"url": f"{BASE}/about", "title": "About Us"}, PROFILE,
                 {"url": f"{BASE}/company-profile", "title": "Company Profile"}] + product_links()
        selected = [ai_product(f"{BASE}/company-profile/", "about"), ai_product(f"{BASE}/product-0")]
        (candidates, pages), _ = await self.plan(links, selected)
        self.assertEqual(candidates[0]["url"], f"{BASE}/company-profile")
        self.assertEqual(sum(p["url"] == f"{BASE}/company-profile" for p in pages), 1)
        self.assertEqual([p["url"] for p in pages], [BASE, f"{BASE}/company-profile", f"{BASE}/product-0"])

    async def test_duplicate_icon_anchor_does_not_hide_later_chinese_identity_title(self):
        links = [{"url": f"{BASE}/identity", "title": "Menu icon"}] + product_links(100)
        links += [{"url": f"{BASE}/identity/", "title": "公司简介"}]
        (candidates, pages), _ = await self.plan(links, [ai_product(f"{BASE}/product-0")])
        self.assertEqual(candidates[0]["url"], f"{BASE}/identity")
        self.assertEqual(candidates[0]["title"], "公司简介")
        self.assertEqual(pages[1]["url"], f"{BASE}/identity")

    async def test_supported_real_identity_paths_do_not_need_english_anchor_titles(self):
        for path in ("company-profile", "company-introduction", "about", "about-us", "who-we-are"):
            with self.subTest(path=path):
                actual = {"url": f"{BASE}/{path}", "title": "Learn more"}
                (_, pages), _ = await self.plan(product_links(20) + [actual], [])
                self.assertEqual(pages[1]["url"], actual["url"])
                self.assertEqual(pages[1]["role"], "about")

    async def test_ai_invented_cross_domain_or_deep_url_is_not_crawled(self):
        links = product_links() + [PROFILE]
        selected = [ai_product("https://unknown.example/about"), ai_product(f"{BASE}/not-discovered"),
                    ai_product(f"{BASE}/products/deep"), ai_product(f"{BASE}/product-0")]
        (_, pages), _ = await self.plan(links, selected)
        self.assertEqual([p["url"] for p in pages], [BASE, f"{BASE}/company-profile", f"{BASE}/product-0"])

    async def test_no_identity_url_is_guessed_when_no_real_link_exists(self):
        links = product_links(3)
        (_, pages), _ = await self.plan(links, [ai_product(p["url"]) for p in links])
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0]["role"], "homepage")
        self.assertNotIn("about", [p["role"] for p in pages])
        self.assertNotIn(f"{BASE}/company-profile", [p["url"] for p in pages])

    async def test_ai_failure_uses_one_attempt_and_keeps_verified_identity(self):
        selector = AsyncMock(side_effect=RuntimeError("Fixture AI unavailable"))
        with patch("app.services.ai_client.ai_client.select_company_pages", selector):
            candidates, pages = await crawl._plan_company_pages(BASE, "Fixture", product_links() + [PROFILE])
        selector.assert_awaited_once()
        self.assertEqual(pages[1]["url"], f"{BASE}/company-profile")
        self.assertEqual(pages[1]["role"], "about")
        self.assertLessEqual(len(candidates), 12)
        self.assertLessEqual(len(pages), 3)


class CrawlHTTPTests(unittest.TestCase):
    def browser_fixture(self, links, status=200):
        page = MagicMock()
        page.goto.return_value = None if status is None else MagicMock(status=status)
        page.content.return_value = "<main>Actual fixture HTML</main>"
        page.title.return_value = "Fixture"
        page.evaluate.side_effect = ["Actual fixture body text", links]
        browser = MagicMock()
        browser.new_page.return_value = page
        playwright = MagicMock()
        playwright.chromium.launch.return_value = browser
        context = MagicMock()
        context.__enter__.return_value = playwright
        return context, browser, page

    def test_identity_after_over_eighty_dom_anchors_is_preserved(self):
        context, browser, page = self.browser_fixture(product_links(120) + [PROFILE])
        with patch("playwright.sync_api.sync_playwright", return_value=context):
            result = crawl._crawl_page(BASE)
        self.assertEqual(len(result["links"]), 80)
        self.assertEqual(result["links"][0]["url"], f"{BASE}/company-profile")
        self.assertNotIn(".slice(0, 80)", page.evaluate.call_args_list[-1].args[0])
        browser.close.assert_called_once()

    def test_http_error_fails_before_capturing_challenge_or_uploading(self):
        for status in (403, 404, 429, 500):
            with self.subTest(status=status):
                context, browser, page = self.browser_fixture([], status=status)
                with patch("playwright.sync_api.sync_playwright", return_value=context):
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                        crawl._crawl_page(BASE)
                page.content.assert_not_called()
                page.wait_for_timeout.assert_not_called()
                browser.close.assert_called_once()

    def test_success_and_no_response_keep_existing_capture_contract(self):
        for status in (200, None):
            with self.subTest(status=status):
                context, browser, _ = self.browser_fixture([PROFILE], status=status)
                with patch("playwright.sync_api.sync_playwright", return_value=context):
                    result = crawl._crawl_page(BASE)
                self.assertEqual(set(result), {"html", "text", "title", "links"})
                self.assertIn("Actual fixture HTML", result["html"])
                browser.close.assert_called_once()


class CrawlPersistenceTests(unittest.TestCase):
    def run_fixture(self, pages, storage_result=True, send_error=None, extra_crawl_error=None,
                    guard_allow=True, receipt_side_effect=None, admission_error=None):
        updates = AsyncMock()
        page_result = {"html": "<main>Fixture source</main>", "title": "Fixture", "text": "Fixture", "links": []}
        task = crawl.crawl_company_website
        store = MagicMock()
        if isinstance(storage_result, list):
            store.put.side_effect = storage_result
        else:
            store.put.return_value = storage_result
        send = MagicMock(side_effect=send_error)
        retry = MagicMock(side_effect=RuntimeError("Fixture retry"))
        log = MagicMock()
        admission = AsyncMock(return_value=guard_allow, side_effect=admission_error)
        receipt = AsyncMock(return_value=True, side_effect=receipt_side_effect)
        planner = AsyncMock(return_value=([], pages))
        capture = MagicMock(return_value=page_result)
        if extra_crawl_error is not None:
            capture.side_effect = [page_result, extra_crawl_error]
        patches = (
            patch.object(crawl, "_admit_crawl_callback", admission),
            patch.object(crawl, "_record_clean_dispatch", receipt),
            patch.object(crawl, "_update_company", updates),
            patch.object(crawl, "_run", side_effect=asyncio.run),
            patch.object(crawl, "_crawl_page", capture),
            patch.object(crawl, "_plan_company_pages", planner),
            patch("app.services.storage.storage", store),
            patch("app.core.celery_app.celery_app.send_task", send),
            patch.object(task, "retry", retry),
            patch.object(crawl, "log_event", log),
        )
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            task.push_request(id="fixture-crawl-task", retries=0)
            try:
                result = task.run("fixture-company-id", BASE)
            finally:
                task.pop_request()
        self.last_fixture = {
            "admission": admission, "receipt": receipt, "planner": planner,
            "capture": capture, "result": result,
        }
        return updates, store, send, retry, log

    def assert_failed_without_completion(self, updates, send, retry, log):
        from app.models.company import PipelineStatus
        self.assertEqual(updates.await_args_list[-1].kwargs["pipeline_status"], PipelineStatus.FAILED)
        retry.assert_not_called()
        self.assertNotIn("task.crawl_company.completed", [call.args[2] for call in log.call_args_list])

    def test_homepage_storage_fallback_blocks_followup_task(self):
        updates, store, send, retry, log = self.run_fixture([HOME], storage_result=False)
        self.assert_failed_without_completion(updates, send, retry, log)
        send.assert_not_called()
        self.assertIn("未成功持久化", updates.await_args_list[-1].kwargs["pipeline_error"])
        self.assertEqual(store.put.call_count, 1)

    def test_about_page_storage_fallback_also_blocks_followup_task(self):
        about = dict(PROFILE, url=f"{BASE}/company-profile", role="about", reason="Fixture identity")
        updates, store, send, retry, log = self.run_fixture([HOME, about], storage_result=[True, False])
        self.assert_failed_without_completion(updates, send, retry, log)
        send.assert_not_called()
        self.assertEqual(store.put.call_count, 2)

    def test_only_about_not_team_populates_about_html_key(self):
        from app.models.company import PipelineStatus
        team = {"url": f"{BASE}/team", "title": "Team", "role": "team", "reason": "Fixture team"}
        updates, store, send, retry, _ = self.run_fixture([HOME, team])
        cleaning = [call.kwargs for call in updates.await_args_list
                    if call.kwargs.get("pipeline_status") == PipelineStatus.CLEANING][0]
        self.assertIsNone(cleaning["about_html_key"])
        self.assertEqual(store.put.call_count, 2)
        send.assert_called_once_with(
            "app.tasks.process.clean_company_data", args=["fixture-company-id"],
            task_id=self.last_fixture["result"]["clean_task_id"],
        )
        retry.assert_not_called()

    def test_verified_about_is_persisted_once_and_populates_about_key(self):
        from app.models.company import PipelineStatus
        about = dict(PROFILE, url=f"{BASE}/company-profile", role="about", reason="Fixture identity")
        updates, store, send, retry, _ = self.run_fixture([HOME, about])
        cleaning = [call.kwargs for call in updates.await_args_list
                    if call.kwargs.get("pipeline_status") == PipelineStatus.CLEANING][0]
        self.assertEqual(cleaning["about_html_key"], "companies/fixture-company-id/about-2.html")
        self.assertEqual(store.put.call_count, 2)
        send.assert_called_once()
        retry.assert_not_called()

    def test_followup_dispatch_exception_is_unknown_without_paid_crawl_retry(self):
        from app.models.company import PipelineStatus
        updates, _, send, retry, log = self.run_fixture([HOME], send_error=RuntimeError("Fixture broker unavailable"))
        self.assertEqual(updates.await_args_list[-1].kwargs["pipeline_status"], PipelineStatus.CLEANING)
        self.assertNotIn(PipelineStatus.FAILED, [call.kwargs.get("pipeline_status") for call in updates.await_args_list])
        retry.assert_not_called()
        send.assert_called_once()
        self.assertEqual(self.last_fixture["result"]["state"], "dispatch_unknown")
        self.assertEqual(self.last_fixture["receipt"].await_args.args[-1], "unknown")
        self.assertNotIn("task.crawl_company.completed", [call.args[2] for call in log.call_args_list])
        self.last_fixture["planner"].assert_awaited_once()

    def test_old_callback_is_skipped_before_any_mutation_crawl_or_model(self):
        updates, store, send, retry, _ = self.run_fixture([HOME], guard_allow=False)
        updates.assert_not_awaited()
        store.put.assert_not_called()
        send.assert_not_called()
        retry.assert_not_called()
        self.last_fixture["capture"].assert_not_called()
        self.last_fixture["planner"].assert_not_awaited()
        self.last_fixture["receipt"].assert_not_awaited()
        self.assertEqual(self.last_fixture["result"]["state"], "stale_callback_skipped")
        self.last_fixture["admission"].assert_awaited_once_with("fixture-company-id", "fixture-crawl-task")

    def test_admission_read_error_fails_closed_without_crawl_or_mutation(self):
        updates, store, send, retry, _ = self.run_fixture([HOME], admission_error=TimeoutError("Fixture DB read unknown"))
        updates.assert_not_awaited()
        store.put.assert_not_called()
        send.assert_not_called()
        retry.assert_not_called()
        self.last_fixture["capture"].assert_not_called()
        self.last_fixture["planner"].assert_not_awaited()
        self.assertEqual(self.last_fixture["result"]["state"], "admission_unknown")

    def test_receipt_db_failure_after_send_is_observe_only_not_a_paid_retry(self):
        from app.models.company import PipelineStatus
        updates, _, send, retry, log = self.run_fixture(
            [HOME], receipt_side_effect=[True, TimeoutError("Fixture receipt unknown"), True],
        )
        send.assert_called_once()
        retry.assert_not_called()
        self.assertEqual(self.last_fixture["result"]["state"], "dispatch_unknown")
        self.assertEqual(self.last_fixture["receipt"].await_args.args[-1], "unknown")
        self.assertNotIn(PipelineStatus.FAILED, [call.kwargs.get("pipeline_status") for call in updates.await_args_list])
        self.assertNotIn("task.crawl_company.completed", [call.args[2] for call in log.call_args_list])

    def test_no_send_when_pre_dispatch_receipt_fails_and_no_automatic_paid_retry(self):
        updates, _, send, retry, _ = self.run_fixture(
            [HOME], receipt_side_effect=[TimeoutError("Fixture prepare receipt unknown"), True],
        )
        send.assert_not_called()
        retry.assert_not_called()
        self.assertEqual(self.last_fixture["result"]["state"], "dispatch_unknown")
        self.assertEqual(self.last_fixture["receipt"].await_args.args[-1], "unknown")

    def test_failed_required_identity_capture_blocks_cleaning_after_recording_failure(self):
        from app.models.company import PipelineStatus
        about = dict(PROFILE, url=f"{BASE}/company-profile", role="about", reason="Fixture identity")
        updates, _, send, retry, log = self.run_fixture(
            [HOME, about], extra_crawl_error=RuntimeError("页面抓取 HTTP 403"),
        )
        self.assert_failed_without_completion(updates, send, retry, log)
        send.assert_not_called()
        self.assertNotIn(PipelineStatus.CLEANING, [call.kwargs.get("pipeline_status") for call in updates.await_args_list])
        captured = next(call.kwargs["crawl_pages"] for call in updates.await_args_list if "crawl_pages" in call.kwargs)
        self.assertEqual(captured[1]["status"], "failed")
        self.assertIsNone(captured[1]["key"])
        self.assertIn("HTTP 403", captured[1]["reason"])
        self.assertIn("必需公司身份页", updates.await_args_list[-1].kwargs["pipeline_error"])

    def test_empty_capture_is_not_persisted(self):
        store = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "未成功持久化"):
            crawl._store_captured_html(store, "fixture/raw.html", "")
        store.put.assert_not_called()


class CrawlDispatchDBTests(unittest.IsolatedAsyncioTestCase):
    """Mock row locks verify stale owner guards and fresh JSON receipt merges."""
    def db_fixture(self, details, pipeline_status=None):
        from app.models.company import PipelineStatus
        company = SimpleNamespace(
            id=uuid.UUID("7c263dd6-e4f5-4430-8b72-d88789221cad"),
            geo_details=details, pipeline_status=pipeline_status or PipelineStatus.CLEANING,
        )
        db = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: company)),
            commit=AsyncMock(), rollback=AsyncMock(),
        )
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=db)
        manager.__aexit__ = AsyncMock(return_value=False)
        return company, db, manager

    async def test_stale_task_id_cannot_claim_or_write_anything(self):
        company, db, manager = self.db_fixture({"pipeline_dispatch": {"task_id": "current-crawl"}})
        with patch("app.core.database.async_session", return_value=manager):
            allowed = await crawl._admit_crawl_callback(str(company.id), "old-crawl")
        self.assertFalse(allowed)
        self.assertEqual(db.execute.await_count, 1)
        db.commit.assert_not_awaited()
        db.rollback.assert_awaited_once()

    async def test_matching_task_is_admitted_without_extra_mutation(self):
        company, db, manager = self.db_fixture({"pipeline_dispatch": {"task_id": "current-crawl"}})
        with patch("app.core.database.async_session", return_value=manager):
            allowed = await crawl._admit_crawl_callback(str(company.id), "current-crawl")
        self.assertTrue(allowed)
        self.assertEqual(db.execute.await_count, 1)
        db.commit.assert_not_awaited()

    async def test_legacy_admission_freezes_actual_task_id_once(self):
        company, db, manager = self.db_fixture({"preserve": {"source": "fixture"}})
        with patch("app.core.database.async_session", return_value=manager):
            allowed = await crawl._admit_crawl_callback(str(company.id), "legacy-crawl")
        self.assertTrue(allowed)
        self.assertEqual(db.execute.await_count, 2)
        values = db.execute.await_args.args[0].compile().params
        metadata = values["geo_details"]
        self.assertEqual(metadata["pipeline_dispatch"]["task_id"], "legacy-crawl")
        self.assertTrue(metadata["pipeline_dispatch"]["legacy_claimed"])
        self.assertEqual(metadata["preserve"], {"source": "fixture"})
        db.commit.assert_awaited_once()

    async def test_unknown_dispatch_receipt_preserves_cleaning_and_unrelated_metadata(self):
        company, db, manager = self.db_fixture({
            "pipeline_dispatch": {"task_id": "current-crawl", "state": "running"},
            "pipeline_quality": {"preserve": True}, "other": [1, 2],
        })
        with patch("app.core.database.async_session", return_value=manager):
            written = await crawl._record_clean_dispatch(str(company.id), "current-crawl", "clean-child", "unknown")
        self.assertTrue(written)
        values = db.execute.await_args.args[0].compile().params
        metadata = values["geo_details"]
        self.assertNotIn("pipeline_status", values)
        self.assertEqual(metadata["pipeline_dispatch"]["task_id"], "current-crawl")
        self.assertEqual(metadata["pipeline_dispatch"]["clean_task_id"], "clean-child")
        self.assertEqual(metadata["pipeline_dispatch"]["clean_state"], "unknown")
        self.assertEqual(metadata["pipeline_dispatch"]["state"], "unknown")
        self.assertEqual(metadata["pipeline_quality"], {"preserve": True})
        self.assertEqual(metadata["other"], [1, 2])
        self.assertIn("COMPANY_CLEAN_DISPATCH_UNKNOWN", values["pipeline_error"])
        db.commit.assert_awaited_once()

    async def test_unknown_receipt_never_overwrites_downstream_status_or_error(self):
        from app.models.company import PipelineStatus
        for state in (PipelineStatus.GRAPH_BUILDING, PipelineStatus.FAILED, PipelineStatus.COMPLETED):
            with self.subTest(state=state):
                company, db, manager = self.db_fixture({"pipeline_dispatch": {"task_id": "current-crawl"}}, state)
                with patch("app.core.database.async_session", return_value=manager):
                    written = await crawl._record_clean_dispatch(str(company.id), "current-crawl", "clean-child", "unknown")
                self.assertTrue(written)
                values = db.execute.await_args.args[0].compile().params
                self.assertNotIn("pipeline_status", values)
                self.assertNotIn("pipeline_error", values)
                self.assertNotEqual(values["geo_details"]["pipeline_dispatch"].get("state"), "unknown")

    async def test_stale_owner_cannot_write_clean_dispatch_receipt(self):
        company, db, manager = self.db_fixture({"pipeline_dispatch": {"task_id": "new-crawl"}})
        with patch("app.core.database.async_session", return_value=manager):
            written = await crawl._record_clean_dispatch(str(company.id), "old-crawl", "old-clean", "unknown")
        self.assertFalse(written)
        self.assertEqual(db.execute.await_count, 1)
        db.commit.assert_not_awaited()
        db.rollback.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
