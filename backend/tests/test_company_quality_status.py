"""Isolated saved-receipt contract tests: no DB, provider, graph or vector calls."""
import ast
import asyncio
import hashlib
import threading
import uuid
from datetime import datetime, timezone
import copy
import enum
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from typing import Optional
from pydantic import BaseModel, Field

# Real response classes, without importing unrelated email/auth package exports.
_schema_tree = ast.parse((BACKEND / "app/schemas/company.py").read_text(encoding="utf-8"))
_schema_nodes = [node for node in _schema_tree.body if isinstance(node, ast.ClassDef)
                 and node.name in {"PipelineSelectedPage", "PipelineStatusResponse", "SubmitCompanyResponse"}]
_schema_scope = {"Optional": Optional, "BaseModel": BaseModel, "Field": Field}
exec(compile(ast.fix_missing_locations(ast.Module(body=_schema_nodes, type_ignores=[])),
             "company.py:real-response-classes", "exec"), _schema_scope)
PipelineStatusResponse = _schema_scope["PipelineStatusResponse"]
SubmitCompanyResponse = _schema_scope["SubmitCompanyResponse"]


class PipelineStatus(enum.Enum):
    PENDING = "pending"
    CRAWLING = "crawling"
    CLEANING = "cleaning"
    GRAPH_BUILDING = "graph_building"
    VECTORIZING = "vectorizing"
    COMPLETED = "completed"
    FAILED = "failed"


class PublishStatus(enum.Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    PUBLISHED = "published"


class FixtureHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


def fixture_quality():
    return {
        "status": "complete", "run_id": "fixture-saved-run",
        "stages": {
            "crawl": {"status": "complete", "document_count": 1},
            "clean": {"status": "complete", "verified": True},
            "graph": {"status": "complete", "verified": True, "entity_count": 7},
            "vector": {"status": "complete", "verified": True, "vector_count": 3},
        },
        "entity_count": 7, "vector_count": 3,
        "source_urls": ["https://example.invalid/about"],
    }


def fixture_company(quality=None, publication=PublishStatus.DRAFT):
    return SimpleNamespace(
        id="fixture-company", name="Fixture Company", url="https://example.invalid",
        pipeline_status=PipelineStatus.COMPLETED, pipeline_error=None,
        publish_status=publication, short_description="A saved company summary.",
        geo_details={"pipeline_quality": quality} if quality is not None else {},
        crawl_pages=[{"url": "https://example.invalid/about", "status": "captured", "role": "about"}],
    )


def production_functions(company):
    # Execute the actual route bodies while replacing all runtime dependencies.
    # Stripping decorators only prevents FastAPI registration during this unit test.
    tree = ast.parse((BACKEND / "app/api/routes/companies.py").read_text(encoding="utf-8"))
    names = {"_public_dispatch_metadata", "_saved_pipeline_quality", "_pipeline_stage_verified", "_pipeline_quality_passed", "get_pipeline_status", "submit_company_for_review"}
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in functions:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    scope = {
        "DbSession": object, "OptionalUser": object, "PipelineStatus": PipelineStatus,
        "PublishStatus": PublishStatus, "PipelineStatusResponse": PipelineStatusResponse,
        "HTTPException": FixtureHTTPException,
        "status": SimpleNamespace(HTTP_409_CONFLICT=409),
        "get_company_by_identifier": AsyncMock(return_value=company),
        "company_profile_needs_hydration": lambda _: False,
        "ensure_company_profile": AsyncMock(side_effect=AssertionError("no paid hydration allowed")),
    }
    exec(compile(module, "companies.py:isolated-real-route", "exec"), scope)
    return scope


class CompanyQualityStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_receipt_survives_the_api_response_model(self):
        quality = fixture_quality()
        company = fixture_company(quality)
        scope = production_functions(company)
        db = SimpleNamespace(commit=AsyncMock(), execute=AsyncMock())
        result = await scope["get_pipeline_status"](company.id, db)
        encoded = result.model_dump()
        self.assertEqual(encoded["pipeline_quality"], quality)
        self.assertEqual(encoded["company_url"], company.url)
        self.assertEqual(encoded["progress"], 100)
        self.assertIn("草稿", encoded["current_activity"])
        db.commit.assert_not_awaited()
        db.execute.assert_not_awaited()
        scope["ensure_company_profile"].assert_not_awaited()

    async def test_legacy_completed_is_unverified_not_a_fake_full_success(self):
        scope = production_functions(fixture_company())
        result = await scope["get_pipeline_status"]("fixture-company", object())
        self.assertEqual(result.status, "completed", "keep the saved legacy status distinct from quality")
        self.assertIsNone(result.pipeline_quality)
        self.assertEqual(result.progress, 0)
        self.assertIn("尚无完整核验", result.current_activity)
        scope["ensure_company_profile"].assert_not_awaited()

    async def test_failed_or_degraded_receipt_preserves_publication_fact(self):
        for quality_status in ("failed", "degraded"):
            for publication in (PublishStatus.DRAFT, PublishStatus.PENDING_REVIEW, PublishStatus.PUBLISHED):
                quality = fixture_quality()
                quality["status"] = quality_status
                quality["stages"]["vector"] = {"status": "failed", "error": "fixture-vector-missing"}
                scope = production_functions(fixture_company(quality, publication))
                result = await scope["get_pipeline_status"]("fixture-company", object())
                self.assertEqual(result.pipeline_quality, quality)
                self.assertEqual(result.publish_status, publication.value)
                self.assertEqual(result.progress, 0)
                self.assertIn("未通过", result.current_activity)
                scope["ensure_company_profile"].assert_not_awaited()

    async def test_passed_receipt_activity_does_not_invent_submission(self):
        for publication, fact in ((PublishStatus.PENDING_REVIEW, "提交后台审核"), (PublishStatus.PUBLISHED, "审核发布")):
            scope = production_functions(fixture_company(fixture_quality(), publication))
            result = await scope["get_pipeline_status"]("fixture-company", object())
            self.assertIn(fact, result.current_activity)
            self.assertEqual(result.publish_status, publication.value)

    async def test_review_gate_rejects_missing_failed_degraded_or_partial_receipts_without_hydration(self):
        bad_quality = [None, {}, {"status": "complete"}, {"status": "passed", "stages": {}}]
        for status_value in ("failed", "degraded"):
            item = fixture_quality()
            item["status"] = status_value
            bad_quality.append(item)
        for name in ("clean", "graph", "vector"):
            item = fixture_quality()
            item["stages"][name] = {"status": "unknown"}
            bad_quality.append(item)
        for quality in bad_quality:
            scope = production_functions(fixture_company(quality))
            with self.assertRaises(FixtureHTTPException) as caught:
                await scope["submit_company_for_review"]("fixture-company", object(), None)
            self.assertEqual(caught.exception.status_code, 409)
            self.assertIn("质量核验", caught.exception.detail)
            scope["ensure_company_profile"].assert_not_awaited()

    async def test_passed_receipt_cannot_trigger_implicit_paid_profile_hydration(self):
        scope = production_functions(fixture_company(fixture_quality()))
        scope["company_profile_needs_hydration"] = lambda _: True
        with self.assertRaises(FixtureHTTPException) as caught:
            await scope["submit_company_for_review"]("fixture-company", object(), None)
        self.assertEqual(caught.exception.status_code, 409)
        scope["ensure_company_profile"].assert_not_awaited()

    async def test_already_published_review_response_is_idempotent_and_read_only(self):
        scope = production_functions(fixture_company(None, PublishStatus.PUBLISHED))
        result = await scope["submit_company_for_review"]("fixture-company", object(), None)
        self.assertEqual(result["status"], "published")
        scope["ensure_company_profile"].assert_not_awaited()

    async def test_profile_alias_and_required_status_values_are_strict(self):
        scope = production_functions(fixture_company())
        quality = fixture_quality()
        quality["stages"]["profile"] = quality["stages"].pop("clean")
        self.assertTrue(scope["_pipeline_quality_passed"](quality))
        for bad_status in ("completed", "success", "unknown", "degraded"):
            item = copy.deepcopy(quality)
            item["stages"]["graph"]["status"] = bad_status
            self.assertFalse(scope["_pipeline_quality_passed"](item))

    async def test_completion_requires_verified_counts_for_every_required_stage(self):
        scope = production_functions(fixture_company())
        for stage_name, key, bad_value in (("crawl", "document_count", 0), ("clean", "verified", False),
                                           ("graph", "entity_count", 0), ("vector", "vector_count", 0),
                                           ("graph", "verified", False), ("vector", "verified", False)):
            quality = fixture_quality()
            quality["stages"][stage_name][key] = bad_value
            self.assertFalse(scope["_pipeline_quality_passed"](quality))
        quality = fixture_quality()
        quality["status"] = "passed"
        for stage in quality["stages"].values():
            stage["status"] = "passed"
        self.assertTrue(scope["_pipeline_quality_passed"](quality), "legacy alias still requires actual artifact evidence")

    async def test_malformed_geo_details_is_unknown_and_unchanged(self):
        for details in (None, [], "invalid", {"pipeline_quality": []}):
            company = fixture_company()
            company.geo_details = details
            scope = production_functions(company)
            self.assertIsNone(scope["_saved_pipeline_quality"](company))
            self.assertEqual(company.geo_details, details)

    async def test_unknown_dispatch_is_visible_without_claiming_queue_or_analysis_success(self):
        company = fixture_company()
        company.pipeline_status = PipelineStatus.PENDING
        company.geo_details["pipeline_dispatch"] = {"task_id": "same-fixture-task", "state": "unknown"}
        scope = production_functions(company)
        result = await scope["get_pipeline_status"]("fixture-company", object())
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.pipeline_dispatch, {"task_id": "same-fixture-task", "state": "unknown"})
        self.assertIn("仅观察同一任务", result.current_activity)
        self.assertIsNone(result.pipeline_quality)
        scope["ensure_company_profile"].assert_not_awaited()

    async def test_poll_route_has_no_hydration_or_external_service_call(self):
        tree = ast.parse((BACKEND / "app/api/routes/companies.py").read_text(encoding="utf-8"))
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_pipeline_status")
        called = {node.func.id for node in ast.walk(route) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue(called.isdisjoint({"ensure_company_profile", "get_ai_runtime_config", "get_ai_client", "httpx"}))
        self.assertEqual(sum(1 for node in ast.walk(route) if isinstance(node, ast.Await)), 1)



class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return self.name, value


class _CompanyFixture:
    id = _Column("id")
    url = _Column("url")

    def __init__(self, **values):
        self.__dict__.update(values)


class _QueryFixture:
    def __init__(self, operation):
        self.operation = operation
        self.clause = None
        self.locked = False
        self.populate_existing = False
        self.data = {}

    def where(self, clause):
        self.clause = clause
        return self

    def with_for_update(self):
        self.locked = True
        return self

    def execution_options(self, **values):
        self.populate_existing = values.get("populate_existing", False)
        return self

    def values(self, **data):
        self.data = data
        return self


class _PublicStore:
    def __init__(self, companies=()):
        self.companies = {company.id: company for company in companies}
        self.url_locks = {}
        self.row_locks = {}


class _PublicDb:
    def __init__(self, store):
        self.store = store
        self.held = []
        self.pending = []
        self.commits = 0
        self.rollbacks = 0
        self.queries = []

    def _company(self, clause):
        name, value = clause
        if name == "id":
            return self.store.companies.get(value)
        return next((company for company in self.store.companies.values() if company.url == value), None)

    async def execute(self, query, params=None):
        self.queries.append(query)
        if query == "advisory":
            key = params["lock_key"]
            assert isinstance(key, int) and -(2 ** 63) <= key < 2 ** 63
            lock = self.store.url_locks.setdefault(key, asyncio.Lock())
            await lock.acquire()
            self.held.append(lock)
            return None
        company = self._company(query.clause)
        if query.operation == "select":
            assert query.locked and query.populate_existing, "production reads must lock and refresh current identity"
            if company:
                lock = self.store.row_locks.setdefault(company.id, asyncio.Lock())
                await lock.acquire()
                self.held.append(lock)
                company = self._company(query.clause)
            return SimpleNamespace(scalar_one_or_none=lambda: company)
        assert self.held, "company writes must be admitted under a database lock"
        company.__dict__.update(copy.deepcopy(query.data))
        return None

    def add(self, company):
        assert self.held
        self.pending.append(company)

    def _unlock(self):
        for lock in reversed(self.held):
            lock.release()
        self.held.clear()

    async def commit(self):
        for company in self.pending:
            assert not any(existing.url == company.url for existing in self.store.companies.values())
            self.store.companies[company.id] = company
        self.pending.clear()
        self.commits += 1
        self._unlock()

    async def rollback(self):
        self.pending.clear()
        self.rollbacks += 1
        self._unlock()


def _failed_public_company():
    return _CompanyFixture(
        id=uuid.uuid4(), url="https://example.invalid", name="Existing fixture company",
        pipeline_status=PipelineStatus.FAILED, pipeline_error="fixture-old-failure",
        publish_status=PublishStatus.DRAFT, submitted_by=None,
        geo_details={"score": {"preserved": 73}, "pipeline_quality": {"status": "failed", "run_id": "old-fixture"}},
        crawl_pages=[{"url": "https://example.invalid/about", "key": "saved-source-key"}],
        crawl_candidates=[{"url": "https://example.invalid/about"}],
    )


def _public_functions():
    tree = ast.parse((BACKEND / "app/api/routes/companies.py").read_text(encoding="utf-8"))
    names = {"_public_dispatch_time", "_public_dispatch_metadata", "_public_url_lock_key",
             "_record_public_pipeline_dispatch", "_dispatch_public_company", "submit_company"}
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in functions:
        node.decorator_list = []
    scope = {
        "asyncio": asyncio, "hashlib": hashlib, "uuid": uuid, "datetime": datetime, "timezone": timezone,
        "DbSession": object, "OptionalUser": object, "SubmitCompanyRequest": object,
        "SubmitCompanyResponse": SubmitCompanyResponse, "Company": _CompanyFixture,
        "PipelineStatus": PipelineStatus, "PublishStatus": PublishStatus,
        "HTTPException": FixtureHTTPException,
        "status": SimpleNamespace(HTTP_422_UNPROCESSABLE_ENTITY=422, HTTP_409_CONFLICT=409),
        "PUBLIC_ACTIVE_PIPELINE_STATUSES": {PipelineStatus.PENDING, PipelineStatus.CRAWLING, PipelineStatus.CLEANING,
                                             PipelineStatus.GRAPH_BUILDING, PipelineStatus.VECTORIZING},
        "PUBLIC_DISPATCH_TIMEOUT_SECONDS": 1.0,
        "select": lambda _: _QueryFixture("select"), "update": lambda _: _QueryFixture("update"),
        "text": lambda sql: "advisory" if sql == "SELECT pg_advisory_xact_lock(:lock_key)" else None,
        "normalize_company_url": lambda value: value.rstrip("/"),
        "resolve_async_ai_access": AsyncMock(return_value=None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 "companies.py:isolated-real-public-submit", "exec"), scope)
    return scope


def _broker_module(send):
    module = ModuleType("app.core.celery_app")
    module.celery_app = SimpleNamespace(send_task=send)
    return module


class PublicCompanyDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_failed_submissions_admit_only_one_stable_paid_run(self):
        company = _failed_public_company()
        store = _PublicStore([company])
        first, second = _PublicDb(store), _PublicDb(store)
        scope = _public_functions()
        send = MagicMock(return_value=SimpleNamespace(id="fixture-broker-result"))
        user = SimpleNamespace(id=uuid.uuid4())
        with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
            results = await asyncio.gather(
                scope["submit_company"](SimpleNamespace(url=company.url), first, user),
                scope["submit_company"](SimpleNamespace(url=company.url + "/"), second, user),
            )
        scope["resolve_async_ai_access"].assert_awaited_once()
        send.assert_called_once()
        task_id = send.call_args.kwargs["task_id"]
        self.assertEqual(str(uuid.UUID(task_id)), task_id)
        self.assertFalse(send.call_args.kwargs["retry"])
        self.assertEqual(send.call_args.kwargs["args"], [str(company.id), company.url])
        self.assertEqual({result.company_id for result in results}, {str(company.id)})
        self.assertEqual({result.task_id for result in results}, {task_id})
        self.assertTrue(all(result.observe_only for result in results))
        self.assertEqual(company.pipeline_status, PipelineStatus.PENDING)
        self.assertEqual(company.geo_details["score"], {"preserved": 73})
        self.assertEqual(company.geo_details["pipeline_previous_quality"]["run_id"], "old-fixture")
        self.assertEqual(company.crawl_pages[0]["key"], "saved-source-key")
        self.assertEqual(company.geo_details["pipeline_dispatch"]["state"], "submitted")
        self.assertEqual(len(store.companies), 1)
        self.assertEqual(scope["resolve_async_ai_access"].await_args.kwargs["current_user"], user)

    async def test_concurrent_new_url_creates_only_one_company_and_one_task(self):
        store = _PublicStore()
        scope = _public_functions()
        send = MagicMock(return_value=SimpleNamespace(id="fixture-broker-result"))
        with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
            results = await asyncio.gather(*[
                scope["submit_company"](SimpleNamespace(url="https://example.invalid/"), _PublicDb(store), None)
                for _ in range(3)
            ])
        self.assertEqual(len(store.companies), 1)
        send.assert_called_once()
        scope["resolve_async_ai_access"].assert_awaited_once()
        self.assertEqual(len({result.company_id for result in results}), 1)
        self.assertEqual(len({result.task_id for result in results}), 1)
        self.assertEqual(sum(not result.resumed for result in results), 1)

    async def test_broker_error_is_dispatch_unknown_and_repost_cannot_create_another_paid_task(self):
        company = _failed_public_company()
        store = _PublicStore([company])
        scope = _public_functions()
        send = MagicMock(side_effect=RuntimeError("fixture-broker-error-do-not-persist-raw-text"))
        with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
            result = await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None)
            again = await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None)
        self.assertEqual(result.status, "dispatch_unknown")
        self.assertEqual(result.dispatch_state, "unknown")
        self.assertEqual(again.status, "dispatch_unknown")
        self.assertEqual(result.task_id, again.task_id)
        self.assertEqual(company.pipeline_status, PipelineStatus.PENDING)
        self.assertIn("尚未确认", company.pipeline_error)
        self.assertNotIn("do-not-persist", company.pipeline_error)
        self.assertEqual(company.geo_details["score"]["preserved"], 73)
        send.assert_called_once()
        scope["resolve_async_ai_access"].assert_awaited_once()

    async def test_dispatch_timeout_retains_original_task_identity_and_is_observe_only(self):
        company = _failed_public_company()
        store = _PublicStore([company])
        scope = _public_functions()
        scope["PUBLIC_DISPATCH_TIMEOUT_SECONDS"] = 0.04
        release = threading.Event()
        def delayed_send(*args, **kwargs):
            release.wait(0.3)
            return SimpleNamespace(id=kwargs["task_id"])
        send = MagicMock(side_effect=delayed_send)
        try:
            with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
                first = await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None)
                second = await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None)
            self.assertEqual(first.status, "dispatch_unknown")
            self.assertEqual(second.status, "dispatch_unknown")
            self.assertEqual(first.task_id, second.task_id)
            send.assert_called_once()
            self.assertEqual(company.pipeline_status, PipelineStatus.PENDING)
        finally:
            release.set()

    async def test_late_broker_receipt_merges_latest_worker_quality_instead_of_replacing_it(self):
        company = _failed_public_company()
        store = _PublicStore([company])
        scope = _public_functions()
        def worker_started(*args, **kwargs):
            company.pipeline_status = PipelineStatus.CLEANING
            company.geo_details = {**company.geo_details, "pipeline_quality": {"status": "running", "run_id": "current-worker"},
                                   "worker_fact": "preserved-new-result"}
            return SimpleNamespace(id=kwargs["task_id"])
        send = MagicMock(side_effect=worker_started)
        with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
            await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None)
        self.assertEqual(company.pipeline_status, PipelineStatus.CLEANING)
        self.assertEqual(company.geo_details["pipeline_quality"]["run_id"], "current-worker")
        self.assertEqual(company.geo_details["worker_fact"], "preserved-new-result")
        self.assertEqual(company.geo_details["pipeline_dispatch"]["state"], "submitted")

    async def test_unknown_dispatch_blocks_repost_even_if_worker_has_written_failed_status(self):
        company = _failed_public_company()
        company.geo_details["pipeline_dispatch"] = {"task_id": "original-fixture-task", "state": "unknown"}
        scope = _public_functions()
        send = MagicMock()
        with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
            result = await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(_PublicStore([company])), None)
        self.assertEqual(result.status, "dispatch_unknown")
        self.assertEqual(result.task_id, "original-fixture-task")
        self.assertTrue(result.observe_only)
        send.assert_not_called()
        scope["resolve_async_ai_access"].assert_not_awaited()

    async def test_stale_receipt_task_id_cannot_mutate_newer_admission(self):
        company = _failed_public_company()
        company.geo_details["pipeline_dispatch"] = {"task_id": "newer-task", "state": "submitted"}
        scope = _public_functions()
        db = _PublicDb(_PublicStore([company]))
        await scope["_record_public_pipeline_dispatch"](db, company.id, "old-task", "unknown")
        self.assertEqual(company.geo_details["pipeline_dispatch"], {"task_id": "newer-task", "state": "submitted"})
        self.assertEqual(db.commits, 0)
        self.assertEqual(db.rollbacks, 1)

    async def test_public_submit_keeps_original_access_policy_and_never_elevates_identity(self):
        company = _failed_public_company()
        scope = _public_functions()
        scope["resolve_async_ai_access"].side_effect = FixtureHTTPException(402, "fixture-platform-policy-denied")
        db = _PublicDb(_PublicStore([company]))
        send = MagicMock()
        try:
            with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
                with self.assertRaises(FixtureHTTPException):
                    await scope["submit_company"](SimpleNamespace(url=company.url), db, None)
        finally:
            await db.rollback()  # emulate the request-scoped dependency's exception cleanup
        send.assert_not_called()
        self.assertEqual(company.pipeline_status, PipelineStatus.FAILED)
        self.assertIsNone(scope["resolve_async_ai_access"].await_args.kwargs["current_user"])
        self.assertEqual(db.commits, 0)

    async def test_cancelled_wait_does_not_release_or_reissue_original_task(self):
        company = _failed_public_company()
        store = _PublicStore([company])
        scope = _public_functions()
        scope["PUBLIC_DISPATCH_TIMEOUT_SECONDS"] = 0.5
        started, release = threading.Event(), threading.Event()
        def delayed_send(*args, **kwargs):
            started.set()
            release.wait(0.4)
            return SimpleNamespace(id=kwargs["task_id"])
        send = MagicMock(side_effect=delayed_send)
        try:
            with patch.dict(sys.modules, {"app.core.celery_app": _broker_module(send)}):
                operation = asyncio.create_task(scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None))
                while not started.is_set():
                    await asyncio.sleep(0.001)
                operation.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await operation
                again = await scope["submit_company"](SimpleNamespace(url=company.url), _PublicDb(store), None)
            self.assertEqual(again.status, "dispatch_unknown")
            self.assertEqual(again.task_id, send.call_args.kwargs["task_id"])
            self.assertEqual(company.pipeline_status, PipelineStatus.PENDING)
            send.assert_called_once()
            scope["resolve_async_ai_access"].assert_awaited_once()
        finally:
            release.set()

    async def test_production_dispatch_wait_is_exactly_eight_seconds_and_never_has_silent_success_pass(self):
        tree = ast.parse((BACKEND / "app/api/routes/companies.py").read_text(encoding="utf-8"))
        timeout = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "PUBLIC_DISPATCH_TIMEOUT_SECONDS" for target in node.targets))
        self.assertEqual(timeout, 8.0)
        submit = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "submit_company")
        self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(submit)))


if __name__ == "__main__":
    unittest.main()
