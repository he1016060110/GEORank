"""Row-lock admission/unknown-dispatch fixtures; never connect to a real database."""
import asyncio
import copy
import sys
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, Response
from sqlalchemy.sql.selectable import Select
from sqlalchemy.sql.dml import Update

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.routes import admin
from app.models.company import PipelineStatus, PublishStatus

CID = uuid.UUID("5315b4c8-b0b1-4ac9-9b12-090dd3cabd0a")
OWNER = SimpleNamespace(id=uuid.UUID("55718526-3bfa-4c3d-9a75-f8f89ea30c9a"))


class FakeStore:
    def __init__(self, pipeline_status=PipelineStatus.COMPLETED):
        self.company = SimpleNamespace(
            id=CID, url="https://company.example", submitted_by=OWNER.id,
            pipeline_status=pipeline_status, publish_status=PublishStatus.PENDING_REVIEW,
            pipeline_error=None,
            geo_details={"unrelated": {"preserve": True}, "pipeline_quality": {
                "status": "degraded", "stages": {"vectors": {"point_count": 0}},
            }},
        )
        self.lock = asyncio.Lock()


class FakeDB:
    def __init__(self, store):
        self.store = store
        self.held = False
        self.commits = 0
        self.rollbacks = 0
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if isinstance(statement, Select):
            if statement._for_update_arg is not None and not self.held:
                await self.store.lock.acquire()
                self.held = True
            return SimpleNamespace(scalar_one_or_none=lambda: copy.deepcopy(self.store.company))
        if isinstance(statement, Update):
            for column, bound in statement._values.items():
                key = column if isinstance(column, str) else column.key
                setattr(self.store.company, key, copy.deepcopy(bound.value))
            return SimpleNamespace(rowcount=1)
        raise AssertionError("unexpected fixture statement")

    async def commit(self):
        self.commits += 1
        if self.held:
            self.held = False
            self.store.lock.release()

    async def rollback(self):
        self.rollbacks += 1
        if self.held:
            self.held = False
            self.store.lock.release()


class CompanyRetryAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def patches(self, send):
        return (
            patch.object(admin, "_load_user_for_usage", AsyncMock(return_value=OWNER)),
            patch.object(admin, "resolve_async_ai_access", AsyncMock(return_value={})),
            patch("app.core.celery_app.celery_app.send_task", send),
        )

    async def test_every_active_status_is_observe_only_without_dispatch(self):
        for state in admin.ACTIVE_PIPELINE_STATUSES:
            with self.subTest(state=state):
                store = FakeStore(state)
                store.company.geo_details["pipeline_dispatch"] = {"task_id": "old-task", "state": "unknown"}
                db = FakeDB(store)
                send = MagicMock()
                p1, p2, p3 = self.patches(send)
                with p1, p2 as ai_access, p3:
                    with self.assertRaises(HTTPException) as error:
                        await admin.retry_pipeline(str(CID), db, OWNER, Response())
                self.assertEqual(error.exception.status_code, 409)
                self.assertEqual(error.exception.detail["code"], "COMPANY_PIPELINE_IN_FLIGHT")
                self.assertTrue(error.exception.detail["observe_only"])
                self.assertEqual(error.exception.detail["task_id"], "old-task")
                self.assertEqual(db.commits, 0)
                self.assertEqual(db.rollbacks, 1)
                send.assert_not_called()
                ai_access.assert_not_awaited()

    async def test_invalid_and_missing_company_fail_before_dispatch(self):
        for company_id, missing, expected in (("not-a-uuid", False, 400), (str(CID), True, 404)):
            store = FakeStore()
            if missing:
                store.company = None
            db = FakeDB(store)
            send = MagicMock()
            p1, p2, p3 = self.patches(send)
            with p1, p2, p3, self.assertRaises(HTTPException) as error:
                await admin.retry_pipeline(company_id, db, OWNER, Response())
            self.assertEqual(error.exception.status_code, expected)
            send.assert_not_called()

    async def test_success_uses_same_company_explicit_task_id_and_preserves_review(self):
        store = FakeStore()
        db = FakeDB(store)
        send = MagicMock(return_value=SimpleNamespace(id="unused-task"))
        p1, p2, p3 = self.patches(send)
        with p1, p2, p3:
            result = await admin.retry_pipeline(str(CID), db, OWNER, Response())
        self.assertEqual(result["status"], "retrying")
        self.assertEqual(result["dispatch_state"], "submitted")
        send.assert_called_once()
        self.assertEqual(send.call_args.args[0], "app.tasks.crawl.crawl_company_website")
        self.assertEqual(send.call_args.kwargs["args"], [str(CID), "https://company.example"])
        self.assertEqual(send.call_args.kwargs["task_id"], result["task_id"])
        self.assertFalse(send.call_args.kwargs["retry"])
        self.assertEqual(store.company.id, CID)
        self.assertEqual(store.company.publish_status, PublishStatus.PENDING_REVIEW)
        self.assertEqual(store.company.pipeline_status, PipelineStatus.PENDING)
        self.assertTrue(store.company.geo_details["unrelated"]["preserve"])
        self.assertIsNone(store.company.geo_details["pipeline_quality"])
        self.assertEqual(store.company.geo_details["pipeline_previous_quality"]["status"], "degraded")
        self.assertEqual(store.company.geo_details["pipeline_dispatch"]["state"], "submitted")
        self.assertEqual(db.commits, 2)

    async def test_two_concurrent_retries_publish_at_most_once(self):
        store = FakeStore()
        send = MagicMock(side_effect=lambda *a, **kw: time.sleep(0.025))
        p1, p2, p3 = self.patches(send)
        with p1, p2, p3:
            results = await asyncio.gather(
                admin.retry_pipeline(str(CID), FakeDB(store), OWNER, Response()),
                admin.retry_pipeline(str(CID), FakeDB(store), OWNER, Response()),
                return_exceptions=True,
            )
        successes = [item for item in results if isinstance(item, dict)]
        blocked = [item for item in results if isinstance(item, HTTPException)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].status_code, 409)
        send.assert_called_once()

    async def test_publish_failure_is_unknown_not_failed_and_cannot_be_redispatched(self):
        store = FakeStore()
        send = MagicMock(side_effect=RuntimeError("broker maybe dispatched fixture-secret"))
        p1, p2, p3 = self.patches(send)
        with p1, p2, p3:
            response = Response()
            result = await admin.retry_pipeline(str(CID), FakeDB(store), OWNER, response)
            self.assertEqual(response.status_code, 202)
            self.assertEqual(result["status"], "dispatch_unknown")
            self.assertTrue(result["observe_only"])
            self.assertNotIn("fixture-secret", str(result))
            self.assertNotIn("fixture-secret", store.company.pipeline_error)
            self.assertEqual(store.company.pipeline_status, PipelineStatus.PENDING)
            self.assertEqual(store.company.geo_details["pipeline_dispatch"]["state"], "unknown")
            with self.assertRaises(HTTPException) as repeated:
                await admin.retry_pipeline(str(CID), FakeDB(store), OWNER, Response())
            self.assertEqual(repeated.exception.status_code, 409)
        send.assert_called_once()

    async def test_dispatch_timeout_stays_pending_even_if_publish_finishes_later(self):
        store = FakeStore()
        send = MagicMock(side_effect=lambda *a, **kw: time.sleep(0.04))
        p1, p2, p3 = self.patches(send)
        with p1, p2, p3, patch.object(admin, "PIPELINE_DISPATCH_TIMEOUT_SECONDS", 0.001):
            result = await admin.retry_pipeline(str(CID), FakeDB(store), OWNER, Response())
        self.assertEqual(result["status"], "dispatch_unknown")
        self.assertEqual(store.company.pipeline_status, PipelineStatus.PENDING)
        self.assertEqual(store.company.geo_details["pipeline_dispatch"]["state"], "unknown")
        send.assert_called_once()

    async def test_receipt_merge_does_not_overwrite_new_worker_progress_or_quality(self):
        store = FakeStore(PipelineStatus.CLEANING)
        store.company.geo_details["pipeline_dispatch"] = {"task_id": "same-task", "state": "admitted"}
        before_quality = copy.deepcopy(store.company.geo_details["pipeline_quality"])
        await admin._record_pipeline_dispatch(FakeDB(store), CID, "same-task", "unknown")
        self.assertEqual(store.company.pipeline_status, PipelineStatus.CLEANING)
        self.assertIsNone(store.company.pipeline_error)
        self.assertEqual(store.company.geo_details["pipeline_quality"], before_quality)

    async def test_stale_receipt_does_not_overwrite_other_attempt(self):
        store = FakeStore()
        store.company.geo_details["pipeline_dispatch"] = {"task_id": "new-task", "state": "submitted"}
        db = FakeDB(store)
        await admin._record_pipeline_dispatch(db, CID, "old-task", "unknown")
        self.assertEqual(store.company.geo_details["pipeline_dispatch"]["task_id"], "new-task")
        self.assertEqual(db.commits, 0)
        self.assertEqual(db.rollbacks, 1)


def complete_saved_quality():
    return {
        "run_id": "fixture-verified-run", "status": "complete",
        "document_count": 3, "entity_count": 4, "vector_count": 6,
        "stages": {
            "crawl": {"status": "complete", "verified": True, "document_count": 3},
            "clean": {"status": "complete", "verified": True,
                      "extraction_quality": {"source": "llm", "status": "passed"}},
            "graph": {"status": "complete", "verified": True, "entity_count": 4},
            "vector": {"status": "complete", "verified": True, "vector_count": 6},
        },
    }


class CompanyPublishQualityGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_complete_without_quality_cannot_publish_or_trigger_ai(self):
        store = FakeStore()
        store.company.geo_details = {}
        db = FakeDB(store)
        with patch.object(admin, "ensure_company_profile", AsyncMock()) as hydrate:
            with self.assertRaises(HTTPException) as error:
                await admin.approve_company(str(CID), db, OWNER)
        self.assertEqual(error.exception.status_code, 409)
        self.assertIn("saved_pipeline_quality_missing", error.exception.detail["issues"])
        self.assertEqual(store.company.publish_status, PublishStatus.PENDING_REVIEW)
        self.assertEqual(db.commits, 0)
        self.assertEqual(db.rollbacks, 1)
        hydrate.assert_not_awaited()

    async def test_zero_or_missing_counts_cannot_publish(self):
        for count_key in ("document_count", "entity_count", "vector_count"):
            for invalid in (0, None, "6", True, -1):
                with self.subTest(count_key=count_key, invalid=invalid):
                    store = FakeStore()
                    quality = complete_saved_quality()
                    quality[count_key] = invalid
                    store.company.geo_details = {"pipeline_quality": quality}
                    db = FakeDB(store)
                    with self.assertRaises(HTTPException) as error:
                        await admin.approve_company(str(CID), db, OWNER)
                    self.assertEqual(error.exception.status_code, 409)
                    self.assertIn(count_key + "_missing_or_zero", error.exception.detail["issues"])
                    self.assertEqual(db.commits, 0)

    async def test_every_stage_requires_complete_and_verified_not_profile_passed(self):
        for stage_name in ("crawl", "clean", "graph", "vector"):
            for invalid in ({"status": "passed"}, {"verified": False}, {"verified": "true"}):
                with self.subTest(stage=stage_name, invalid=invalid):
                    store = FakeStore()
                    quality = complete_saved_quality()
                    quality["stages"][stage_name].update(invalid)
                    store.company.geo_details = {"pipeline_quality": quality}
                    db = FakeDB(store)
                    with self.assertRaises(HTTPException) as error:
                        await admin.approve_company(str(CID), db, OWNER)
                    self.assertEqual(error.exception.status_code, 409)
                    self.assertIn("stage_" + stage_name + "_not_verified", error.exception.detail["issues"])

    async def test_overall_passed_and_profile_fallback_are_not_complete(self):
        for change in ("overall", "profile", "counter_mismatch", "missing_run"):
            with self.subTest(change=change):
                store = FakeStore()
                quality = complete_saved_quality()
                if change == "overall":
                    quality["status"] = "passed"
                elif change == "profile":
                    quality["stages"]["clean"]["extraction_quality"]["source"] = "html_fallback"
                elif change == "counter_mismatch":
                    quality["stages"]["vector"]["vector_count"] = 9
                else:
                    quality.pop("run_id")
                store.company.geo_details = {"pipeline_quality": quality}
                with self.assertRaises(HTTPException) as error:
                    await admin.approve_company(str(CID), FakeDB(store), OWNER)
                self.assertEqual(error.exception.status_code, 409)

    async def test_verified_run_still_cannot_publish_if_pipeline_is_failed(self):
        store = FakeStore(PipelineStatus.FAILED)
        store.company.geo_details = {"pipeline_quality": complete_saved_quality()}
        with self.assertRaises(HTTPException) as error:
            await admin.approve_company(str(CID), FakeDB(store), OWNER)
        self.assertIn("pipeline_not_completed", error.exception.detail["issues"])

    async def test_successful_saved_verified_run_can_publish_without_model_hydration(self):
        store = FakeStore()
        store.company.geo_details = {"pipeline_quality": complete_saved_quality()}
        db = FakeDB(store)
        with (
            patch.object(admin, "company_profile_needs_hydration", return_value=False),
            patch.object(admin, "ensure_company_profile", AsyncMock()) as hydrate,
        ):
            result = await admin.approve_company(str(CID), db, OWNER)
        self.assertEqual(result, {"status": "published", "company_id": str(CID)})
        self.assertEqual(store.company.publish_status, PublishStatus.PUBLISHED)
        self.assertEqual(db.commits, 1)
        hydrate.assert_not_awaited()

    async def test_successful_receipt_cannot_implicitly_pay_to_hydrate_profile(self):
        store = FakeStore()
        store.company.geo_details = {"pipeline_quality": complete_saved_quality()}
        db = FakeDB(store)
        with (
            patch.object(admin, "company_profile_needs_hydration", return_value=True),
            patch.object(admin, "ensure_company_profile", AsyncMock()) as hydrate,
        ):
            with self.assertRaises(HTTPException) as error:
                await admin.approve_company(str(CID), db, OWNER)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(db.commits, 0)
        hydrate.assert_not_awaited()

    async def test_list_exposes_redacted_saved_receipts_in_one_query_not_n_details(self):
        company = FakeStore().company
        company.name = "Fixture company"
        company.short_description = "Fixture description"
        company.category = "manufacturing"
        company.is_geo_certified = False
        company.geo_score = 60
        company.upvotes = 0
        from datetime import datetime, timezone
        company.created_at = datetime.now(timezone.utc)
        company.updated_at = company.created_at
        quality = complete_saved_quality()
        quality["api_key"] = "fixture-secret"
        quality["stages"]["vector"]["request_body"] = {"authorization": "fixture-secret"}
        company.geo_details = {
            "pipeline_quality": quality,
            "pipeline_dispatch": {"task_id": "fixture-task", "state": "submitted", "token": "fixture-secret"},
        }
        db = SimpleNamespace(
            scalar=AsyncMock(return_value=1),
            execute=AsyncMock(return_value=SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [company]),
            )),
        )
        with (
            patch.object(admin, "get_company_admin_detail", AsyncMock()) as detail,
            patch.object(admin, "ensure_company_profile", AsyncMock()) as hydrate,
        ):
            result = await admin.list_companies_admin(db, OWNER)
        self.assertEqual(result["items"][0]["pipeline_quality"]["status"], "complete")
        self.assertEqual(result["items"][0]["pipeline_dispatch"]["task_id"], "fixture-task")
        self.assertNotIn("fixture-secret", str(result))
        db.execute.assert_awaited_once()
        db.scalar.assert_awaited_once()
        detail.assert_not_awaited()
        hydrate.assert_not_awaited()

    def test_metadata_serializer_is_bounded_and_omits_secret_fields(self):
        company = FakeStore().company
        company.geo_details["pipeline_dispatch"] = {
            "task_id": "fixture", "llm_api_key": "fixture-secret",
            "nested": {"authorization": "fixture-secret", "safe": "safe"},
            "array": list(range(100)), "long": "x" * 5000,
        }
        result = admin._company_pipeline_metadata(company, "pipeline_dispatch")
        self.assertNotIn("fixture-secret", str(result))
        self.assertEqual(len(result["array"]), 64)
        self.assertEqual(len(result["long"]), 2048)
        self.assertEqual(result["nested"], {"safe": "safe"})


if __name__ == "__main__":
    unittest.main()
