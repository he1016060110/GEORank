"""No-network/no-real-DB tests for the scoped embedding settings protocol."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.routes import admin
from app.services import runtime_settings


LOCAL = {
    "provider": "local_tei", "base_url": "http://host.docker.internal:45310/v1",
    "model": "intfloat/multilingual-e5-small", "dimensions": 384,
    "collection": "companies_e5_small_v1",
}
REMOTE = {
    "provider": "remote", "base_url": "https://embedding.example/v1",
    "model": "embedding-small", "dimensions": 1536, "collection": "companies",
}


def config_from(payload, api_key=""):
    return {
        **{admin.EMBEDDING_SETTING_FIELDS[k]: value for k, value in payload.items()},
        "embedding_api_key": api_key,
    }


class EmbeddingSettingsSchemaTests(unittest.TestCase):
    def test_local_requires_verified_four_part_identity(self):
        self.assertEqual(admin._normalize_embedding_payload(
            admin.EmbeddingSettingsRequest(**LOCAL)
        )["dimensions"], 384)
        for field, value in (
            ("base_url", "http://127.0.0.1:45310/v1"),
            ("base_url", "http://host.docker.internal:45311/v1"),
            ("base_url", "http://host.docker.internal:45310/v1?target=secret"),
            ("base_url", "http://user:pass@host.docker.internal:45310/v1"),
            ("model", "other-model"), ("dimensions", 1536), ("collection", "companies"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(HTTPException):
                admin._normalize_embedding_payload(admin.EmbeddingSettingsRequest(**{
                    **LOCAL, field: value,
                }))

    def test_dimensions_collections_and_extra_fields_are_bounded(self):
        for update in (
            {"dimensions": 0}, {"dimensions": 8193}, {"dimensions": "384"},
            {"dimensions": True}, {"collection": "../companies"},
            {"collection": "a" * 97}, {"provider": "unknown"},
            {"unknown": True}, {"model": "a" * 161}, {"api_key": "k" * 4097},
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                admin.EmbeddingSettingsRequest(**{**LOCAL, **update})

    def test_remote_still_blocks_private_urls_without_dns_or_network(self):
        with patch.object(admin.settings, "ALLOW_PRIVATE_LLM_PROVIDER_URLS", False):
            for url in (
                "http://169.254.169.254/latest", "https://127.0.0.1/v1",
                "https://localhost/v1", "https://api.example/v1?key=secret",
                "https://user:pass@api.example/v1", "https://api.example/v1#fragment",
            ):
                with self.subTest(url=url), self.assertRaises(HTTPException):
                    admin._normalize_embedding_payload(admin.EmbeddingSettingsRequest(**{
                        **REMOTE, "base_url": url,
                    }))
            admin._normalize_embedding_payload(admin.EmbeddingSettingsRequest(**LOCAL))
            self.assertFalse(admin.settings.ALLOW_PRIVATE_LLM_PROVIDER_URLS)

    def test_readiness_is_configuration_only_and_never_exposes_key(self):
        local = admin._serialize_embedding_config(config_from(LOCAL, "fixture-secret"))
        self.assertTrue(local["has_api_key"])
        self.assertEqual(local["readiness"]["state"], "configured")
        self.assertIsNone(local["readiness"]["connected"])
        self.assertFalse(local["readiness"]["collection_verified"])
        self.assertEqual(local["readiness"]["verification"], "configuration_only")
        self.assertNotIn("api_key", local)
        self.assertNotIn("fixture-secret", str(local))
        remote = admin._serialize_embedding_config(config_from(REMOTE))
        self.assertEqual(remote["readiness"]["state"], "blocked")
        self.assertTrue(remote["readiness"]["issues"])

    def test_runtime_exposes_provider_collection_and_five_second_ttl(self):
        config = runtime_settings._build_ai_runtime_config({
            **config_from(LOCAL), "llm_api_key": "fixture-llm-only",
        })
        self.assertEqual(config["embedding_provider"], "local_tei")
        self.assertEqual(config["embedding_collection"], "companies_e5_small_v1")
        self.assertEqual(config["embedding_dimensions"], 384)
        self.assertEqual(config["embedding_api_key"], "")
        self.assertEqual(runtime_settings._cache_ttl_seconds, 5)

    def test_remote_legacy_defaults_are_preserved(self):
        config = runtime_settings._build_ai_runtime_config({})
        self.assertEqual(config["embedding_provider"], "remote")
        self.assertEqual(config["embedding_collection"], admin.settings.QDRANT_COLLECTION)


class EmbeddingSettingsPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_save_does_not_touch_old_secret_llm_or_collections(self):
        current = config_from(REMOTE, "fixture-existing-secret")
        db = SimpleNamespace(commit=AsyncMock())
        owner = SimpleNamespace(id="fixture-owner")
        with (
            patch.object(admin, "_load_admin_embedding_config", AsyncMock(return_value=current)),
            patch.object(admin, "_store_setting_value", AsyncMock()) as store,
            patch.object(admin, "invalidate_runtime_settings_cache", AsyncMock()) as invalidate,
            patch.object(admin, "validate_provider_base_url", AsyncMock()) as network_check,
            patch.object(admin, "ai_client", MagicMock()) as ai,
        ):
            result = await admin.update_embedding_settings_admin(
                admin.EmbeddingSettingsRequest(**LOCAL), db, owner,
            )
        self.assertEqual(result["status"], "saved")
        self.assertTrue(result["has_api_key"])
        self.assertNotIn("fixture-existing-secret", str(result))
        self.assertEqual([call.args[2] for call in store.await_args_list],
                         list(admin.EMBEDDING_SETTING_FIELDS.values()))
        db.commit.assert_awaited_once()
        invalidate.assert_awaited_once()
        network_check.assert_not_awaited()
        self.assertEqual(ai.mock_calls, [])

    async def test_empty_or_masked_key_cannot_clear_existing_key(self):
        for key in (None, "", "••••••••••••••••"):
            with (
                self.subTest(key=key),
                patch.object(admin, "_load_admin_embedding_config", AsyncMock(
                    return_value=config_from(REMOTE, "fixture-existing-key"))),
                patch.object(admin, "_store_setting_value", AsyncMock()) as store,
                patch.object(admin, "invalidate_runtime_settings_cache", AsyncMock()),
            ):
                db = SimpleNamespace(commit=AsyncMock())
                response = await admin.update_embedding_settings_admin(
                    admin.EmbeddingSettingsRequest(**REMOTE, api_key=key),
                    db, SimpleNamespace(id="owner"),
                )
                self.assertTrue(response["has_api_key"])
                self.assertNotIn("embedding_api_key", [call.args[2] for call in store.await_args_list])

    async def test_changed_remote_endpoint_requires_explicit_replacement_key(self):
        with (
            patch.object(admin, "_load_admin_embedding_config", AsyncMock(
                return_value=config_from(REMOTE, "fixture-old-secret"))),
            patch.object(admin, "_store_setting_value", AsyncMock()) as store,
            patch.object(admin, "invalidate_runtime_settings_cache", AsyncMock()) as invalidate,
        ):
            db = SimpleNamespace(commit=AsyncMock())
            with self.assertRaises(HTTPException) as error:
                await admin.update_embedding_settings_admin(
                    admin.EmbeddingSettingsRequest(**{
                        **REMOTE, "base_url": "https://other.example/v1",
                    }), db, SimpleNamespace(id="owner"),
                )
            self.assertEqual(error.exception.status_code, 400)
            store.assert_not_awaited()
            db.commit.assert_not_awaited()
            invalidate.assert_not_awaited()

    async def test_new_remote_key_is_stored_sensitive_without_network_test(self):
        with (
            patch.object(admin, "_load_admin_embedding_config", AsyncMock(return_value=config_from(REMOTE))),
            patch.object(admin, "_store_setting_value", AsyncMock()) as store,
            patch.object(admin, "invalidate_runtime_settings_cache", AsyncMock()),
            patch.object(admin, "validate_provider_base_url", AsyncMock()) as network_check,
        ):
            response = await admin.update_embedding_settings_admin(
                admin.EmbeddingSettingsRequest(**REMOTE, api_key="fixture-new-key"),
                SimpleNamespace(commit=AsyncMock()), SimpleNamespace(id="owner"),
            )
            self.assertEqual(store.await_args_list[-1].args[2:4],
                             ("embedding_api_key", "fixture-new-key"))
            self.assertEqual(store.await_args_list[-1].kwargs["category"], "api_keys")
            self.assertTrue(response["has_api_key"])
            self.assertNotIn("fixture-new-key", str(response))
            network_check.assert_not_awaited()

    async def test_get_is_read_only_and_redacted(self):
        with patch.object(admin, "_load_admin_embedding_config", AsyncMock(
            return_value=config_from(LOCAL, "fixture-secret"))):
            db = MagicMock()
            result = await admin.get_embedding_settings_admin(db, SimpleNamespace())
            self.assertEqual(db.mock_calls, [])
            self.assertNotIn("fixture-secret", str(result))
            self.assertEqual(result["provider"], "local_tei")

    async def test_invalidation_is_awaited_after_commit(self):
        sequence = []
        async def commit():
            sequence.append("committed")
        async def invalidate():
            sequence.append("invalidated")
        with (
            patch.object(admin, "_load_admin_embedding_config", AsyncMock(return_value=config_from(LOCAL))),
            patch.object(admin, "_store_setting_value", AsyncMock()),
            patch.object(admin, "invalidate_runtime_settings_cache", invalidate),
        ):
            await admin.update_embedding_settings_admin(
                admin.EmbeddingSettingsRequest(**LOCAL),
                SimpleNamespace(commit=commit), SimpleNamespace(id="owner"),
            )
        self.assertEqual(sequence, ["committed", "invalidated"])


if __name__ == "__main__":
    unittest.main()
