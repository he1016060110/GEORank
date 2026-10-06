"""Mock-only graph provenance/transaction contracts; never connects to Neo4j or AI."""
import copy
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "app/services/graph_store.py"
_COMPANY = "00000000-0000-0000-0000-000000000001"
_OTHER = "00000000-0000-0000-0000-000000000002"
_URL = "https://example.com/products/sensor"
_DIGEST = "a" * 64


def _load_module():
    # Isolate both imports while executing this file, without changing the real
    # app module or importing an environment/.env or initializing a real driver.
    neo4j = types.ModuleType("neo4j")
    neo4j.AsyncGraphDatabase = types.SimpleNamespace(driver=lambda *_args, **_kwargs: None)
    config = types.ModuleType("app.core.config")
    config.settings = types.SimpleNamespace(NEO4J_URI="bolt://fixture.invalid:7687",
                                           NEO4J_USER="fixture", NEO4J_PASSWORD="not-a-real-password")
    spec = importlib.util.spec_from_file_location("_graph_provenance_fixture", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"neo4j": neo4j, "app.core.config": config}):
        spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, record=None):
        self.record = record

    async def consume(self):
        return None

    async def single(self):
        return self.record


class _Tx:
    def __init__(self, state, driver):
        self.state, self.driver = state, driver

    async def run(self, query, **params):
        self.driver.queries.append((query, copy.deepcopy(params)))
        marker = next((line.strip() for line in query.splitlines() if "graph-store:" in line), "")
        if self.driver.fail_marker and self.driver.fail_marker in marker:
            raise RuntimeError("fixture write failure")
        company_id = params.get("company_id", params.get("id"))
        if "company-lock" in marker:
            company = self.state["companies"].setdefault(company_id, {"id": company_id})
            company.update(params["props"])
            company["graph_write_revision"] = company.get("graph_write_revision", 0) + 1
            return _Result({"company_id": company_id})
        if "archive-bindings" in marker:
            kept, archived = [], 0
            for binding in self.state["bindings"]:
                if binding["company_id"] == company_id and binding.get("graph_version", "") != params["graph_version"]:
                    history = copy.deepcopy(binding)
                    history.update(active=False, replaced_by_graph_version=params["graph_version"],
                                   archived_by_run_id=params["run_id"], archived_at=params["updated_at"])
                    self.state["history"].append(history)
                    archived += 1
                else:
                    kept.append(binding)
            self.state["bindings"] = kept
            return _Result({"archived_bindings": archived})
        if "upsert-entity" in marker:
            entity = dict(params["props"])
            entity.update({key: copy.deepcopy(value) for key, value in params.items() if key != "props"})
            self.state["entities"][params["entity_key"]] = entity
            binding = {"company_id": company_id, "graph_version": params["graph_version"],
                       "entity_key": params["entity_key"], "binding_id": params["entity_key"],
                       "active": True, "run_id": params["run_id"]}
            if not any(item.get("binding_id") == binding["binding_id"] for item in self.state["bindings"]):
                self.state["bindings"].append(binding)
            return _Result()
        if "upsert-relation" in marker:
            props = dict(params["props"])
            props.update({key: copy.deepcopy(value) for key, value in params.items() if key != "props"})
            self.state["edges"][params["relation_key"]] = {
                "relation_key": params["relation_key"], "type": params["type"],
                "from_key": params["from_key"], "to_key": params["to_key"], "properties": props,
            }
            return _Result()
        if "activate-version" in marker:
            company = self.state["companies"][company_id]
            company.update({"graph_" + key: copy.deepcopy(params[key]) for key in
                            ("run_id", "schema_version", "updated_at", "source_urls", "source_hashes",
                             "source_hashes_json", "source_sha256")})
            company["graph_version"] = params["graph_version"]
            return _Result()
        if "read-current" in marker:
            company = self.state["companies"].get(company_id)
            if company is None:
                return _Result()
            entities = [self.state["entities"][binding["entity_key"]] for binding in self.state["bindings"]
                        if binding["company_id"] == company_id and binding.get("active")
                        and binding.get("graph_version") == company.get("graph_version")
                        and binding["entity_key"] in self.state["entities"]]
            keys = {row["entity_key"] for row in entities}
            edges = [row for row in self.state["edges"].values() if row["from_key"] in keys
                     and row["to_key"] in keys and row["properties"].get("company_id") == company_id
                     and row["properties"].get("graph_version") == company.get("graph_version")]
            if self.driver.duplicate_readback:
                entities = entities + entities
                edges = edges + edges
            if self.driver.omit_readback_edges:
                edges = []
            if self.driver.inject_foreign_readback:
                entities.append({"entity_key": "foreign", "company_id": _OTHER, "graph_version": company.get("graph_version")})
                edges.append({"relation_key": "foreign", "from_key": "foreign", "to_key": "foreign",
                              "properties": {"company_id": _OTHER, "graph_version": company.get("graph_version")}})
            return _Result({"company": copy.deepcopy(company), "entities": copy.deepcopy(entities),
                            "relationships": copy.deepcopy(edges)})
        if query.startswith("MERGE (c:Company"):
            self.state["companies"].setdefault(company_id, {"id": company_id}).update(params["props"])
            return _Result()
        raise AssertionError("Unexpected graph fixture query")


class _Session:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute_write(self, callback, *args):
        self.driver.write_transactions += 1
        transaction_state = copy.deepcopy(self.driver.state)
        result = await callback(_Tx(transaction_state, self.driver), *args)
        self.driver.state = transaction_state
        return result

    async def execute_read(self, callback, *args):
        self.driver.read_transactions += 1
        return await callback(_Tx(copy.deepcopy(self.driver.state), self.driver), *args)


class _Driver:
    def __init__(self):
        self.state = {"companies": {}, "entities": {}, "bindings": [], "history": [],
                      "edges": {}, "legacy_nodes": [], "legacy_edges": []}
        self.queries = []
        self.write_transactions = self.read_transactions = 0
        self.fail_marker = None
        self.duplicate_readback = self.omit_readback_edges = self.inject_foreign_readback = False

    def session(self):
        return _Session(self)


class GraphCompanyProvenanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = _load_module()
        self.driver = _Driver()
        self.module._driver = self.driver
        self.entities = [{"name": "Sensor", "type": "Product", "props": {"description": "measured sensor"}},
                         {"name": "MEMS", "type": "Technology"}]
        self.relations = [{"from": "Sensor", "to": "MEMS", "type": "USES_TECH"}]

    async def write(self, company=_COMPANY, run="real-fixture-run-1", **kwargs):
        return await self.module.upsert_company_entities(
            company, self.entities, self.relations, source_urls=[_URL], run_id=run,
            source_hashes={_URL: _DIGEST}, source_sha256="b" * 64, **kwargs,
        )

    async def test_atomic_managed_write_persists_real_source_metadata_and_readback(self):
        result = await self.write(properties={"name": "Test company", "url": "https://example.com"})
        self.assertEqual(1, self.driver.write_transactions)
        self.assertEqual(0, self.driver.read_transactions)
        self.assertTrue(result["verified"])
        self.assertEqual((2, 1), (result["node_count"], result["relation_count"]))
        self.assertEqual([{"url": _URL, "sha256": _DIGEST}], json.loads(result["source_hashes_json"]))
        self.assertEqual("b" * 64, result["readback"]["source_sha256"])
        for entity in result["readback"]["entities"]:
            self.assertTrue(entity["entity_key"].startswith(_COMPANY + ":"))
            self.assertIn(entity["normalized_type"] + ":" + entity["normalized_name"], entity["entity_key"])
            self.assertEqual("real-fixture-run-1", entity["run_id"])
            self.assertEqual([_URL], entity["source_urls"])
            self.assertEqual([_DIGEST], entity["source_hashes"])
        edge = result["readback"]["relationships"][0]
        self.assertEqual(_COMPANY, edge["properties"]["company_id"])
        self.assertEqual(result["graph_version"], edge["properties"]["graph_version"])

    async def test_same_names_in_different_companies_never_share_entity_keys(self):
        first = await self.write()
        second = await self.write(company=_OTHER)
        first_keys = {row["entity_key"] for row in first["readback"]["entities"]}
        second_keys = {row["entity_key"] for row in second["readback"]["entities"]}
        self.assertFalse(first_keys & second_keys)
        self.assertEqual(first["readback"], await self.module.get_company_graph(_COMPANY))

    async def test_legacy_bindings_archive_only_target_company_without_deleting_history(self):
        self.driver.state["companies"] = {_COMPANY: {"id": _COMPANY}, _OTHER: {"id": _OTHER}}
        self.driver.state["bindings"] = [
            {"company_id": _COMPANY, "entity_key": "legacy-shared", "binding_id": "legacy-first"},
            {"company_id": _OTHER, "entity_key": "legacy-shared", "binding_id": "legacy-other"},
        ]
        self.driver.state["legacy_nodes"] = [{"name": "Sensor"}, {"name": "Partner"}]
        self.driver.state["legacy_edges"] = [{"from": "Sensor", "to": "Partner", "type": "COMPETES_WITH"}]
        preserved = copy.deepcopy(self.driver.state)
        result = await self.write()
        self.assertEqual(1, result["archived_bindings"])
        self.assertEqual(preserved["legacy_nodes"], self.driver.state["legacy_nodes"])
        self.assertEqual(preserved["legacy_edges"], self.driver.state["legacy_edges"])
        self.assertIn(preserved["bindings"][1], self.driver.state["bindings"])
        self.assertEqual("legacy-first", self.driver.state["history"][0]["binding_id"])
        self.assertFalse(self.driver.state["history"][0]["active"])
        self.assertEqual(["USES_TECH"], [row["type"] for row in result["readback"]["relationships"]])
        archive_query = next(query for query, _ in self.driver.queries if "archive-bindings" in query)
        self.assertIn("Company {id: $company_id}", archive_query)
        self.assertIn("HAS_ENTITY_HISTORY", archive_query)
        self.assertNotIn("DELETE old", archive_query)
        self.assertNotIn("DETACH DELETE", archive_query)

    async def test_new_version_retains_old_entity_properties_and_relations(self):
        first = await self.write()
        old_nodes = copy.deepcopy(self.driver.state["entities"])
        old_edges = copy.deepcopy(self.driver.state["edges"])
        self.entities[0]["props"]["description"] = "updated source evidence"
        second = await self.write(run="real-fixture-run-2")
        self.assertNotEqual(first["graph_version"], second["graph_version"])
        for key, value in old_nodes.items():
            self.assertEqual(value, self.driver.state["entities"][key])
        for key, value in old_edges.items():
            self.assertEqual(value, self.driver.state["edges"][key])
        self.assertEqual(2, second["archived_bindings"])
        self.assertEqual((2, 1), (second["node_count"], second["relation_count"]))

    async def test_replay_same_run_and_content_is_idempotent(self):
        first = await self.write()
        state = copy.deepcopy(self.driver.state)
        second = await self.write()
        self.assertEqual(first["graph_version"], second["graph_version"])
        self.assertEqual(0, second["archived_bindings"])
        for field in ("bindings", "history"):
            self.assertEqual(state[field], self.driver.state[field])
        self.assertEqual(len(state["entities"]), len(self.driver.state["entities"]))
        self.assertEqual(len(state["edges"]), len(self.driver.state["edges"]))

    async def test_normalized_duplicate_rows_and_path_multiplicity_do_not_inflate_counts(self):
        self.entities.append({"name": " sensor ", "type": "product"})
        self.relations.append({"from": " SENSOR ", "to": "mems", "type": "USES_TECH"})
        self.driver.duplicate_readback = True
        result = await self.write()
        self.assertEqual((2, 1), (result["node_count"], result["relation_count"]))
        graph = await self.module.get_company_graph(_COMPANY)
        self.assertEqual(2, graph["nodes"])
        self.assertEqual(2, len(graph["entities"]))
        self.assertEqual(1, len(graph["relationships"]))

    async def test_foreign_entities_and_edges_are_excluded_at_readback_boundary(self):
        self.driver.inject_foreign_readback = True
        result = await self.write()
        self.assertEqual((2, 1), (result["node_count"], result["relation_count"]))
        self.assertNotIn("foreign", {row["entity_key"] for row in result["readback"]["entities"]})

    async def test_relationship_queries_use_company_version_keys_not_global_names(self):
        await self.write()
        query, params = next((q, p) for q, p in self.driver.queries if "upsert-relation" in q)
        self.assertIn("company_id: $company_id, graph_version: $graph_version", query)
        self.assertIn("entity_key: $from_key", query)
        self.assertIn("entity_key: $to_key", query)
        self.assertNotIn("name:", query)
        self.assertTrue(params["from_key"].startswith(_COMPANY + ":"))
        read_query = next(q for q, _ in self.driver.queries if "read-current" in q)
        self.assertIn("b IN entity_nodes", read_query)
        self.assertNotIn("[*", read_query)

    async def test_unknown_or_ambiguous_relation_endpoints_fail_before_any_write(self):
        self.relations[0]["to"] = "Other company entity"
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            await self.write()
        self.assertEqual(0, self.driver.write_transactions)
        self.relations[0]["to"] = "MEMS"
        self.entities.append({"name": "Sensor", "type": "Company"})
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            await self.write()
        self.relations[0]["from_type"] = "Product"
        result = await self.write()
        self.assertEqual(3, result["node_count"])

    async def test_invalid_dynamic_types_and_reserved_bindings_fail_before_write(self):
        for unsafe in ("USES_TECH]->(x) DELETE x //", "HAS_ENTITY", "HAS_ENTITY_HISTORY"):
            self.relations[0]["type"] = unsafe
            with self.assertRaises(ValueError):
                await self.write()
        self.assertEqual(0, self.driver.write_transactions)
        self.relations[0]["type"] = "USES_TECH"
        self.entities[0]["type"] = "Product) DETACH DELETE e //"
        with self.assertRaises(ValueError):
            await self.write()

    async def test_reserved_entity_and_relationship_props_cannot_forge_company_or_run(self):
        self.entities[0]["props"].update(company_id=_OTHER, entity_key="foreign", run_id="forged")
        self.relations[0]["props"] = {"company_id": _OTHER, "graph_version": "forged"}
        result = await self.write()
        self.assertTrue(all(row["company_id"] == _COMPANY for row in result["readback"]["entities"]))
        self.assertEqual(_COMPANY, result["readback"]["relationships"][0]["properties"]["company_id"])

    async def test_partial_write_failure_rolls_back_archiving_and_new_nodes(self):
        await self.write()
        preserved = copy.deepcopy(self.driver.state)
        self.driver.fail_marker = "upsert-relation"
        with self.assertRaisesRegex(RuntimeError, "fixture write failure"):
            await self.write(run="real-fixture-run-2")
        self.assertEqual(preserved, self.driver.state)

    async def test_failed_readback_rolls_back_instead_of_claiming_success(self):
        await self.write()
        preserved = copy.deepcopy(self.driver.state)
        self.driver.omit_readback_edges = True
        with self.assertRaises(self.module.GraphPersistenceError):
            await self.write(run="real-fixture-run-2")
        self.assertEqual(preserved, self.driver.state)

    async def test_missing_provenance_stays_unknown_and_legacy_entry_point_is_scoped(self):
        result = await self.module.add_entities_and_relations(_COMPANY, self.entities, self.relations)
        self.assertEqual("", result["run_id"])
        self.assertEqual([], result["source_urls"])
        self.assertEqual([], result["source_hashes"])
        self.assertEqual("[]", result["source_hashes_json"])
        self.assertEqual("", result["source_sha256"])
        self.assertEqual(2, result["node_count"])

    async def test_missing_company_returns_zero_counts_without_fabricated_records(self):
        graph = await self.module.get_company_graph(_COMPANY)
        self.assertEqual((0, 0), (graph["node_count"], graph["relation_count"]))
        self.assertEqual([], graph["entities"])
        self.assertIsNone(graph["graph_version"])

    async def test_source_sha256_or_url_mismatch_rejected_before_write(self):
        for hashes in ({_URL: "not-a-hash"}, {"https://other.invalid": _DIGEST}):
            with self.assertRaises(ValueError):
                await self.module.upsert_company_entities(_COMPANY, self.entities, self.relations,
                                                         source_urls=[_URL], source_hashes=hashes)
        with self.assertRaises(ValueError):
            await self.module.upsert_company_entities(_COMPANY, self.entities, self.relations,
                                                     source_urls=["https://user:secret@example.com"], run_id="run")
        self.assertEqual(0, self.driver.write_transactions)


    async def test_real_claim_url_and_quote_aliases_persist_with_matching_page_hash_only(self):
        other_url = "https://example.com/about"
        self.entities[0].update(source_url=_URL, source_quote="measured sensor")
        self.relations[0].update(source_url=_URL, evidence="sensor uses MEMS")
        result = await self.module.upsert_company_entities(
            _COMPANY, self.entities, self.relations, source_urls=[_URL, other_url],
            source_hashes={_URL: _DIGEST, other_url: "c" * 64}, run_id="real-fixture-claims",
        )
        entity = next(row for row in result["readback"]["entities"] if row["name"] == "Sensor")
        self.assertEqual("measured sensor", entity["source_quote"])
        self.assertEqual([_URL], entity["source_urls"])
        self.assertEqual([_DIGEST], entity["source_hashes"])
        self.assertEqual("claim_reference", entity["provenance_scope"])
        self.assertEqual("sensor uses MEMS", result["readback"]["relationships"][0]["properties"]["evidence"])

    async def test_claim_url_outside_real_corpus_fails_before_any_write(self):
        self.relations[0]["source_url"] = "https://hallucination.invalid/not-crawled"
        with self.assertRaisesRegex(ValueError, "outside the supplied analysis corpus"):
            await self.write()
        self.assertEqual(0, self.driver.write_transactions)

    async def test_company_display_name_is_updated_but_cannot_replace_company_identity(self):
        result = await self.write(properties={"name": "Source-confirmed company", "id": _OTHER,
                                               "graph_version": "forged", "graph_run_id": "forged"})
        company = self.driver.state["companies"][_COMPANY]
        self.assertEqual("Source-confirmed company", company["name"])
        self.assertEqual(_COMPANY, company["id"])
        self.assertEqual(result["graph_version"], company["graph_version"])
        self.assertEqual("real-fixture-run-1", company["graph_run_id"])

if __name__ == "__main__":
    unittest.main()
