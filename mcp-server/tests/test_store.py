import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hypertrace_mcp.store import MemoryStore


def proposal(description, *, likelihood=5, baseline=1, action="new", hid=None,
             kind="web app", axis="testing", detail=None):
    return {"action": action, "id": hid, "description": description,
            "detail": detail or description, "project_kind": kind,
            "preference_axis": axis, "likelihood": likelihood, "baseline": baseline}


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)

    def save(self, event, *items, mode="subagents"):
        return self.store.save_feedback(event, "Please test before finishing", "demo", list(items), mode)

    def test_provisional_creation_reinforcement_and_idempotency(self):
        created = self.save("turn-1", proposal("Run focused tests before finishing"))
        hid = created["hypotheses"][0]["id"]
        initial = created["hypotheses"][0]["weight"]
        self.assertLessEqual(initial, 0.45)
        self.assertEqual(self.save("turn-1", proposal("Unrelated")), {"duplicate": True, "event_id": "turn-1"})
        updated = self.save("turn-2", proposal("Run focused tests before finishing", action="reuse", hid=hid))
        self.assertGreater(updated["hypotheses"][0]["weight"], initial)
        contradicted = self.save("turn-3", proposal("Run focused tests before finishing", action="reuse", hid=hid,
                                                    likelihood=0, baseline=5))
        self.assertLess(contradicted["hypotheses"][0]["weight"], updated["hypotheses"][0]["weight"])
        self.assertEqual(len(MemoryStore(self.path).browse()["hypotheses"]), 1)

    def test_main_agent_scores_have_less_influence(self):
        a = self.save("a", proposal("Prefer small changes"))["hypotheses"][0]
        b = self.save("b", proposal("Prefer short patches"))["hypotheses"][0]
        direct = self.save("c", proposal(a["description"], action="reuse", hid=a["id"]))["hypotheses"][0]
        fallback = self.save("d", proposal(b["description"], action="reuse", hid=b["id"]), mode="main_agent")["hypotheses"][0]
        self.assertGreater(direct["weight"], fallback["weight"])

    def test_category_discovery_aliases_and_weight_ranking(self):
        first = self.save("1", proposal("Write a narrow regression test", kind="Web Apps"))["hypotheses"][0]
        second = self.save("2", proposal("Explain tradeoffs before coding", kind="web apps", axis="planning"))["hypotheses"][0]
        self.assertEqual(len([c for c in self.store.browse()["categories"] if c["kind"] == "project_kind"]), 1)
        self.store.alias_category("preference_axis", "test habits", "testing")
        self.save("3", proposal("Check representative edge cases", kind="web apps", axis="test habits"))
        page = self.store.browse(project_kinds=["web apps"], preference_axes=["test habits"])
        self.assertEqual(len(page["hypotheses"]), 2)
        self.assertEqual(page["hypotheses"][0]["preference_axis"], "testing")
        self.assertNotIn(second["id"], [x["id"] for x in page["hypotheses"]])
        self.save("4", proposal(first["description"], action="reuse", hid=first["id"]))
        self.assertEqual(self.store.browse()["hypotheses"][0]["id"], first["id"])

    def test_reject_invalid_event_atomically(self):
        with self.assertRaises(ValueError):
            self.save("bad", proposal("Valid item"), proposal("Invalid score", likelihood=float("nan")))
        self.assertEqual(self.store.browse()["hypotheses"], [])
        self.assertFalse(self.store.browse()["categories"])
        with self.assertRaises(ValueError):
            self.store.save_feedback("secret", "api_key=abc123456", "demo", [proposal("Prefer tests")])
        self.assertEqual(self.store.browse()["hypotheses"], [])
        with self.assertRaises(ValueError):
            self.save("weak", proposal("Mention testing in every final answer", likelihood=3, baseline=2))
        self.assertEqual(self.store.browse()["hypotheses"], [])

    def test_concurrent_clients_share_memory(self):
        def write(index):
            client = MemoryStore(self.path)
            return client.save_feedback(str(index), "Use focused checks", "demo",
                                        [proposal(f"Check case {index} before finishing")])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(write, range(12)))
        self.assertEqual(len({item["hypotheses"][0]["id"] for item in results}), 12)
        self.assertEqual(len(self.store.browse(limit=20)["hypotheses"]), 12)

    def test_consolidation_and_undo(self):
        a = self.save("a", proposal("Run focused tests before finishing"))["hypotheses"][0]
        b = self.save("b", proposal("Run focused tests before finalizing"))["hypotheses"][0]
        queue = self.store.next_consolidation()["candidates"]
        self.assertEqual(len(queue), 1)
        merge = self.store.finish_consolidation(queue[0]["queue_id"], "merge", "Run focused tests before finishing",
                                                "The user favors focused verification before reporting completion.")
        self.assertEqual(len(self.store.browse()["hypotheses"]), 1)
        self.assertEqual(len(self.store.details([merge["id"]])[0]["evidence"]), 2)
        self.store.manage(merge["id"], "undo_merge")
        self.assertEqual({x["id"] for x in self.store.browse()["hypotheses"]}, {a["id"], b["id"]})

    def test_forget_removes_evidence(self):
        item = self.save("a", proposal("Prefer concise reports"))["hypotheses"][0]
        self.store.manage(item["id"], "forget")
        self.assertEqual(self.store.browse()["hypotheses"], [])
        self.assertEqual(self.store.details([item["id"]])[0]["evidence"], [])


if __name__ == "__main__":
    unittest.main()
