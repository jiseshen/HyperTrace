import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.branch import branch_hypotheses
from core.hypothesis_set import Hypothesis, HypothesisSet, WorkingBelief
from core.preference_tracer import apply_belief_retrieval, replace_global_priors_from_current_belief, retrieve_belief_for_update
from core.summary import retrieve_inference_profile, retrieve_long_term_summary_items
from core.utils import TracerConfig, TracerContext
from data import Conversation, Turn, UserData
from model import EmbedConfig, GenerationConfig
from prompt import load_prompt_adapter
from run_prism_ablation import (
    PRISM_ABLATION_USER_IDS,
    ablation_record_files,
    select_prism_ablation_users,
    target_ablation_trace_users,
)


def fake_embed(contents, embed_cfg):
    return np.ones((len(contents), embed_cfg.dim), dtype=np.float32)


def make_user(user_id: str) -> UserData:
    return UserData(
        user_id=user_id,
        conversations=[
            Conversation(
                conversation_id=f"{user_id}_c1",
                turns=[
                    Turn(
                        turn_id=f"{user_id}_t1",
                        user_message="Pick a response.",
                        candidates=["A direct answer.", "A vague answer."],
                        chosen="A direct answer.",
                    )
                ],
            )
        ],
    )


class FakeBranchModel:
    async def async_generate(self, prompts, *args, **kwargs):
        return [
            {
                "output": {
                    "action": "replace",
                    "relevance": "none",
                    "justification": "unrelated",
                    "updated_hypothesis": {
                        "category": "replacement",
                        "content": f"replacement slot {idx}",
                    },
                }
            }
            for idx, _ in enumerate(prompts)
        ]


class FakeSummaryModel:
    def generate(self, *args, **kwargs):
        return {"output": "retrieved summary"}


class FakeRetrieveSet:
    def __init__(self):
        self.items = [
            (Hypothesis(id="h1", category="a", content="one"), 0.2, 0.70),
            (Hypothesis(id="h2", category="b", content="two"), 0.4, 0.90),
            (Hypothesis(id="h3", category="c", content="three"), 0.1, 0.80),
        ]
        self.hypotheses = {item[0].id: item[0] for item in self.items}
        self.global_prior = {item[0].id: item[1] for item in self.items}
        self.replaced = None
        self.retrieve_calls = 0

    def retrieve_hypotheses_with_scores(self, query, top_k=5):
        self.retrieve_calls += 1
        return (
            [item[0] for item in self.items[:top_k]],
            [item[1] for item in self.items[:top_k]],
            [item[2] for item in self.items[:top_k]],
        )

    def get_similarity(self, key1, key2):
        return 0.0

    def replace_belief_priors(self, ids, weights):
        self.replaced = (list(ids), [float(w) for w in weights])
        for hid, weight in zip(ids, weights):
            self.global_prior[hid] = float(weight)


class PrismAblationTests(unittest.TestCase):
    def test_selects_fixed_prism_ablation_users_and_ignores_stray_records(self):
        users = [make_user("stray")] + [make_user(user_id) for user_id in reversed(PRISM_ABLATION_USER_IDS)]
        selected = select_prism_ablation_users(users)

        self.assertEqual([user.user_id for user in selected], PRISM_ABLATION_USER_IDS)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_record = root / f"{PRISM_ABLATION_USER_IDS[0]}.json"
            first_record.write_text("{}", encoding="utf-8")
            (root / "stray.json").write_text("{}", encoding="utf-8")

            target_users, finished_target_ids, stray_finished_ids = target_ablation_trace_users(selected, root)
            self.assertEqual(finished_target_ids, {PRISM_ABLATION_USER_IDS[0]})
            self.assertEqual(stray_finished_ids, {"stray"})
            self.assertNotIn(PRISM_ABLATION_USER_IDS[0], {user.user_id for user in target_users})
            self.assertEqual(ablation_record_files(root), [first_record])

    def test_no_topic_hypothesis_set_excludes_category_from_store_and_logs(self):
        embed_cfg = EmbedConfig(backend="fake", model="fake", dim=3)
        with patch("core.hypothesis_set.embed", side_effect=fake_embed):
            repo = HypothesisSet(n_hypotheses=5, embed_config=embed_cfg, use_topics=False)
            ids = repo.add_hypotheses([
                {"category": "Politics", "content": "The user prefers direct answers."}
            ])

        hid = ids[0]
        self.assertEqual(repo.hypotheses[hid].category, "")
        self.assertEqual(list(repo.vector_store.contents.values()), ["The user prefers direct answers."])
        self.assertNotIn("Category", repo.hypotheses[hid].format(include_category=False))

        belief = WorkingBelief(ids=ids, priors=[1.0], repo=repo)
        self.assertNotIn("category", belief.log_dict(include_category=False)[0])

    def test_flat5_branch_replaces_slots_without_reinit_or_growth(self):
        embed_cfg = EmbedConfig(backend="fake", model="fake", dim=3)
        with patch("core.hypothesis_set.embed", side_effect=fake_embed):
            repo = HypothesisSet(n_hypotheses=5, embed_config=embed_cfg)
            ids = repo.add_hypotheses([
                {"category": "old", "content": f"old slot {idx}"}
                for idx in range(5)
            ])
        context = TracerContext(
            model=FakeBranchModel(),
            hypothesis_set=repo,
            current_belief=WorkingBelief(ids=ids, priors=[1, 1, 1, 1, 1], repo=repo),
            tracer_config=TracerConfig(hypothesis_update_mode="flat5"),
            generation_config=GenerationConfig(max_retries=1),
            prompts=load_prompt_adapter("prism_flat5_ablation"),
        )
        turn = Turn(
            user_message="Current evidence.",
            candidates=["chosen", "rejected"],
            chosen="chosen",
        )

        with patch("core.hypothesis_set.embed", side_effect=fake_embed):
            status = branch_hypotheses([turn], "[CandidateSet]\n[]", context)

        self.assertTrue(status["flat5"])
        self.assertNotIn("reinit", status)
        self.assertEqual(context.belief.ids, ids)
        self.assertEqual(len(repo.hypotheses), 5)
        self.assertTrue(all(repo.hypotheses[hid].content.startswith("replacement slot") for hid in ids))

    def test_retrieve_replace_normalizes_retrieved_priors_and_replaces_global_priors(self):
        repo = FakeRetrieveSet()
        context = TracerContext(
            model=None,
            hypothesis_set=repo,
            tracer_config=TracerConfig(
                hypothesis_update_mode="retrieve_replace",
                belief_retrieve_top_k=2,
                belief_retrieve_pool_k=3,
            ),
            generation_config=GenerationConfig(max_retries=1),
            prompts=load_prompt_adapter("prism"),
        )

        status = retrieve_belief_for_update("query", context)
        replace_global_priors_from_current_belief(context)

        self.assertTrue(status["success"])
        self.assertEqual(context.belief.ids, ["h2", "h3"])
        self.assertAlmostEqual(float(context.belief.weights.sum()), 1.0, places=6)
        self.assertEqual(repo.replaced[0], ["h2", "h3"])
        self.assertAlmostEqual(repo.global_prior["h2"], 0.8, places=6)
        self.assertAlmostEqual(repo.global_prior["h3"], 0.2, places=6)

    def test_retrieve_replace_can_share_inference_retrieval_for_belief_update(self):
        repo = FakeRetrieveSet()
        context = TracerContext(
            model=FakeSummaryModel(),
            hypothesis_set=repo,
            tracer_config=TracerConfig(
                hypothesis_update_mode="retrieve_replace",
                inference_profile_source="retrieved",
                inference_retrieve_top_k=2,
                inference_retrieve_pool_k=3,
                belief_retrieve_top_k=2,
                belief_retrieve_pool_k=3,
            ),
            generation_config=GenerationConfig(max_retries=1),
            prompts=load_prompt_adapter("prism"),
        )
        history = [
            Turn(user_message="Current evidence.", candidates=["chosen", "rejected"], chosen="chosen")
        ]

        profile, diagnostics = retrieve_inference_profile(
            history,
            context,
            include_belief_retrieval=True,
        )
        applied = apply_belief_retrieval(diagnostics["belief_retrieval"], context)

        self.assertEqual(profile, "retrieved summary")
        self.assertTrue(applied)
        self.assertEqual(repo.retrieve_calls, 1)
        self.assertEqual(diagnostics["belief_retrieval"]["source"], "shared_inference_retrieval")
        self.assertEqual(context.belief.ids, ["h2", "h3"])

    def test_summary_long_term_retrieval_excludes_current_and_low_prior_items(self):
        repo = FakeRetrieveSet()
        repo.items.extend([
            (Hypothesis(id="h4", category="d", content="four"), 0.03, 0.95),
            (Hypothesis(id="h5", category="e", content="five"), 0.3, 0.75),
            (Hypothesis(id="h6", category="f", content="six"), 0.25, 0.74),
            (Hypothesis(id="h7", category="g", content="seven"), 0.2, 0.73),
            (Hypothesis(id="h8", category="h", content="eight"), 0.2, 0.72),
            (Hypothesis(id="h9", category="i", content="nine"), 0.2, 0.71),
        ])
        repo.hypotheses = {item[0].id: item[0] for item in repo.items}
        repo.global_prior = {item[0].id: item[1] for item in repo.items}
        context = TracerContext(
            model=None,
            hypothesis_set=repo,
            current_belief=WorkingBelief(ids=["h2", "h3"], priors=[0.8, 0.2], repo=repo),
            tracer_config=TracerConfig(
                summary_profile_source="belief_with_retrieved_long_term",
                summary_retrieve_pool_k=9,
                summary_long_term_top_k=5,
                summary_min_prior=0.1,
            ),
            generation_config=GenerationConfig(max_retries=1),
            prompts=load_prompt_adapter("prism"),
        )

        items = retrieve_long_term_summary_items(context)

        self.assertEqual([item["id"] for item in items], ["h5", "h6", "h7", "h8", "h9"])
        self.assertNotIn("h2", [item["id"] for item in items])
        self.assertNotIn("h3", [item["id"] for item in items])
        self.assertNotIn("h4", [item["id"] for item in items])

    def test_flat5_prompt_adapter_is_separate_from_prism_adapter(self):
        prism = load_prompt_adapter("prism")
        flat5 = load_prompt_adapter("prism_flat5_ablation")

        self.assertNotIn("Flat-5 ablation mode", prism.branching)
        self.assertIn("Flat-5 ablation mode", flat5.branching)


if __name__ == "__main__":
    unittest.main()
