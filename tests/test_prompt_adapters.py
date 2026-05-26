import unittest
from dataclasses import fields

from prompt import load_prompt_adapter
from prompt.base import PROFILE_EVALUATION_PROMPT, PromptSet
from prompt.personamem_adapter import PERSONAMEM_PROFILE_EVALUATION_PROMPT
from prompt.prism_adapter import PRISM_PROFILE_EVALUATION_PROMPT


TRACE_PROMPT_NAMES = [
    "skip",
    "preprocessing",
    "initialization",
    "likelihood",
    "branching",
    "axis",
    "merge",
    "perturb",
    "consolidate",
    "summary",
    "profile",
    "response",
    "prediction",
    "response_evaluation",
]

GT_METADATA_FIELD_TOKENS = [
    "expanded_persona",
    "pref_type",
    "prev_pref",
    "sensitive_info",
    "gt_profile",
]


def render_prompt(prompt_name: str, prompt: str) -> str:
    kwargs = {
        "user_message": "Can you help me plan dinner?",
        "candidates": "[CandidateSet]\n[]",
        "n": 2,
        "n_hypotheses": 2,
        "prev_turns": "User: Earlier question\nModel: Earlier answer",
        "retrieved_hypotheses": "h1: The user likes practical answers.",
        "hypothesis": "The user prefers vegetable-forward home cooking.",
        "current_hypothesis": "h1: The user prefers concise instructions.",
        "hypotheses": "h1: The user prefers concise instructions.",
        "collapsed_cluster": "The user prefers concise instructions.",
        "conversation_history": "User: Earlier question\nModel: Earlier answer",
        "global_axes_summary": "cooking style, specificity",
        "K": 2,
        "consolidated_hypotheses": "The user generally likes practical answers.",
        "current_hypotheses": "0.7 The user prefers concise instructions.",
        "profile": "The user prefers concise, practical cooking advice.",
        "l": 20,
        "r": 60,
        "current_message": "Can you help me plan dinner?",
        "current_turn": "User: Can you help me plan dinner?\nCandidates:\n[C1]\nA\n\n[C2]\nB\n\n",
        "c": 2,
        "adapted": "Make a quick vegetable stew.",
        "survey": "[PersonaMem-v2 Ground Truth Profile]\nSelf preferences: ...",
    }
    try:
        return prompt.format(**kwargs)
    except Exception as exc:
        raise AssertionError(f"{prompt_name} failed to render") from exc


class PromptAdapterTests(unittest.TestCase):
    def test_prism_adapter_uses_base_profile_eval_prompt(self):
        prompts = load_prompt_adapter("prism")
        self.assertNotEqual(prompts.profile_evaluation, PROFILE_EVALUATION_PROMPT)
        self.assertEqual(prompts.profile_evaluation, PRISM_PROFILE_EVALUATION_PROMPT)
        self.assertIn("Internal Plausibility", prompts.profile_evaluation)
        self.assertIn("demographic compatibility check", prompts.profile_evaluation)

    def test_personamem_adapter_overrides_profile_eval_prompt(self):
        prompts = load_prompt_adapter("personamem_v2")
        self.assertEqual(prompts.profile_evaluation, PERSONAMEM_PROFILE_EVALUATION_PROMPT)
        self.assertIn("PersonaMem-v2", prompts.profile_evaluation)
        self.assertIn("Ground truth profile", prompts.profile_evaluation)
        self.assertIn("Privacy and ownership", prompts.profile_evaluation)

    def test_all_prompt_fields_render_with_runtime_variables(self):
        self.assertEqual(
            [field.name for field in fields(PromptSet)],
            TRACE_PROMPT_NAMES + ["profile_evaluation"],
        )
        for adapter_name in ("prism", "personamem_v2"):
            prompts = load_prompt_adapter(adapter_name)
            for prompt_name in TRACE_PROMPT_NAMES + ["profile_evaluation"]:
                rendered = render_prompt(prompt_name, getattr(prompts, prompt_name))
                self.assertIsInstance(rendered, str)
                self.assertNotIn("{user_message}", rendered)

    def test_prism_adapter_preserves_base_prompt_semantics(self):
        prompts = load_prompt_adapter("prism")
        self.assertIn("Prioritize conversational style/value preferences over topic facts.", prompts.initialization)
        self.assertIn("demographic compatibility check", prompts.profile_evaluation)
        self.assertNotIn("PersonaMem-v2 guidance", prompts.initialization)

    def test_personamem_tracing_prompts_keep_choice_signal_and_exclude_gt_metadata(self):
        prompts = load_prompt_adapter("personamem_v2")
        self.assertIn("choice: chosen | rejected", prompts.initialization)
        self.assertIn("choice: chosen | rejected", prompts.branching)
        self.assertIn("chosen vs rejected", prompts.initialization)
        self.assertIn("chosen vs rejected", prompts.branching)
        self.assertIn("user_specific_need", prompts.preprocessing)
        self.assertIn("uses background", prompts.preprocessing)
        self.assertIn("[background_fact]", prompts.initialization)
        self.assertIn("PersonaMem memory units", prompts.initialization)
        self.assertIn("A topic change by itself is not evidence", prompts.branching)
        self.assertIn("action=\"revise\"", prompts.branching)
        self.assertIn("profile cues are relevant", prompts.prediction)
        self.assertIn("adaptation_plan", prompts.response)
        self.assertIn("do-not-remember", prompts.profile)
        self.assertIn("ownership audit", prompts.profile)
        self.assertIn("who=others", prompts.profile)
        self.assertIn("one-off information requests", prompts.profile)

        for prompt_name in TRACE_PROMPT_NAMES:
            prompt = getattr(prompts, prompt_name)
            for field_name in GT_METADATA_FIELD_TOKENS:
                self.assertNotIn(field_name, prompt, msg=f"{field_name} leaked into {prompt_name}")

    def test_unknown_adapter_fails_clearly(self):
        with self.assertRaises(ValueError):
            load_prompt_adapter("unknown")


if __name__ == "__main__":
    unittest.main()
