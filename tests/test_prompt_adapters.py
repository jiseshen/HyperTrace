import unittest

from prompt import load_prompt_adapter
from prompt.base import PROFILE_EVALUATION_PROMPT
from prompt.personamem_adapter import PERSONAMEM_PROFILE_EVALUATION_PROMPT


class PromptAdapterTests(unittest.TestCase):
    def test_prism_adapter_uses_base_profile_eval_prompt(self):
        prompts = load_prompt_adapter("prism")
        self.assertEqual(prompts.profile_evaluation, PROFILE_EVALUATION_PROMPT)
        self.assertIn("Internal Plausibility", prompts.profile_evaluation)
        self.assertIn("demographic compatibility check", prompts.profile_evaluation)

    def test_personamem_adapter_overrides_profile_eval_prompt(self):
        prompts = load_prompt_adapter("personamem_v2")
        self.assertEqual(prompts.profile_evaluation, PERSONAMEM_PROFILE_EVALUATION_PROMPT)
        self.assertIn("PersonaMem-v2 ground truth", prompts.profile_evaluation)
        self.assertIn("Memory boundaries", prompts.profile_evaluation)

    def test_unknown_adapter_fails_clearly(self):
        with self.assertRaises(ValueError):
            load_prompt_adapter("unknown")


if __name__ == "__main__":
    unittest.main()
