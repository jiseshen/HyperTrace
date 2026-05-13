import json
import unittest

from data.personamem_v2 import (
    _parse_incorrect_answers,
    _parse_user_query,
    _records_to_users,
)


def _record(**overrides):
    base = {
        "persona_id": 7,
        "expanded_persona": json.dumps(
            {
                "short_persona": {"persona": "A careful home cook."},
                "age": 34,
                "gender": "Nonbinary",
                "occupation": {"title": "Teacher"},
                "personality": {"traits": ["precise", "warm"]},
                "speaking_style_with_chatbot": {"tone": "direct and friendly"},
                "hobbies_interests": ["gardening"],
            }
        ),
        "user_query": "{'role': 'user', 'content': 'What should I make for dinner?'}",
        "correct_answer": "Make a vegetable stew with precise steps.",
        "incorrect_answers": "['Order fast food.', 'Try a vague dinner idea.', 'Make dessert only.']",
        "topic_query": "Cooking",
        "preference": "Prefers vegetable-forward home cooking",
        "topic_preference": "Food",
        "pref_type": "neutral_preferences",
        "who": "self",
        "updated": False,
        "prev_pref": None,
        "sensitive_info": False,
        "related_conversation_snippet": "SECRET_SNIPPET",
    }
    base.update(overrides)
    return base


class PersonaMemV2LoaderTests(unittest.TestCase):
    def test_parse_user_query_from_dict_string(self):
        self.assertEqual(
            _parse_user_query("{'role': 'user', 'content': 'Hello there'}"),
            "Hello there",
        )

    def test_parse_incorrect_answers_from_list_string(self):
        self.assertEqual(
            _parse_incorrect_answers("['a', 'b', 'c']"),
            ["a", "b", "c"],
        )

    def test_records_to_users_groups_and_shuffles_deterministically(self):
        records = [
            _record(correct_answer="chosen one"),
            _record(user_query="{'role': 'user', 'content': 'Second question'}", correct_answer="chosen two"),
        ]
        users_a = _records_to_users(records, n_users=None, seed=123, split="benchmark_text")
        users_b = _records_to_users(records, n_users=None, seed=123, split="benchmark_text")

        self.assertEqual(len(users_a), 1)
        self.assertEqual(users_a[0].user_id, "personamem_v2_7")
        turns = users_a[0].conversations[0].turns
        self.assertEqual(len(turns), 2)
        self.assertTrue(all(len(turn.candidates) == 4 for turn in turns))
        self.assertTrue(all(turn.candidates[turn.chosen_idx] == turn.chosen for turn in turns))
        self.assertEqual(
            [turn.candidates for turn in users_a[0].conversations[0].turns],
            [turn.candidates for turn in users_b[0].conversations[0].turns],
        )

    def test_gt_profile_excludes_snippets_candidates_and_sensitive_details(self):
        records = [
            _record(
                correct_answer="ANSWER_SHOULD_NOT_APPEAR",
                incorrect_answers="['BAD1', 'BAD2', 'BAD3']",
            ),
            _record(
                preference="Do not remember 'Favors craft beer over mass-market beer brands' in memory",
                topic_preference="Beer",
                pref_type="ask_to_forget",
                updated=True,
                prev_pref="Favors craft beer over mass-market beer brands",
            ),
            _record(
                preference="SSN is 123-45-6789",
                topic_preference="Private ID",
                pref_type="sensitive_info",
                sensitive_info=True,
            ),
            _record(
                preference="A friend prefers spicy food",
                topic_preference="Food",
                who="others",
            ),
        ]
        profile = _records_to_users(records, n_users=None, seed=123, split="benchmark_text")[0].gt_profile

        self.assertIn("[PersonaMem-v2 Ground Truth Profile]", profile)
        self.assertIn("Prefers vegetable-forward home cooking", profile)
        self.assertIn("do-not-remember boundary", profile)
        self.assertIn("not be attributed to the user", profile)
        self.assertNotIn("SECRET_SNIPPET", profile)
        self.assertNotIn("ANSWER_SHOULD_NOT_APPEAR", profile)
        self.assertNotIn("BAD1", profile)
        self.assertNotIn("craft beer", profile.lower())
        self.assertNotIn("123-45-6789", profile)


if __name__ == "__main__":
    unittest.main()
