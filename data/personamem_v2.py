import argparse
import ast
import json
import math
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

try:
    from .base import Conversation, Turn, UserData
except ImportError:
    from base import Conversation, Turn, UserData


DATASET_NAME = "bowen-upenn/PersonaMem-v2"
DATASET_CONFIG = "benchmark"
DEFAULT_SPLIT = "benchmark_text"
SUPPORTED_TEXT_SPLITS = {"benchmark_text", "train_text", "val_text"}
PERSONAMEM_GT_MARKER = "[PersonaMem-v2 Ground Truth Profile]"
MIN_USER_TURNS = 20


def _parse_structured_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            pass
    return value


def _parse_user_query(value: Any) -> str:
    parsed = _parse_structured_value(value)
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    raise ValueError(f"Unable to parse user_query: {value!r}")


def _parse_incorrect_answers(value: Any) -> List[str]:
    parsed = _parse_structured_value(value)
    if not isinstance(parsed, list):
        raise ValueError(f"incorrect_answers must parse to a list: {value!r}")
    answers = [str(item).strip() for item in parsed if str(item).strip()]
    if len(answers) != 3:
        raise ValueError(f"Expected 3 incorrect answers, got {len(answers)}")
    return answers


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(_as_text(item) for item in value if _as_text(item))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _as_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    return str(value).strip()


def _format_section(title: str, value: Any) -> str:
    text = _as_text(value)
    return f"{title}: {text}" if text else ""


def _add_unique(items: List[str], item: str) -> None:
    item = item.strip()
    if item and item not in items:
        items.append(item)


def _row_bool(row: Dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _profile_from_expanded_persona(expanded_persona: Any) -> List[str]:
    persona = _parse_structured_value(expanded_persona)
    if not isinstance(persona, dict):
        text = _as_text(persona)
        return [f"Expanded persona: {text}"] if text else []

    sections: List[str] = []
    short_persona = persona.get("short_persona")
    if isinstance(short_persona, dict):
        short_persona = short_persona.get("persona") or short_persona
    section = _format_section("Short persona", short_persona)
    if section:
        sections.append(section)

    context_keys = [
        "age",
        "gender",
        "sexual_orientation",
        "race_ethnicity",
        "nationality",
        "political_affiliation",
        "location",
        "languages_spoken",
        "family_background",
    ]
    context = {key: persona.get(key) for key in context_keys if persona.get(key)}
    section = _format_section("Demographics/context", context)
    if section:
        sections.append(section)

    education_occupation = {
        key: persona.get(key)
        for key in ("education", "occupation")
        if persona.get(key)
    }
    section = _format_section("Education/occupation", education_occupation)
    if section:
        sections.append(section)

    personality_values = {
        key: persona.get(key)
        for key in ("personality", "values_beliefs")
        if persona.get(key)
    }
    section = _format_section("Personality/values", personality_values)
    if section:
        sections.append(section)

    speaking_style = (
        persona.get("speaking_style_with_chatbot")
        or persona.get("speaking_style_to_chatbot")
    )
    section = _format_section("Speaking style with chatbot", speaking_style)
    if section:
        sections.append(section)

    stable_background = {
        key: persona.get(key)
        for key in (
            "hobbies_interests",
            "technology_use",
            "stereotypical_preferences",
            "anti_stereotypical_preferences",
            "neutral_preferences",
            "therapy_background",
            "health_and_medical_conditions",
        )
        if persona.get(key)
    }
    section = _format_section("Stable interests/background", stable_background)
    if section:
        sections.append(section)

    return sections


def _build_gt_profile(rows: List[Dict[str, Any]]) -> str:
    first = rows[0]
    sections = [
        PERSONAMEM_GT_MARKER,
        "",
        "[Persona]",
        *_profile_from_expanded_persona(first.get("expanded_persona")),
        "",
        "[Ground-truth benchmark preferences]",
    ]

    self_preferences: List[str] = []
    updated_preferences: List[str] = []
    memory_boundaries: List[str] = []
    sensitive_constraints: List[str] = []
    non_user_preferences: List[str] = []

    for row in rows:
        preference = _as_text(row.get("preference"))
        pref_type = _as_text(row.get("pref_type")) or "unknown"
        topic = _as_text(row.get("topic_preference")) or "unknown"
        who = (_as_text(row.get("who")) or "self").lower()
        updated = _row_bool(row, "updated")
        sensitive = _row_bool(row, "sensitive_info") or pref_type == "sensitive_info"
        prev_pref = _as_text(row.get("prev_pref"))

        if sensitive:
            _add_unique(
                sensitive_constraints,
                (
                    f"topic={topic}; type={pref_type}: sensitive/private information "
                    "may appear in evidence; evaluate profiles for safe abstraction only "
                    "and penalize exact private-detail retention."
                ),
            )
            continue

        if pref_type == "ask_to_forget" or preference.lower().startswith("do not remember"):
            _add_unique(
                memory_boundaries,
                (
                    f"topic={topic}: the user set a do-not-remember boundary; "
                    "do not reward retaining the forbidden fact verbatim."
                ),
            )
            continue

        if who == "others":
            _add_unique(
                non_user_preferences,
                (
                    f"topic={topic}; type={pref_type}: {preference} "
                    "(belongs to someone else; should not be attributed to the user)."
                ),
            )
            continue

        if updated and prev_pref:
            _add_unique(
                updated_preferences,
                (
                    f"topic={topic}; type={pref_type}: current preference is "
                    f"{preference}; previous preference was {prev_pref}."
                ),
            )
        else:
            _add_unique(
                self_preferences,
                f"topic={topic}; type={pref_type}: {preference}",
            )

    if self_preferences:
        sections.extend(["", "Self preferences:", *[f"- {item}" for item in self_preferences]])
    if updated_preferences:
        sections.extend(["", "Updated preferences:", *[f"- {item}" for item in updated_preferences]])
    if memory_boundaries:
        sections.extend(["", "Memory boundaries / do-not-remember:", *[f"- {item}" for item in memory_boundaries]])
    if sensitive_constraints:
        sections.extend(["", "Sensitive/private constraints:", *[f"- {item}" for item in sensitive_constraints]])
    if non_user_preferences:
        sections.extend(["", "Not the user's own preferences:", *[f"- {item}" for item in non_user_preferences]])

    return "\n".join(line for line in sections if line is not None)


def _shuffled_candidates(row: Dict[str, Any], seed: int, row_idx: int) -> tuple[List[str], str, int]:
    chosen = str(row.get("correct_answer", "")).strip()
    if not chosen:
        raise ValueError(f"Missing correct_answer for row {row_idx}")
    candidates = [chosen] + _parse_incorrect_answers(row.get("incorrect_answers"))
    persona_id = row.get("persona_id")
    rng = random.Random(f"{seed}:{persona_id}:{row_idx}")
    rng.shuffle(candidates)
    return candidates, chosen, candidates.index(chosen)


def _records_to_users(
    records: Iterable[Dict[str, Any]],
    n_users: Optional[int],
    seed: int,
    split: str,
    min_user_turns: int = MIN_USER_TURNS,
) -> List[UserData]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    user_order: List[str] = []

    for row_idx, row in enumerate(records):
        persona_id = row.get("persona_id")
        if persona_id is None:
            raise ValueError(f"Missing persona_id for row {row_idx}")
        uid = f"personamem_v2_{persona_id}"
        if uid not in grouped:
            grouped[uid] = []
            user_order.append(uid)
        normalized = dict(row)
        normalized["_row_idx"] = row_idx
        grouped[uid].append(normalized)

    users: List[UserData] = []
    for uid in user_order:
        rows = grouped[uid]
        turns = []
        for row in rows:
            row_idx = int(row["_row_idx"])
            candidates, chosen, chosen_idx = _shuffled_candidates(row, seed=seed, row_idx=row_idx)
            turns.append(
                Turn(
                    turn_id=f"{uid}_{row_idx}",
                    user_message=_parse_user_query(row.get("user_query")),
                    candidates=candidates,
                    chosen=chosen,
                    chosen_idx=chosen_idx,
                )
            )
        if len(turns) < min_user_turns:
            continue
        users.append(
            UserData(
                user_id=uid,
                conversations=[
                    Conversation(
                        conversation_id=f"{uid}_{split}",
                        turns=turns,
                    )
                ],
                gt_profile=_build_gt_profile(rows),
            )
        )

    if n_users is not None and len(users) > n_users:
        rng = random.Random(seed)
        users = rng.sample(users, n_users)
    return users


def load_personamem_v2(
    n_users: Optional[int] = None,
    seed: int = 42,
    split: str = DEFAULT_SPLIT,
    min_user_turns: int = MIN_USER_TURNS,
) -> List[UserData]:
    if split not in SUPPORTED_TEXT_SPLITS:
        raise ValueError(
            f"Unsupported PersonaMem-v2 split '{split}'. "
            f"Use one of {sorted(SUPPORTED_TEXT_SPLITS)}."
        )
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return _records_to_users(
        dataset,
        n_users=n_users,
        seed=seed,
        split=split,
        min_user_turns=min_user_turns,
    )


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _summarize_numeric(name: str, values: List[float]) -> None:
    if not values:
        print(f"{name}: no data")
        return
    print(f"{name}:")
    print(f"  count={len(values)}")
    print(f"  mean={sum(values) / len(values):.4f} min={min(values):.4f} max={max(values):.4f}")
    print(
        "  "
        + " ".join(
            f"p{p}={_percentile(values, p):.4f}"
            for p in [5, 25, 50, 75, 90, 95]
        )
    )


def print_dataset_stats(users: List[UserData]) -> None:
    turns_per_user = [sum(len(conv.turns) for conv in user.conversations) for user in users]
    candidate_counts = [
        len(turn.candidates)
        for user in users
        for conv in user.conversations
        for turn in conv.turns
    ]
    print("=== PersonaMem-v2 Dataset Stats ===")
    print(f"users={len(users)} turns={sum(turns_per_user)}")
    _summarize_numeric("turns_per_user", [float(value) for value in turns_per_user])
    print(f"candidate_count_distribution={dict(Counter(candidate_counts))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for PersonaMem-v2 adapter output")
    parser.add_argument("--n-users", type=int, default=None, help="Number of users to sample for stats/debug.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for user sampling and candidate shuffling.")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="PersonaMem-v2 text split to load.")
    parser.add_argument("--print-stats", action="store_true", help="Print dataset stats.")
    parser.add_argument("--print-preview", action="store_true", help="Print preview of the first loaded user.")
    args = parser.parse_args()

    loaded_users = load_personamem_v2(n_users=args.n_users, seed=args.seed, split=args.split)
    if args.print_stats:
        print_dataset_stats(loaded_users)
    if args.print_preview:
        print(loaded_users[:1])
