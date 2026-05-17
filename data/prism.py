from typing import Dict, List
from datasets import load_dataset
from .base import Turn, Conversation, UserData
import random
import math

MIN_USER_TURNS = 20

def group_by_turns(conversation_history: List[Dict], id_prefix: str) -> List[Turn]:
    turns = {}
    for msg in conversation_history:
        turn = msg.get("turn", 0)
        if turn not in turns:
            turns[turn] = {
                "user_message": "",
                "candidates": [],
                "chosen": "",
            }
        role = msg.get("role")
        if role == "user":
            turns[turn]["user_message"] = msg.get("content")
        else:
            content = msg.get("content")
            turns[turn]["candidates"].append(content)
            if msg.get("if_chosen", False):
                turns[turn]["chosen"] = content
    return [
        Turn(
            turn_id=id_prefix + str(t),
            user_message=data["user_message"],
            candidates=data["candidates"],
            chosen=data["chosen"],
        )
        for t, data in sorted(turns.items()) if data["user_message"] != "EMPTY STRING"  # Filter out turns with PRISM empty placeholder (2 in total)
    ]

def extract_profile(survey: dict) -> str:
    key_fields = [
        'age',
        'gender',
        'religion',
        'ethnicity',
        'self_description',
        'system_string',
    ]
    profile = "\n".join([f"{field.replace('_', ' ').title()}: {survey[field]}" for field in key_fields if field in survey and survey[field]])
    stated_prefs: dict = survey["stated_prefs"]
    stated_prefs.pop("other")
    stated_prefs.pop("other_text")
    mean_score = sum(stated_prefs.values()) / len(stated_prefs)
    prior_prefs = [k for k, v in stated_prefs.items() if v > mean_score]
    low_prefs = [k for k, v in stated_prefs.items() if v <= mean_score]
    profile += "\nPrioritized aspects: " + ", ".join(prior_prefs)
    profile += "\nComparatively less prioritzed aspects: " + ", ".join(low_prefs)
    return profile

def load_prism(n_users: int = None, seed: int = 42) -> List[UserData]:
    """
    Load PRISM conversations and return per-user bundles with normalized conversations and turns.
    If n_users is provided, limit to the first n unique users (by dataset order).
    """
    train_data = load_dataset("HannahRoseKirk/prism-alignment", "conversations")['train']

    user_conversations: Dict[List[Dict]] = {}
    user_order: List[str] = []
    for rec in train_data:
        uid = rec['user_id']
        cid = rec.get('conversation_id', 'unknown')
        conv_hist = rec.get('conversation_history', None)
        if uid not in user_conversations:
            user_conversations[uid] = []
            user_order.append(uid)
        user_conversations[uid].append({
            'conversation_id': cid,
            'conversation_history': conv_hist
        })

    survey_data = load_dataset("HannahRoseKirk/prism-alignment", "survey")['train']
    survey_rec = {rec['user_id']: rec for rec in survey_data}
    
    users: List[UserData] = []
    for uid in user_order:
        convs = []
        for conv in user_conversations.get(uid, []):
            cid = conv['conversation_id']
            history = conv['conversation_history']
            turns = group_by_turns(history, id_prefix=uid + "_" + cid + "_")
            convs.append(Conversation(
                conversation_id=cid,
                turns=turns
            ))
        total_turns = sum(len(conv.turns) for conv in convs)
        if total_turns < MIN_USER_TURNS:
            continue
        gt_profile = extract_profile(survey_rec[uid])
        users.append(UserData(
            user_id=uid,
            conversations=convs,
            gt_profile=gt_profile
        ))
    if n_users is not None:
        random.seed(seed)
        users = random.sample(users, min(n_users, len(users)))
    return users


def percentile(values: List[float], p: float) -> float:
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


def summarize_numeric(name: str, values: List[float], percentiles: List[int] = [5, 25, 50, 75, 90, 95]) -> None:
    if not values:
        print(f"{name}: no data")
        return
    mean_v = sum(values) / len(values)
    print(f"{name}:")
    print(f"  count={len(values)}")
    print(f"  mean={mean_v:.4f} min={min(values):.4f} max={max(values):.4f}")
    print("  " + " ".join([f"p{p}={percentile(values, p):.4f}" for p in percentiles]))


def print_dataset_stats(users: List[UserData]) -> None:
    conv_turn_counts: List[int] = []
    user_turn_counts: List[int] = []
    candidate_word_counts: List[int] = []
    n_conversations = 0
    n_turns = 0
    n_candidates = 0

    for user in users:
        total_turns = 0
        for conv in user.conversations:
            n_conversations += 1
            turn_count = len(conv.turns)
            conv_turn_counts.append(turn_count)
            total_turns += turn_count
            for turn in conv.turns:
                n_turns += 1
                for cand in turn.candidates:
                    n_candidates += 1
                    candidate_word_counts.append(len(cand.split()))
                    if len(cand.split()) == 1:
                        print(cand)
        user_turn_counts.append(total_turns)

    print("=== PRISM Dataset Stats ===")
    print(f"users={len(users)} conversations={n_conversations} turns={n_turns} candidates={n_candidates}")
    summarize_numeric("turns_per_conversation", [float(x) for x in conv_turn_counts])
    summarize_numeric("turns_per_user", [float(x) for x in user_turn_counts])
    summarize_numeric("candidate_word_count_per_turn_candidate", [float(x) for x in candidate_word_counts])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for PRISM adapter output")
    parser.add_argument("--n-users", type=int, default=None, help="Number of users to sample for stats/debug (default: all users).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when n-users is set.")
    parser.add_argument("--print-stats", action="store_true", help="Print dataset stats.")
    parser.add_argument("--print-preview", action="store_true", help="Print preview of loaded data for the first user.")
    parser.add_argument("--user-id", type=str, default=None, help="User ID to load for preview")
    args = parser.parse_args()

    users = load_prism(n_users=args.n_users, seed=args.seed)
    if args.print_stats:
        print_dataset_stats(users)

    if args.print_preview:
        if args.user_id is not None:
            selected = [user for user in users if user.user_id == args.user_id]
            print(selected)
        else:
            print(users[:1])
