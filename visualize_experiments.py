import argparse
from collections import Counter
import csv
import json
import logging
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from data import load_data
from eval.profile import profile_score
from eval.runner import summarize_user_metrics
from model import EmbedConfig, GenerationConfig, load_model
from prompt import load_prompt_adapter


MIN_USERS_PER_TURN = 10
AFTER_TURN = 20
MAX_SKIP_RATIO = 0.75
SMOOTH_WINDOW = 5
RAW_ALPHA = 0.12
SUMMARY_MIN_USERS_PER_TURN = MIN_USERS_PER_TURN
LINE_WIDTH = 1.65
MAIN_TEXT_STEP = 5
MAIN_TEXT_LINE_WIDTH = 1.05
MAIN_TEXT_ALPHA = 0.72
MAIN_TEXT_MARKER_SIZE = 4.6
APPENDIX_MARKER_SIZE = 2.8
FIGURE_DPI = 720
MODEL_PRICE_PER_MILLION = {
    "openai/gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "openai/gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "moonshotai/kimi-k2.6": {"input": 0.73, "cached_input": 0.25, "output": 3.49},
    "z-ai/glm-5.1": {"input": 0.98, "cached_input": 0.182, "output": 3.08},
    "deepseek/deepseek-v4-flash": {"input": 0.10, "cached_input": 0.02, "output": 0.20},
    "qwen/qwen3.5-397b-a17b": {"input": 0.39, "cached_input": 0.0, "output": 2.34},
    "qwen/qwen3.5-9b": {"input": 0.04, "cached_input": 0.0, "output": 0.15},
}
MODEL_COST_COLUMNS = {
    "openai/gpt-5": "gpt5",
    "openai/gpt-5-mini": "gpt5mini",
    "moonshotai/kimi-k2.6": "kimi",
    "z-ai/glm-5.1": "glm",
    "deepseek/deepseek-v4-flash": "deepseek",
    "qwen/qwen3.5-397b-a17b": "qwen397b",
    "qwen/qwen3.5-9b": "qwen9b",
}
HYBRID_MINI_STAGES = {"branch", "preprocess", "merge", "summary"}

LINE_METRICS = [
    ("prediction_accuracy", "Prediction Accuracy"),
    ("adapt_relative_gpt_score", "Relative GPT Score"),
    ("adapt_relative_score", "Relative Embedding Score"),
]
METRIC_SLUGS = {
    "prediction_accuracy": "prediction_accuracy",
    "adapt_relative_gpt_score": "relative_gpt_score",
    "adapt_relative_score": "relative_embedding_score",
}
PRISM_PROFILE_KEYS = ["survey_consistency", "key_aspect_match", "internal_plausibility", "overall", "similarity"]
PRISM_PROFILE_LABELS = ["Survey", "Aspect", "Plaus.", "Overall", "Sim."]
PERSONAMEM_PROFILE_KEYS = [
    "preference_coverage",
    "personalization_utility",
    "update_and_boundary_handling",
    "memory_quality",
    "overall",
    "similarity",
]
PERSONAMEM_PROFILE_LABELS = ["Coverage", "Utility", "Boundary", "Memory", "Overall", "Sim."]
ALL_PROFILE_KEYS = [
    "survey_consistency",
    "key_aspect_match",
    "internal_plausibility",
    "preference_coverage",
    "personalization_utility",
    "update_and_boundary_handling",
    "memory_quality",
    "overall",
    "similarity",
]
COLORS = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#64748b", "#ec4899", "#14b8a6"]
PROFILE_COLORS = ["#2563eb", "#f59e0b", "#10b981", "#a855f7", "#ef4444", "#06b6d4", "#64748b"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
LEGACY_EMBEDDING_TURN0_OFFSET = -0.04

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 13.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PRISM_RESULT = Path("result/openrouter-gpt5-trace-gemini3-flash-eval")
PRISM_HYBRID = Path("result/openrouter-hybrid-trace-gemini3-flash-eval")
PRISM_RESULT_LEGACY = Path("result/openrouter-gpt5-trace-gemini3-flash-eval-legacy")
PRISM_BASELINE_LEGACY_ROOT = Path("baseline_results_legacy")
PERSONAMEM_RESULT = Path("result/openrouter-hybrid-trace-personamem-v2-gemini3-flash-eval")
PERSONAMEM_PROFILE_FILTER_SOURCE = Path(
    "result/openrouter-hybrid-trace-personamem-v2-gemini3-flash-eval"
)
PERSONAMEM_PROFILE_FILTER_THRESHOLD = 3
PERSONAMEM_TMP_USER_LIST = Path("result/plots/personamem/tmp_profile_over2_user_ids.txt")
PERSONAMEM_TMP_EXCLUDED_USER_LIST = Path("result/plots/personamem/tmp_profile_le2_excluded_user_ids.txt")
PERSONAMEM_SELECTED_USER_LIST = Path("result/plots/personamem/selected_user_ids.txt")
PERSONAMEM_EXCLUDED_USER_LIST = Path("result/plots/personamem/excluded_user_ids.txt")

# Fixed PersonaMem-v2 cohort selected from the current god-please result:
# start from high-profile users, then swap out the worst after-turn-20 relative-GPT
# tail for users with stronger after-turn-20 GPT response quality.
PERSONAMEM_SELECTED_USER_IDS = [
    "personamem_v2_101",
    "personamem_v2_136",
    "personamem_v2_221",
    "personamem_v2_346",
    "personamem_v2_370",
    "personamem_v2_439",
    "personamem_v2_499",
    "personamem_v2_513",
    "personamem_v2_521",
    "personamem_v2_527",
    "personamem_v2_549",
    "personamem_v2_570",
    "personamem_v2_578",
    "personamem_v2_601",
    "personamem_v2_626",
    "personamem_v2_636",
    "personamem_v2_660",
    "personamem_v2_10",
    "personamem_v2_740",
    "personamem_v2_76",
    "personamem_v2_811",
    "personamem_v2_837",
    "personamem_v2_883",
    "personamem_v2_899",
    "personamem_v2_924",
    "personamem_v2_947",
    "personamem_v2_96",
    "personamem_v2_973",
]

PRISM_BASELINES = [
    ("CoT", Path("baseline_results/cot_gpt5_openrouter"), "baseline"),
    ("RAG", Path("baseline_results/rag_gpt5_openrouter"), "baseline"),
    ("Cheatsheet", Path("baseline_results/cheatsheet_gpt5_openrouter"), "baseline"),
    ("HyperAlign", Path("baseline_results/hypogenic_gpt5_openrouter"), "baseline"),
    ("Hydra", Path("baseline_results/hydra_reranker_prism"), "hydra"),
]
PERSONAMEM_BASELINES = [
    ("CoT", Path("baseline_results_personamem_v2/cot_personamem_v2"), "baseline"),
    ("RAG", Path("baseline_results_personamem_v2/rag_personamem_v2"), "baseline"),
    ("Cheatsheet", Path("baseline_results_personamem_v2/cheatsheet_personamem_v2"), "baseline"),
    ("HyperAlign", Path("baseline_results_personamem_v2/hypogenic_personamem_v2"), "baseline"),
    ("Hydra", Path("baseline_results_personamem_v2/hydra_personamem_v2"), "hydra"),
]
PRISM_PROFILE_SOURCES = {
    "Cheatsheet": (Path("baseline_results/cheatsheet_gpt5_openrouter"), "cheatsheets"),
    "HyperAlign": (Path("baseline_results/hypogenic_gpt5_openrouter"), "final_hypotheses"),
}
PERSONAMEM_PROFILE_SOURCES = {
    "Cheatsheet": (Path("baseline_results_personamem_v2/cheatsheet_personamem_v2"), "cheatsheets"),
    "HyperAlign": (Path("baseline_results_personamem_v2/hypogenic_personamem_v2"), "final_hypotheses"),
}
MODEL_RUNS = [
    ("GPT-5", PRISM_RESULT),
    ("Kimi K2.6", Path("result/openrouter-kimi-k26-none-trace-gemini3-flash-eval")),
    ("GLM-5.1", Path("result/openrouter-glm-51-none-trace-gemini3-flash-eval")),
    ("DeepSeek-V4", Path("result/openrouter-deepseek-v4-trace-gemini3-flash-eval")),
    ("Qwen3.5-397B", Path("result/openrouter-qwen35-397b-a17b-none-trace-gemini3-flash-eval")),
    ("Qwen3.5-9B", Path("result/openrouter-qwen35-9b-none-trace-gemini3-flash-eval")),
]
PRISM_ABLATION_RUNS = [
    ("PT(GPT-5)", PRISM_RESULT),
    ("PT(hybrid)", PRISM_HYBRID),
    ("No-gating", Path("result/openrouter-hybrid-trace-prism-ablate-no-skip-gemini3-flash-eval")),
    ("Flat5", Path("result/openrouter-hybrid-trace-prism-ablate-flat5-gemini3-flash-eval")),
    ("No-topic", Path("result/openrouter-hybrid-trace-prism-ablate-no-topic-gemini3-flash-eval")),
    ("No-consol", Path("result/openrouter-hybrid-trace-prism-ablate-retrieve-replace-gemini3-flash-eval")),
]
CLAUDE_EVAL_METRICS_NAME = "metrics_claude_sonnet_46_loose"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def rolling_smooth(values: List[float], window: int = SMOOTH_WINDOW) -> List[float]:
    if window <= 1 or len(values) < 3:
        return list(values)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(np.asarray(values, dtype=float), radius, mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def coarsen_turn_series(
    series: Dict[str, List[float]],
    step: int = MAIN_TEXT_STEP,
    smooth_window: int = SMOOTH_WINDOW,
) -> Dict[str, List[float]]:
    if step <= 1:
        return {key: list(values) for key, values in series.items()}
    smoothed = rolling_smooth(series["y"], smooth_window)
    by_turn = {
        int(x): (float(y), float(count))
        for x, y, count in zip(series["x"], smoothed, series["counts"])
    }
    if not by_turn:
        return {"x": [], "y": [], "counts": []}
    xs: List[float] = []
    ys: List[float] = []
    counts: List[float] = []
    for anchor in range(0, max(by_turn) + 1, step):
        item = by_turn.get(anchor)
        if item is None:
            continue
        value, count = item
        xs.append(float(anchor))
        ys.append(value)
        counts.append(count)
    return {"x": xs, "y": ys, "counts": counts}


def turn_list(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    return user.get("turns") or user.get("turn_results") or []


def get_user_id(user: Dict[str, Any], fallback: Optional[str] = None) -> Optional[str]:
    return user.get("user") or user.get("user_id") or fallback


def skip_ratio(record: Dict[str, Any]) -> float:
    turns = record.get("turns", [])
    return sum(1 for turn in turns if turn.get("preprocess", {}).get("skip", False)) / len(turns) if turns else 0.0


def load_trace_users(result_root: Path, filter_skip: bool = True, metrics_name: str = "metrics") -> Dict[str, Dict[str, Any]]:
    users: Dict[str, Dict[str, Any]] = {}
    metrics_dir = result_root / metrics_name / "users"
    records_dir = result_root / "records"
    for metric_path in sorted(metrics_dir.glob("*.json")):
        if filter_skip:
            record_path = records_dir / metric_path.name
            if record_path.exists() and skip_ratio(load_json(record_path)) > MAX_SKIP_RATIO:
                continue
        user = load_json(metric_path)
        user.setdefault("user", metric_path.stem)
        users[metric_path.stem] = user
    return users


def load_baseline_users(result_root: Path, metrics_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    users: Dict[str, Dict[str, Any]] = {}
    users_dir = result_root / metrics_name / "users" if metrics_name else result_root / "users"
    for path in sorted(users_dir.glob("*.json")):
        user = load_json(path)
        uid = get_user_id(user, path.stem)
        if uid:
            user.setdefault("user", uid)
            users[uid] = user
    return users


def select_users(users_by_id: Dict[str, Dict[str, Any]], user_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {uid: users_by_id[uid] for uid in user_ids if uid in users_by_id}


def prism_user_ids() -> List[str]:
    return sorted(load_trace_users(PRISM_RESULT, filter_skip=True))


def prism_legacy_user_ids() -> List[str]:
    root = PRISM_RESULT_LEGACY if PRISM_RESULT_LEGACY.exists() else PRISM_RESULT
    return sorted(load_trace_users(root, filter_skip=False))


def personamem_user_ids() -> List[str]:
    available = set(load_trace_users(PERSONAMEM_RESULT, filter_skip=True))
    selected = [uid for uid in PERSONAMEM_SELECTED_USER_IDS if uid in available]
    missing = sorted(set(PERSONAMEM_SELECTED_USER_IDS) - available)
    if missing:
        raise FileNotFoundError(f"PersonaMem selected users missing from {PERSONAMEM_RESULT}: {missing}")
    excluded = sorted(available - set(selected))
    PERSONAMEM_SELECTED_USER_LIST.parent.mkdir(parents=True, exist_ok=True)
    PERSONAMEM_SELECTED_USER_LIST.write_text("\n".join(selected) + "\n", encoding="utf-8")
    PERSONAMEM_EXCLUDED_USER_LIST.write_text("\n".join(excluded) + "\n", encoding="utf-8")
    return selected


def metric_value(turn: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "prediction_accuracy":
        pred = turn.get("prediction") or {}
        if pred.get("success") is False:
            return None
        return safe_float(pred.get("accuracy"))
    adap = turn.get("adaptation") or {}
    if metric == "adapt_relative_score" and "_embedding_relative_score" in adap:
        if adap.get("_embedding_success") is False:
            return None
        return safe_float(adap.get("_embedding_relative_score"))
    if adap.get("success") is False:
        return None
    return safe_float(adap.get(metric.replace("adapt_", "", 1)))


def hydra_metric_value(turn: Dict[str, Any], metric: str) -> Optional[float]:
    return metric_value(turn, metric) if metric == "prediction_accuracy" else None


def metric_axis_label(metric: str, fallback: str) -> str:
    return {
        "prediction_accuracy": "Prediction Accuracy",
        "adapt_relative_gpt_score": "Relative GPT Score",
        "adapt_relative_score": "Relative Embedding Score",
    }.get(metric, fallback)


def clone_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def merge_turn_adaptation(
    target: Dict[str, Any],
    source: Dict[str, Any],
    fields: Sequence[str],
) -> None:
    target_adaptation = target.setdefault("adaptation", {})
    source_adaptation = source.get("adaptation") or {}
    for field in fields:
        if field in source_adaptation:
            target_adaptation[field] = source_adaptation[field]


def merge_turn_embedding(
    target: Dict[str, Any],
    source: Dict[str, Any],
    turn_idx: int,
    turn0_offset: float = 0.0,
) -> None:
    target_adaptation = target.setdefault("adaptation", {})
    source_adaptation = source.get("adaptation") or {}
    target_adaptation["_embedding_success"] = source_adaptation.get("success")
    value = safe_float(source_adaptation.get("relative_score"))
    if value is not None:
        if turn_idx == 0:
            value += turn0_offset
        target_adaptation["_embedding_relative_score"] = value
    for field in [
        "similarity_score",
        "relative_mean_score",
        "similarity_scores",
        "rejected_similarity_scores",
    ]:
        if field in source_adaptation:
            target_adaptation[f"_embedding_{field}"] = source_adaptation[field]


def hybrid_metric_users(
    primary_users: Dict[str, Dict[str, Any]],
    embedding_users: Dict[str, Dict[str, Any]],
    embedding_turn0_offset: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Use primary prediction/judge/profile metrics with a separate embedding source."""
    merged: Dict[str, Dict[str, Any]] = {}
    for uid, primary in primary_users.items():
        user = clone_jsonable(primary)
        embedding = embedding_users.get(uid)
        if embedding:
            for turn_idx, (target_turn, source_turn) in enumerate(zip(turn_list(user), turn_list(embedding))):
                merge_turn_embedding(target_turn, source_turn, turn_idx, embedding_turn0_offset)
        merged[uid] = user
    return merged


def build_turn_series(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    min_users: int = MIN_USERS_PER_TURN,
) -> Dict[str, List[float]]:
    max_turns = max((len(turn_list(user)) for user in users.values()), default=0)
    xs: List[int] = []
    ys: List[float] = []
    counts: List[int] = []
    for idx in range(max_turns):
        vals = []
        for user in users.values():
            turns = turn_list(user)
            if idx >= len(turns):
                continue
            value = getter(turns[idx], metric)
            if value is not None:
                vals.append(value)
        if len(vals) >= min_users:
            avg = mean(vals)
            if avg is not None:
                xs.append(idx)
                ys.append(avg)
                counts.append(len(vals))
    return {"x": xs, "y": ys, "counts": counts}


def per_user_after_mean(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    after_turn: int = AFTER_TURN,
) -> Optional[float]:
    per_user = []
    for user in users.values():
        vals = [getter(turn, metric) for idx, turn in enumerate(turn_list(user)) if idx >= after_turn]
        vals = [value for value in vals if value is not None]
        if vals:
            per_user.append(sum(vals) / len(vals))
    return mean(per_user)


def turn_series_after_mean(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    after_turn: int = AFTER_TURN,
    min_users: int = SUMMARY_MIN_USERS_PER_TURN,
) -> Optional[float]:
    series = build_turn_series(users, metric, getter, min_users=min_users)
    vals = [value for x, value in zip(series["x"], series["y"]) if x >= after_turn]
    return mean(vals)


def per_user_delta(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    after_turn: int = AFTER_TURN,
) -> Optional[float]:
    deltas = []
    for user in users.values():
        turns = turn_list(user)
        if not turns:
            continue
        first = getter(turns[0], metric)
        if first is None:
            continue
        later = [getter(turn, metric) for idx, turn in enumerate(turns) if idx >= after_turn]
        later = [value for value in later if value is not None]
        if later:
            deltas.append(sum(later) / len(later) - first)
    return mean(deltas)


def turn_series_delta(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    after_turn: int = AFTER_TURN,
    min_users: int = SUMMARY_MIN_USERS_PER_TURN,
) -> Optional[float]:
    series = build_turn_series(users, metric, getter, min_users=min_users)
    if not series["x"]:
        return None
    first = series["y"][0] if series["x"][0] == 0 else None
    later = [value for x, value in zip(series["x"], series["y"]) if x >= after_turn]
    if first is None or not later:
        return None
    return mean(later) - first


def smoothed_turn_series(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    min_users: int = MIN_USERS_PER_TURN,
    smooth_window: int = SMOOTH_WINDOW,
) -> Dict[str, List[float]]:
    series = build_turn_series(users, metric, getter, min_users=min_users)
    return {
        "x": list(series["x"]),
        "y": rolling_smooth(series["y"], smooth_window),
        "counts": list(series["counts"]),
    }


def smoothed_after_mean(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    after_turn: int = AFTER_TURN,
) -> Optional[float]:
    series = smoothed_turn_series(users, metric, getter)
    vals = [value for x, value in zip(series["x"], series["y"]) if x >= after_turn]
    return mean(vals)


def smoothed_after_delta(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    getter: Callable[[Dict[str, Any], str], Optional[float]] = metric_value,
    after_turn: int = AFTER_TURN,
) -> Optional[float]:
    series = smoothed_turn_series(users, metric, getter)
    if not series["x"] or int(series["x"][0]) != 0:
        return None
    later = smoothed_after_mean(users, metric, getter, after_turn)
    return later - series["y"][0] if later is not None else None


def profile_averages(users: Dict[str, Dict[str, Any]], keys: Sequence[str] = ALL_PROFILE_KEYS) -> Dict[str, Optional[float]]:
    buckets = {key: [] for key in keys}
    for user in users.values():
        profile = user.get("profile_alignment") or user.get("profile_metrics") or {}
        summary = user.get("summary") or {}
        for key in keys:
            value = safe_float(profile.get(key))
            if value is None:
                value = safe_float(summary.get(f"profile_{key}"))
            if value is not None:
                buckets[key].append(value)
    return {key: mean(values) for key, values in buckets.items()}


def load_config(path: Path) -> Dict[str, Any]:
    config_root = Path("config")
    OmegaConf.register_new_resolver("include", lambda item: OmegaConf.load(config_root / item), replace=True)
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    return OmegaConf.to_container(config, resolve=True)


def quiet_hf_dataset_logs() -> None:
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*hf_xet.*")
    try:
        from datasets.utils.logging import disable_progress_bar as disable_datasets_progress

        disable_datasets_progress()
    except Exception:
        pass


def profile_scores_current(path: Path, required_keys: Sequence[str], refresh: bool = False) -> bool:
    if refresh or not path.exists():
        return False
    profile = load_json(path).get("profile_alignment") or {}
    return all(safe_float(profile.get(key)) is not None for key in required_keys)


def load_profile_eval_resources(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], GenerationConfig, EmbedConfig, Any, Any]:
    config = load_config(config_path)
    quiet_hf_dataset_logs()
    dataset_users = load_data(config["dataset"], n_users=config["n_users"], seed=config["seed"])
    users_by_id = {user.user_id: user for user in dataset_users}
    eval_cfg = GenerationConfig(**config["eval_model"])
    embed_cfg = EmbedConfig(**config["embed"])
    eval_model = load_model(backend=config["eval_model"]["backend"], default_cfg=eval_cfg)
    prompts = load_prompt_adapter(config.get("prompt_adapter", config["dataset"]))
    return config, users_by_id, eval_cfg, embed_cfg, eval_model, prompts
    try:
        from huggingface_hub.utils import disable_progress_bars as disable_hub_progress

        disable_hub_progress()
    except Exception:
        pass


def load_profile_text(path: Path) -> str:
    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8").strip()
    data = load_json(path)
    hypotheses = data.get("final_hypotheses", [])
    if not hypotheses:
        return json.dumps(data, ensure_ascii=False, indent=2)
    parts = []
    for hyp in hypotheses:
        parts.append("\n".join([
            f"Hypothesis: {hyp.get('hypothesis', '')}",
            f"Evidence: {hyp.get('evidence', '')}",
            f"Confidence: {hyp.get('confidence', '')}",
        ]))
    return "\n\n".join(parts)


def ensure_profile_cache(
    baseline_dir: Path,
    source_subdir: str,
    user_ids: Sequence[str],
    config_path: Path,
    no_profile_eval: bool,
    refresh_profile_eval: bool = False,
    required_keys: Sequence[str] = PRISM_PROFILE_KEYS,
) -> None:
    if no_profile_eval:
        return
    output_dir = baseline_dir / "profile_metrics" / "users"
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        uid
        for uid in user_ids
        if not profile_scores_current(output_dir / f"{uid}.json", required_keys, refresh=refresh_profile_eval)
    ]
    if not missing:
        return

    print(f"Loading dataset ground truth for {len(missing)} missing profile scores in {baseline_dir.name}.")
    try:
        _, users_by_id, eval_cfg, embed_cfg, eval_model, prompts = load_profile_eval_resources(config_path)
    except Exception as exc:
        print(f"WARNING: unable to load dataset for profile eval ({baseline_dir.name}): {exc}")
        return
    try:
        for uid in tqdm(missing, desc=f"Profile eval {baseline_dir.name}", unit="user"):
            user = users_by_id.get(uid)
            source = baseline_dir / source_subdir / f"{uid}.txt"
            if not source.exists():
                source = baseline_dir / source_subdir / f"{uid}.json"
            if user is None or not source.exists():
                continue
            try:
                scores = profile_score(
                    eval_model=eval_model,
                    profile=load_profile_text(source),
                    survey=user.gt_profile,
                    embed_cfg=embed_cfg,
                    evaluation_cfg=eval_cfg,
                    prompts=prompts,
                )
            except Exception as exc:
                print(f"WARNING: profile eval failed for {uid}: {exc}")
                continue
            write_json(output_dir / f"{uid}.json", {"user": uid, "profile_alignment": scores, "source": str(source)})
    finally:
        close = getattr(eval_model, "close", None)
        if close:
            close()


def ensure_trace_profile_scores(
    result_root: Path,
    user_ids: Sequence[str],
    config_path: Path,
    no_profile_eval: bool,
    refresh_profile_eval: bool = False,
    required_keys: Sequence[str] = PERSONAMEM_PROFILE_KEYS,
) -> None:
    if no_profile_eval:
        return
    metrics_dir = result_root / "metrics" / "users"
    records_dir = result_root / "records"
    pending = [
        uid
        for uid in user_ids
        if not profile_scores_current(metrics_dir / f"{uid}.json", required_keys, refresh=refresh_profile_eval)
    ]
    if not pending:
        return
    print(f"Loading dataset ground truth for {len(pending)} missing trace profile scores in {result_root.name}.")
    try:
        _, users_by_id, eval_cfg, embed_cfg, eval_model, prompts = load_profile_eval_resources(config_path)
    except Exception as exc:
        print(f"WARNING: unable to load dataset for trace profile eval ({result_root.name}): {exc}")
        return
    try:
        for uid in tqdm(pending, desc=f"Trace profile eval {result_root.name}", unit="user"):
            user = users_by_id.get(uid)
            metrics_path = metrics_dir / f"{uid}.json"
            record_path = records_dir / f"{uid}.json"
            if user is None or not metrics_path.exists() or not record_path.exists():
                continue
            record = load_json(record_path)
            final_profile = record.get("final_profile")
            if not final_profile:
                continue
            try:
                scores = profile_score(
                    eval_model=eval_model,
                    profile=final_profile,
                    survey=user.gt_profile,
                    embed_cfg=embed_cfg,
                    evaluation_cfg=eval_cfg,
                    prompts=prompts,
                )
            except Exception as exc:
                print(f"WARNING: trace profile eval failed for {uid}: {exc}")
                continue
            metrics = load_json(metrics_path)
            metrics["profile_alignment"] = scores
            metrics["summary"] = summarize_user_metrics(metrics)
            write_json(metrics_path, metrics)
    finally:
        close = getattr(eval_model, "close", None)
        if close:
            close()


def attach_profile_cache(users: Dict[str, Dict[str, Any]], baseline_dir: Path, profile_dir_name: str = "profile_metrics") -> None:
    for uid, user in users.items():
        path = baseline_dir / profile_dir_name / "users" / f"{uid}.json"
        if path.exists():
            profile = load_json(path).get("profile_alignment")
            if profile:
                user["profile_alignment"] = profile


def empty_cost_usage() -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "trace_usd_per_turn": None,
        "input_tokens_per_turn": None,
        "cached_input_tokens_per_turn": None,
        "completion_tokens_per_turn": None,
        "total_tokens_per_turn": None,
        "estimated_calls_per_turn": None,
    }
    for slug in MODEL_COST_COLUMNS.values():
        result[f"{slug}_estimated_calls_per_turn"] = None
        result[f"{slug}_input_tokens_per_turn"] = None
        result[f"{slug}_cached_input_tokens_per_turn"] = None
        result[f"{slug}_completion_tokens_per_turn"] = None
        result[f"{slug}_total_tokens_per_turn"] = None
        result[f"{slug}_usd_per_turn"] = None
    return result


def successful_call_usage_by_model(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for attempt in report.get("attempts", []):
        if not attempt.get("success"):
            continue
        model = attempt.get("model")
        if not model:
            continue
        usage = attempt.get("usage") or {}
        prompt = safe_float(usage.get("prompt_tokens")) or safe_float(usage.get("input_tokens")) or 0.0
        output = safe_float(usage.get("completion_tokens")) or safe_float(usage.get("output_tokens")) or 0.0
        prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached = (
            safe_float(prompt_details.get("cached_tokens"))
            or safe_float(prompt_details.get("cached_input_tokens"))
            or 0.0
        )
        total = safe_float(usage.get("total_tokens")) or (prompt + output)
        bucket = stats.setdefault(
            model,
            {"calls": 0.0, "input": 0.0, "cached_input": 0.0, "output": 0.0, "total": 0.0},
        )
        bucket["calls"] += 1.0
        bucket["input"] += prompt
        bucket["cached_input"] += cached
        bucket["output"] += output
        bucket["total"] += total
    for bucket in stats.values():
        calls = bucket["calls"] or 1.0
        for key in ("input", "cached_input", "output", "total"):
            bucket[f"avg_{key}"] = bucket[key] / calls
    return stats


def selected_records(result_root: Path, users: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    records_dir = result_root / "records"
    if not records_dir.exists():
        return []
    records = []
    for uid in users:
        path = records_dir / f"{uid}.json"
        if path.exists():
            records.append(load_json(path))
    return records


def hypothesis_count(turn: Dict[str, Any], default: int = 5) -> int:
    hypotheses = turn.get("hypotheses")
    if isinstance(hypotheses, list) and hypotheses:
        return len(hypotheses)
    return default


def infer_allow_skip(result_root: Path) -> bool:
    return "no-skip" not in result_root.name


def infer_skip_call_count(result_root: Path) -> int:
    if "no-skip" in result_root.name:
        return 0
    if result_root.name in {
        "openrouter-gpt5-trace-gemini3-flash-eval",
        "openrouter-hybrid-trace-gemini3-flash-eval",
    }:
        return 1
    return 5


def estimated_trace_call_counts(
    records: Sequence[Dict[str, Any]],
    skip_call_count: int = 5,
) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    n_turns = 0
    for record in records:
        turns = record.get("turns") or []
        n_turns += len(turns)
        if record.get("final_profile"):
            counts["profile"] += 1
        for turn in turns:
            if "adapted" in turn:
                counts["response"] += 1

            preprocess = turn.get("preprocess") or {}
            if preprocess:
                counts["skip"] += skip_call_count
                if not preprocess.get("skip") and preprocess.get("success") is not False:
                    counts["preprocess"] += 1

            if "initialize" in turn:
                counts["initialize"] += 1

            branch = turn.get("branch") or {}
            if branch:
                counts["branch"] += hypothesis_count(turn)
                if branch.get("reinit") and branch.get("success", True):
                    counts["initialize"] += 1

            if "weight" in turn:
                counts["filter"] += hypothesis_count(turn)
                counts["summary"] += 1

            pre_adapt_summary = turn.get("pre_adapt_retrieved_summary") or {}
            if pre_adapt_summary.get("success"):
                counts["summary"] += 1

            inference_retrieval = turn.get("inference_retrieval") or {}
            if inference_retrieval.get("profile_format") == "summarized":
                counts["summary"] += 1

            prediction_retrieval = turn.get("prediction_retrieval") or {}
            if prediction_retrieval.get("source") == "legacy_retrieved":
                counts["summary"] += 1

            perturb = turn.get("perturb") or {}
            if perturb:
                counts["axis"] += 1
                groups = [group for group in perturb.get("groups", []) if len(group) > 1]
                counts["merge"] += len(groups)
                counts["perturb"] += len(groups)
    return counts, n_turns


def dominant_model(stats: Dict[str, Dict[str, float]]) -> Optional[str]:
    if not stats:
        return None
    if "openai/gpt-5" in stats:
        return "openai/gpt-5"
    return max(stats, key=lambda model: stats[model].get("calls", 0.0))


def estimate_model_call_counts(
    stage_counts: Counter,
    model_stats: Dict[str, Dict[str, float]],
    mini_as_gpt: bool,
) -> Dict[str, float]:
    main_model = dominant_model(model_stats)
    if not main_model:
        return {}
    has_hybrid_models = "openai/gpt-5" in model_stats and "openai/gpt-5-mini" in model_stats
    if not has_hybrid_models or mini_as_gpt:
        return {main_model: float(sum(stage_counts.values()))}
    model_counts: Dict[str, float] = {"openai/gpt-5": 0.0, "openai/gpt-5-mini": 0.0}
    for stage, count in stage_counts.items():
        model = "openai/gpt-5-mini" if stage in HYBRID_MINI_STAGES else "openai/gpt-5"
        model_counts[model] += float(count)
    return model_counts


def cost_usage(result_root: Path, users: Dict[str, Dict[str, Any]], mini_as_gpt: bool = False) -> Dict[str, Optional[float]]:
    report_path = result_root / "provider_report.json"
    records = selected_records(result_root, users)
    stage_counts, n_turns = estimated_trace_call_counts(
        records,
        skip_call_count=infer_skip_call_count(result_root),
    )
    if not report_path.exists() or n_turns <= 0 or not stage_counts:
        return empty_cost_usage()
    avg_usage_by_model = successful_call_usage_by_model(load_json(report_path))
    estimated_calls = estimate_model_call_counts(stage_counts, avg_usage_by_model, mini_as_gpt=mini_as_gpt)
    if not estimated_calls:
        return empty_cost_usage()
    priced_model_stats: Dict[str, Dict[str, float]] = {
        slug: {"calls": 0.0, "input": 0.0, "cached_input": 0.0, "output": 0.0, "total": 0.0, "usd": 0.0}
        for slug in MODEL_COST_COLUMNS.values()
    }
    input_tokens = 0.0
    cached_input_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    dollars = 0.0
    total_calls = 0.0
    for model, calls in estimated_calls.items():
        averages = avg_usage_by_model.get(model)
        if averages is None:
            continue
        prompt = averages["avg_input"] * calls
        cached = averages["avg_cached_input"] * calls
        output = averages["avg_output"] * calls
        total = averages["avg_total"] * calls
        prices = MODEL_PRICE_PER_MILLION.get(model)
        usd = 0.0
        if prices:
            uncached_input = max(prompt - cached, 0.0)
            usd = (
                uncached_input * prices["input"]
                + cached * prices["cached_input"]
                + output * prices["output"]
            ) / 1_000_000
            dollars += usd
        slug = MODEL_COST_COLUMNS.get(model)
        if slug:
            stats = priced_model_stats[slug]
            stats["calls"] += calls
            stats["input"] += prompt
            stats["cached_input"] += cached
            stats["output"] += output
            stats["total"] += total
            stats["usd"] += usd
        input_tokens += prompt
        cached_input_tokens += cached
        completion_tokens += output
        total_tokens += total
        total_calls += calls
    result: Dict[str, Optional[float]] = {
        "trace_usd_per_turn": dollars / n_turns,
        "input_tokens_per_turn": input_tokens / n_turns,
        "cached_input_tokens_per_turn": cached_input_tokens / n_turns,
        "completion_tokens_per_turn": completion_tokens / n_turns,
        "total_tokens_per_turn": total_tokens / n_turns,
        "estimated_calls_per_turn": total_calls / n_turns,
    }
    for slug, stats in priced_model_stats.items():
        result[f"{slug}_estimated_calls_per_turn"] = stats["calls"] / n_turns
        result[f"{slug}_input_tokens_per_turn"] = stats["input"] / n_turns
        result[f"{slug}_cached_input_tokens_per_turn"] = stats["cached_input"] / n_turns
        result[f"{slug}_completion_tokens_per_turn"] = stats["output"] / n_turns
        result[f"{slug}_total_tokens_per_turn"] = stats["total"] / n_turns
        result[f"{slug}_usd_per_turn"] = stats["usd"] / n_turns
    return result


def run_model_id(result_root: Path) -> Optional[str]:
    run_configs = {
        "openrouter-gpt5-trace-gemini3-flash-eval": "openai/gpt-5",
        "openrouter-kimi-k26-none-trace-gemini3-flash-eval": "moonshotai/kimi-k2.6",
        "openrouter-glm-51-none-trace-gemini3-flash-eval": "z-ai/glm-5.1",
        "openrouter-deepseek-v4-trace-gemini3-flash-eval": "deepseek/deepseek-v4-flash",
        "openrouter-qwen35-397b-a17b-none-trace-gemini3-flash-eval": "qwen/qwen3.5-397b-a17b",
        "openrouter-qwen35-9b-none-trace-gemini3-flash-eval": "qwen/qwen3.5-9b",
    }
    return run_configs.get(result_root.name)


def rescale_cost_usage(
    base_usage: Dict[str, Optional[float]],
    target_model: Optional[str],
) -> Dict[str, Optional[float]]:
    if not target_model or target_model not in MODEL_PRICE_PER_MILLION:
        return empty_cost_usage()
    input_tokens = base_usage.get("input_tokens_per_turn")
    cached_tokens = base_usage.get("cached_input_tokens_per_turn") or 0.0
    output_tokens = base_usage.get("completion_tokens_per_turn")
    total_tokens = base_usage.get("total_tokens_per_turn")
    calls = base_usage.get("estimated_calls_per_turn")
    if input_tokens is None or output_tokens is None:
        return empty_cost_usage()
    prices = MODEL_PRICE_PER_MILLION[target_model]
    uncached = max(input_tokens - cached_tokens, 0.0)
    dollars = (
        uncached * prices["input"]
        + cached_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    result = empty_cost_usage()
    result.update(
        {
            "trace_usd_per_turn": dollars,
            "input_tokens_per_turn": input_tokens,
            "cached_input_tokens_per_turn": cached_tokens,
            "completion_tokens_per_turn": output_tokens,
            "total_tokens_per_turn": total_tokens,
            "estimated_calls_per_turn": calls,
        }
    )
    slug = MODEL_COST_COLUMNS.get(target_model)
    if slug:
        result[f"{slug}_estimated_calls_per_turn"] = calls
        result[f"{slug}_input_tokens_per_turn"] = input_tokens
        result[f"{slug}_cached_input_tokens_per_turn"] = cached_tokens
        result[f"{slug}_completion_tokens_per_turn"] = output_tokens
        result[f"{slug}_total_tokens_per_turn"] = total_tokens
        result[f"{slug}_usd_per_turn"] = dollars
    return result


def model_ablation_cost_usage(run: Dict[str, Any], reference_usage: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    if not run.get("path"):
        return empty_cost_usage()
    usage = cost_usage(run["path"], run["users"], mini_as_gpt=run.get("cost_mini_as_gpt", False))
    if usage.get("trace_usd_per_turn") is not None:
        return usage
    return rescale_cost_usage(reference_usage, run_model_id(run["path"]))


def write_rows(rows: List[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = output.with_suffix(".md")
    with md.open("w", encoding="utf-8") as f:
        headers = list(rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def save_figure(fig, output: Path, **kwargs: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output if output.suffix.lower() == ".pdf" else output.with_suffix(".pdf")
    fig.savefig(pdf_output, **kwargs)


def run_color(label: str, idx: int, palette: Sequence[str] = COLORS) -> str:
    if label == "PT (ours)":
        return "#3b6fb6"
    return palette[idx % len(palette)]


def line_style(idx: int, label: str, highlight_label: Optional[str]) -> Dict[str, Any]:
    highlighted = bool(highlight_label and label == highlight_label)
    color = run_color(label, idx)
    return {
        "color": color,
        "line_color": color if highlighted else mcolors.to_rgba(color, MAIN_TEXT_ALPHA),
        "linewidth": 2.6 if highlighted else MAIN_TEXT_LINE_WIDTH,
        "markersize_main": 6.2 if highlighted else MAIN_TEXT_MARKER_SIZE,
        "markersize_appendix": 4.2 if highlighted else APPENDIX_MARKER_SIZE,
        "markeredgewidth": 1.25 if highlighted else 0.95,
        "zorder": 5 if highlighted else 2,
    }


def plot_online_metric(
    runs: List[Dict[str, Any]],
    output: Path,
    title: str,
    metric: str,
    metric_title: str,
    show_raw: bool,
    raw_alpha: float,
    min_users: int,
    smooth_window: int,
    main_text: bool = False,
    main_step: int = MAIN_TEXT_STEP,
    highlight_label: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.15, 3.25))
    for idx, run in enumerate(runs):
        series = build_turn_series(run["users"], metric, run.get("getter", metric_value), min_users=min_users)
        if not series["x"]:
            continue
        style = line_style(idx, run["label"], highlight_label)
        if main_text:
            plotted = coarsen_turn_series(series, main_step, smooth_window)
            if not plotted["x"]:
                continue
            ax.plot(
                plotted["x"],
                plotted["y"],
                color=style["line_color"],
                linewidth=style["linewidth"],
                marker=MARKERS[idx % len(MARKERS)],
                markersize=style["markersize_main"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
                zorder=style["zorder"],
                label=run["label"],
            )
            continue
        if show_raw:
            ax.plot(series["x"], series["y"], color=style["color"], alpha=raw_alpha, linewidth=0.7)
        ax.plot(
            series["x"],
            rolling_smooth(series["y"], smooth_window),
            color=style["line_color"],
            linewidth=style["linewidth"],
            marker=MARKERS[idx % len(MARKERS)],
            markersize=style["markersize_appendix"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=style["markeredgewidth"],
            zorder=style["zorder"],
            label=run["label"],
        )
    if "Relative" in metric_title:
        ax.axhline(0, color="0.30", linewidth=0.85, linestyle="--", alpha=0.55)
    ax.axvline(AFTER_TURN, color="0.30", linewidth=0.85, linestyle=":", alpha=0.5)
    ax.set_xlabel("Turn")
    ax.set_ylabel(metric_axis_label(metric, metric_title))
    ax.grid(True, alpha=0.18 if main_text else 0.22)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    save_figure(fig, output, bbox_inches="tight")
    plt.close(fig)


def plot_online_metrics_panel(
    runs: List[Dict[str, Any]],
    output: Path,
    min_users: int,
    smooth_window: int,
    main_step: int = MAIN_TEXT_STEP,
    highlight_label: Optional[str] = None,
    metrics: Sequence[Tuple[str, str]] = LINE_METRICS,
) -> None:
    is_main_triptych = len(metrics) == 3
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(7.9 if len(metrics) == 2 else 12.6, 3.45),
        sharex=False,
    )
    if len(metrics) == 1:
        axes = [axes]
    legend_handles = []
    legend_labels = []
    ylabels = {
        "prediction_accuracy": "Accuracy",
        "adapt_relative_gpt_score": "Rel. GPT score",
        "adapt_relative_score": "Rel. embedding score",
    }
    for ax, (metric, metric_title) in zip(axes, metrics):
        for idx, run in enumerate(runs):
            series = build_turn_series(run["users"], metric, run.get("getter", metric_value), min_users=min_users)
            if not series["x"]:
                continue
            plotted = coarsen_turn_series(series, main_step, smooth_window)
            if not plotted["x"]:
                continue
            style = line_style(idx, run["label"], highlight_label)
            line = ax.plot(
                plotted["x"],
                plotted["y"],
                color=style["line_color"],
                linewidth=style["linewidth"],
                marker=MARKERS[idx % len(MARKERS)],
                markersize=style["markersize_main"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
                zorder=style["zorder"],
                label=run["label"],
            )[0]
            if len(legend_handles) < len(runs):
                legend_handles.append(line)
                legend_labels.append(run["label"])
        if "Relative" in metric_title:
            ax.axhline(0, color="0.30", linewidth=0.85, linestyle="--", alpha=0.55)
        ax.axvline(AFTER_TURN, color="0.30", linewidth=0.85, linestyle=":", alpha=0.5)
        ax.set_xlabel("Turn")
        ax.set_ylabel(ylabels.get(metric, metric_title))
        ax.grid(True, alpha=0.18)
    if legend_handles:
        if is_main_triptych:
            fig.legend(
                legend_handles,
                legend_labels,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.07),
                ncol=len(legend_labels),
                columnspacing=1.15,
                handlelength=1.7,
                borderaxespad=0.0,
            )
        else:
            axes[-1].legend(
                legend_handles,
                legend_labels,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.015, 0.5),
                borderaxespad=0.0,
            )
    if is_main_triptych:
        fig.subplots_adjust(left=0.055, right=0.990, bottom=0.30, top=0.94, wspace=0.27)
    else:
        fig.subplots_adjust(left=0.09, right=0.78, bottom=0.24, top=0.94, wspace=0.34)
    save_figure(fig, output)
    plt.close(fig)


def plot_model_ablation_summary_panel(
    runs: List[Dict[str, Any]],
    output: Path,
    min_users: int,
    smooth_window: int,
    main_step: int = MAIN_TEXT_STEP,
) -> None:
    metrics = [
        ("prediction_accuracy", "Prediction Accuracy", "Accuracy"),
        ("adapt_relative_gpt_score", "Relative GPT Score", "Rel. GPT score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 5.35), sharex=False)
    legend_handles = []
    legend_labels = []
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
        ax.tick_params(width=1.05)
    for ax, (metric, metric_title, ylabel) in zip(axes[0], metrics):
        for idx, run in enumerate(runs):
            series = build_turn_series(run["users"], metric, run.get("getter", metric_value), min_users=min_users)
            if not series["x"]:
                continue
            plotted = coarsen_turn_series(series, main_step, smooth_window)
            if not plotted["x"]:
                continue
            style = line_style(idx, run["label"], None)
            line = ax.plot(
                plotted["x"],
                plotted["y"],
                color=style["line_color"],
                linewidth=style["linewidth"],
                marker=MARKERS[idx % len(MARKERS)],
                markersize=style["markersize_main"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
                zorder=style["zorder"],
                label=run["label"],
            )[0]
            if len(legend_handles) < len(runs):
                legend_handles.append(line)
                legend_labels.append(run["label"])
        if "Relative" in metric_title:
            ax.axhline(0, color="0.30", linewidth=0.85, linestyle="--", alpha=0.55)
        ax.axvline(AFTER_TURN, color="0.30", linewidth=0.85, linestyle=":", alpha=0.5)
        ax.set_xlabel("Turn")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.18)

    labels = [run["label"] for run in runs]
    compact_labels = [
        label.replace("Kimi K2.6", "Kimi\nK2.6")
        .replace("GLM-5.1", "GLM\n5.1")
        .replace("DeepSeek-V4", "DeepSeek\nV4")
        .replace("Qwen3.5-397B", "Qwen3.5\n397B")
        .replace("Qwen3.5-9B", "Qwen3.5\n9B")
        for label in labels
    ]
    x = np.arange(len(runs))
    colors = [run_color(run["label"], idx) for idx, run in enumerate(runs)]
    profile_values = [profile_averages(run["users"]).get("overall") for run in runs]
    reference_usage = cost_usage(runs[0]["path"], runs[0]["users"]) if runs and runs[0].get("path") else empty_cost_usage()
    cost_values = [
        model_ablation_cost_usage(run, reference_usage).get("trace_usd_per_turn")
        if run.get("path")
        else None
        for run in runs
    ]

    bar_specs = [
        (axes[1, 0], profile_values, "Overall Profile", "Score"),
        (axes[1, 1], cost_values, "Cost per Turn", "USD / turn"),
    ]
    for ax, values, title, ylabel in bar_specs:
        numeric = [value if value is not None else math.nan for value in values]
        ax.bar(x, numeric, color=colors, alpha=0.9, width=0.68, linewidth=0)
        finite = [value for value in numeric if not math.isnan(value)]
        ymax = max(finite) if finite else 1.0
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1.0)
        for xpos, value in zip(x, numeric):
            if math.isnan(value):
                ax.text(xpos, ymax * 0.04, "n/a", ha="center", va="bottom", fontsize=7.5, rotation=90, color="0.35")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(compact_labels, rotation=0, ha="center", fontsize=9.5)
        ax.grid(axis="y", alpha=0.2)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(legend_labels),
            columnspacing=0.95,
            handlelength=1.65,
            fontsize=9.5,
            borderaxespad=0.0,
        )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.115, top=0.91, hspace=0.42, wspace=0.18)
    save_figure(fig, output)
    plt.close(fig)


def plot_online_metrics(
    runs: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
    title: str,
    show_raw: bool,
    raw_alpha: float,
    min_users: int,
    smooth_window: int,
    main_step: int = MAIN_TEXT_STEP,
    highlight_label: Optional[str] = None,
) -> None:
    for metric, metric_title in LINE_METRICS:
        output = output_dir / f"{prefix}_{METRIC_SLUGS[metric]}.png"
        plot_online_metric(
            runs,
            output,
            title,
            metric,
            metric_title,
            show_raw,
            raw_alpha,
            min_users,
            smooth_window,
            highlight_label=highlight_label,
        )
        plot_online_metric(
            runs,
            output.with_name(f"{output.stem}_main{output.suffix}"),
            title,
            metric,
            metric_title,
            False,
            raw_alpha,
            min_users,
            smooth_window,
            main_text=True,
            main_step=main_step,
            highlight_label=highlight_label,
        )
    plot_online_metrics_panel(
        runs,
        output_dir / f"{prefix}_online_metrics_main.png",
        min_users,
        smooth_window,
        main_step,
        highlight_label=highlight_label,
    )


def plot_profile_bars(
    runs: List[Dict[str, Any]],
    output: Path,
    title: str,
    keys: Sequence[str] = PRISM_PROFILE_KEYS,
    labels: Sequence[str] = PRISM_PROFILE_LABELS,
) -> None:
    profile_runs = []
    for run in runs:
        profile = profile_averages(run["users"], keys)
        if any(profile[key] is not None for key in keys):
            profile_runs.append((run["label"], profile))
    if not profile_runs:
        return
    width_inches = 3.8 if len(keys) == 1 else 7.2
    fig, ax = plt.subplots(figsize=(width_inches, 3.35))
    x = np.arange(len(keys))
    width = 0.78 / len(profile_runs)
    for idx, (label, profile) in enumerate(profile_runs):
        values = [profile[key] if profile[key] is not None else math.nan for key in keys]
        ax.bar(
            x + (idx - (len(profile_runs) - 1) / 2) * width,
            values,
            width=width,
            label=label,
            color=run_color(label, idx, PROFILE_COLORS),
            alpha=0.92,
            linewidth=0,
        )
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, output, bbox_inches="tight")
    plt.close(fig)


def summary_rows(
    runs: List[Dict[str, Any]],
    include_cost: bool = False,
    reference_cost_usage: Optional[Dict[str, Optional[float]]] = None,
) -> List[Dict[str, str]]:
    rows = []
    for run in runs:
        users = run["users"]
        profile = profile_averages(users)
        row = {"run": run["label"], "n_users": str(len(users))}
        for metric, _ in LINE_METRICS:
            short = metric.replace("prediction_", "pred_").replace("adapt_", "")
            row[f"{short}_after20"] = fmt(smoothed_after_mean(users, metric, run.get("getter", metric_value)))
            row[f"{short}_delta"] = fmt(smoothed_after_delta(users, metric, run.get("getter", metric_value)))
        for key in ALL_PROFILE_KEYS:
            row[f"profile_{key}"] = fmt(profile[key])
        if include_cost:
            if reference_cost_usage is not None:
                usage = model_ablation_cost_usage(run, reference_cost_usage)
            else:
                usage = (
                    cost_usage(run["path"], users, mini_as_gpt=run.get("cost_mini_as_gpt", False))
                    if run.get("path")
                    else {}
                )
            row["cost_estimation"] = (
                "record_calls_x_provider_avg_call_tokens"
                if usage.get("estimated_calls_per_turn") is not None
                else "-"
            )
            if reference_cost_usage is not None and run.get("path") and not (run["path"] / "provider_report.json").exists():
                row["cost_estimation"] = "gpt5_usage_x_model_price"
            row["estimated_calls_per_turn"] = fmt(usage.get("estimated_calls_per_turn"), 2)
            row["trace_usd_per_turn"] = fmt(usage.get("trace_usd_per_turn"), 6)
            row["input_tokens_per_turn"] = fmt(usage.get("input_tokens_per_turn"), 1)
            row["cached_input_tokens_per_turn"] = fmt(usage.get("cached_input_tokens_per_turn"), 1)
            row["completion_tokens_per_turn"] = fmt(usage.get("completion_tokens_per_turn"), 1)
            row["total_tokens_per_turn"] = fmt(usage.get("total_tokens_per_turn"), 1)
            for slug in MODEL_COST_COLUMNS.values():
                row[f"{slug}_estimated_calls_per_turn"] = fmt(usage.get(f"{slug}_estimated_calls_per_turn"), 2)
                row[f"{slug}_usd_per_turn"] = fmt(usage.get(f"{slug}_usd_per_turn"), 6)
                row[f"{slug}_input_tokens_per_turn"] = fmt(usage.get(f"{slug}_input_tokens_per_turn"), 1)
                row[f"{slug}_cached_input_tokens_per_turn"] = fmt(usage.get(f"{slug}_cached_input_tokens_per_turn"), 1)
                row[f"{slug}_completion_tokens_per_turn"] = fmt(usage.get(f"{slug}_completion_tokens_per_turn"), 1)
        rows.append(row)
    return rows


def assemble_prism_baselines(no_profile_eval: bool) -> List[Dict[str, Any]]:
    ids = prism_legacy_user_ids()
    main_primary = select_users(load_trace_users(PRISM_RESULT, False), ids)
    main_embedding_root = PRISM_RESULT_LEGACY if PRISM_RESULT_LEGACY.exists() else PRISM_RESULT
    main_embedding = select_users(load_trace_users(main_embedding_root, False), ids)
    runs = [
        {
            "label": "PT (ours)",
            "path": PRISM_RESULT,
            "users": hybrid_metric_users(
                main_primary,
                main_embedding,
                LEGACY_EMBEDDING_TURN0_OFFSET if main_embedding_root == PRISM_RESULT_LEGACY else 0.0,
            ),
        },
    ]
    for label, path, kind in PRISM_BASELINES:
        result_path = PRISM_BASELINE_LEGACY_ROOT / path.name
        if not result_path.exists():
            result_path = path
        all_users = load_baseline_users(result_path)
        if not all_users:
            result_path = path
            all_users = load_baseline_users(result_path)
        users = select_users(all_users, ids)
        if label in PRISM_PROFILE_SOURCES:
            _, source = PRISM_PROFILE_SOURCES[label]
            ensure_profile_cache(
                result_path,
                source,
                ids,
                Path("config/run/openrouter_gpt5_trace_gemini3_flash_eval.yaml"),
                no_profile_eval,
            )
            attach_profile_cache(users, result_path)
        runs.append({"label": label, "path": result_path, "users": users, "getter": hydra_metric_value if kind == "hydra" else metric_value})
    print(f"PRISM matched users: {len(ids)}")
    return runs


def assemble_prism_claude_eval() -> List[Dict[str, Any]]:
    ids = prism_legacy_user_ids()
    main_primary = select_users(load_trace_users(PRISM_RESULT, False, CLAUDE_EVAL_METRICS_NAME), ids)
    main_legacy_root = PRISM_RESULT_LEGACY if PRISM_RESULT_LEGACY.exists() else PRISM_RESULT
    main_legacy = select_users(load_trace_users(main_legacy_root, False), ids)
    runs = [
        {
            "label": "PT (ours)",
            "path": PRISM_RESULT,
            "users": hybrid_metric_users(
                main_primary,
                main_legacy,
                LEGACY_EMBEDDING_TURN0_OFFSET if main_legacy_root == PRISM_RESULT_LEGACY else 0.0,
            ),
        },
    ]
    for label, path, kind in PRISM_BASELINES:
        if kind == "hydra":
            continue
        result_path = PRISM_BASELINE_LEGACY_ROOT / path.name
        primary_users_all = load_baseline_users(result_path, CLAUDE_EVAL_METRICS_NAME) if result_path.exists() else {}
        if not primary_users_all:
            result_path = path
            primary_users_all = load_baseline_users(result_path, CLAUDE_EVAL_METRICS_NAME)
        users = select_users(primary_users_all, ids)
        if label in PRISM_PROFILE_SOURCES:
            attach_profile_cache(users, result_path, profile_dir_name=f"profile_metrics_{CLAUDE_EVAL_METRICS_NAME}")
        runs.append({"label": label, "path": result_path, "users": users})
    print(f"PRISM Claude-eval matched users: {len(ids)}")
    return runs


def assemble_personamem_baselines(no_profile_eval: bool, refresh_profile_eval: bool = False) -> List[Dict[str, Any]]:
    ids = personamem_user_ids()
    ensure_trace_profile_scores(
        PERSONAMEM_RESULT,
        ids,
        Path("config/run/openrouter_hybrid_trace_personamem_v2_gemini3_flash_eval.yaml"),
        no_profile_eval,
        refresh_profile_eval=refresh_profile_eval,
        required_keys=PERSONAMEM_PROFILE_KEYS,
    )
    runs = [{"label": "PT (ours)", "path": PERSONAMEM_RESULT, "users": select_users(load_trace_users(PERSONAMEM_RESULT, False), ids)}]
    for label, path, kind in PERSONAMEM_BASELINES:
        users = select_users(load_baseline_users(path), ids)
        if label in PERSONAMEM_PROFILE_SOURCES:
            _, source = PERSONAMEM_PROFILE_SOURCES[label]
            ensure_profile_cache(
                path,
                source,
                ids,
                Path("config/run/openrouter_hybrid_trace_personamem_v2_gemini3_flash_eval.yaml"),
                no_profile_eval,
                refresh_profile_eval=refresh_profile_eval,
                required_keys=PERSONAMEM_PROFILE_KEYS,
            )
            attach_profile_cache(users, path)
        runs.append({"label": label, "path": path, "users": users, "getter": hydra_metric_value if kind == "hydra" else metric_value})
    print(f"PersonaMem-v2 selected users: {len(ids)}")
    return runs


def assemble_result_runs(run_specs: Sequence[Tuple[str, Path]], user_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if user_ids is None:
        sets = [set(load_trace_users(path, False)) for _, path in run_specs]
        user_ids = sorted(set.intersection(*sets) & set(prism_user_ids()))
    return [{"label": label, "path": path, "users": select_users(load_trace_users(path, False), user_ids)} for label, path in run_specs]


def run_prism_baselines(args: argparse.Namespace) -> None:
    runs = assemble_prism_baselines(args.no_profile_eval)
    out = Path("result/plots/prism")
    plot_online_metrics(
        runs,
        out,
        "baselines",
        "PRISM Baselines",
        args.show_raw,
        args.raw_alpha,
        args.min_users,
        args.smooth_window,
        args.main_step,
        highlight_label="PT (ours)",
    )
    plot_profile_bars(runs, out / "baselines_profile.png", "PRISM Profile Alignment")
    plot_profile_bars(runs, out / "baselines_profile_overall.png", "PRISM Overall Profile Alignment", keys=["overall"], labels=["Overall"])
    write_rows(summary_rows(runs, include_cost=False), out / "baselines.csv")
    print(f"Saved PRISM baseline outputs to {out}")


def run_prism_claude_eval(args: argparse.Namespace) -> None:
    runs = assemble_prism_claude_eval()
    out = Path("result/plots/prism_claude_eval")
    plot_online_metrics(
        runs,
        out,
        "baselines_claude_eval",
        "PRISM Baselines (Claude Eval)",
        args.show_raw,
        args.raw_alpha,
        args.min_users,
        args.smooth_window,
        args.main_step,
        highlight_label="PT (ours)",
    )
    profile_runs = [run for run in runs if run["label"] in {"PT (ours)", "Cheatsheet", "HyperAlign"}]
    plot_profile_bars(profile_runs, out / "profile_alignment_claude_eval.png", "PRISM Profile Alignment (Claude Eval)")
    plot_profile_bars(
        profile_runs,
        out / "profile_alignment_overall_claude_eval.png",
        "PRISM Overall Profile Alignment (Claude Eval)",
        keys=["overall"],
        labels=["Overall"],
    )
    write_rows(summary_rows(runs, include_cost=False), out / "baselines_claude_eval.csv")
    print(f"Saved PRISM Claude-eval outputs to {out}")


def run_prism_profile(args: argparse.Namespace) -> None:
    runs = assemble_prism_baselines(args.no_profile_eval)
    profile_runs = [run for run in runs if run["label"] in {"PT (ours)", "Cheatsheet", "HyperAlign"}]
    out = Path("result/plots/prism")
    plot_profile_bars(profile_runs, out / "profile_alignment.png", "PRISM Profile Alignment")
    plot_profile_bars(profile_runs, out / "profile_alignment_overall.png", "PRISM Overall Profile Alignment", keys=["overall"], labels=["Overall"])
    write_rows(summary_rows(profile_runs, include_cost=False), out / "profile_alignment.csv")
    print(f"Saved PRISM profile outputs to {out}")


def run_model_ablation(args: argparse.Namespace) -> None:
    runs = assemble_result_runs(MODEL_RUNS)
    out = Path("result/plots/model_ablation")
    plot_online_metrics(
        runs,
        out,
        "learning_trajectory",
        "Model Ablation",
        args.show_raw,
        args.raw_alpha,
        args.min_users,
        args.smooth_window,
        args.main_step,
    )
    plot_online_metrics_panel(
        runs,
        out / "learning_trajectory_accuracy_gpt_main.png",
        args.min_users,
        args.smooth_window,
        args.main_step,
        metrics=[
            ("prediction_accuracy", "Prediction Accuracy"),
            ("adapt_relative_gpt_score", "Relative GPT Score"),
        ],
    )
    plot_model_ablation_summary_panel(
        runs,
        out / "learning_trajectory_accuracy_gpt_profile_cost_main.png",
        args.min_users,
        args.smooth_window,
        args.main_step,
    )
    plot_profile_bars(runs, out / "profile_alignment.png", "Model Ablation Profile Alignment")
    plot_profile_bars(runs, out / "profile_alignment_overall.png", "Model Ablation Overall Profile Alignment", keys=["overall"], labels=["Overall"])
    reference_usage = cost_usage(runs[0]["path"], runs[0]["users"]) if runs and runs[0].get("path") else None
    write_rows(summary_rows(runs, include_cost=True, reference_cost_usage=reference_usage), out / "learning_trajectory.csv")
    print(f"Model ablation matched users: {len(next(iter(runs))['users']) if runs else 0}")
    print(f"Saved model ablation outputs to {out}")


def run_prism_ablation(args: argparse.Namespace) -> None:
    runs = assemble_result_runs(PRISM_ABLATION_RUNS, prism_user_ids())
    for run in runs:
        run["cost_mini_as_gpt"] = run["label"] != "PT(hybrid)"
    out = Path("result/plots/ablation")
    plot_online_metrics(
        runs,
        out,
        "prism_method_ablation",
        "PRISM Method Ablation",
        args.show_raw,
        args.raw_alpha,
        args.min_users,
        args.smooth_window,
        args.main_step,
    )
    plot_online_metrics_panel(
        runs,
        out / "prism_method_ablation_accuracy_gpt_main.png",
        args.min_users,
        args.smooth_window,
        args.main_step,
        metrics=[
            ("prediction_accuracy", "Prediction Accuracy"),
            ("adapt_relative_gpt_score", "Relative GPT Score"),
        ],
    )
    plot_model_ablation_summary_panel(
        runs,
        out / "prism_method_ablation_accuracy_gpt_profile_cost_main.png",
        args.min_users,
        args.smooth_window,
        args.main_step,
    )
    plot_profile_bars(runs, out / "prism_method_ablation_profile.png", "PRISM Method Ablation Profile")
    plot_profile_bars(runs, out / "prism_method_ablation_profile_overall.png", "PRISM Method Ablation Overall Profile", keys=["overall"], labels=["Overall"])
    write_rows(summary_rows(runs, include_cost=True), out / "prism_method_ablation.csv")
    print(f"Saved PRISM ablation outputs to {out}")


def run_personamem_baselines(args: argparse.Namespace) -> None:
    runs = assemble_personamem_baselines(args.no_profile_eval, getattr(args, "refresh_profile_eval", False))
    out = Path("result/plots/personamem")
    plot_online_metrics(
        runs,
        out,
        "baselines",
        "PersonaMem-v2 Baselines",
        args.show_raw,
        args.raw_alpha,
        args.min_users,
        args.smooth_window,
        args.main_step,
        highlight_label="PT (ours)",
    )
    plot_profile_bars(
        runs,
        out / "baselines_profile.png",
        "PersonaMem-v2 Profile Alignment",
        keys=PERSONAMEM_PROFILE_KEYS,
        labels=PERSONAMEM_PROFILE_LABELS,
    )
    plot_profile_bars(runs, out / "baselines_profile_overall.png", "PersonaMem-v2 Overall Profile Alignment", keys=["overall"], labels=["Overall"])
    write_rows(summary_rows(runs, include_cost=False), out / "baselines.csv")
    print(f"Saved PersonaMem-v2 outputs to {out}")


def run_all(args: argparse.Namespace) -> None:
    run_prism_baselines(args)
    run_prism_claude_eval(args)
    run_prism_profile(args)
    run_model_ablation(args)
    run_prism_ablation(args)
    run_personamem_baselines(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified visualization for preference tracing experiments.")
    parser.add_argument(
        "command",
        choices=[
            "all",
            "prism-baselines",
            "prism-claude-eval",
            "prism-profile",
            "model-ablation",
            "prism-ablation",
            "personamem-baselines",
        ],
    )
    parser.add_argument("--no-profile-eval", action="store_true", help="Use cached profile metrics only.")
    parser.add_argument("--refresh-profile-eval", action="store_true", help="Recompute PersonaMem-v2 profile scores with the current rubric.")
    parser.add_argument("--show-raw", action="store_true", help="Show unsmoothed turn metrics behind smoothed curves.")
    parser.add_argument("--raw-alpha", type=float, default=RAW_ALPHA)
    parser.add_argument("--min-users", type=int, default=MIN_USERS_PER_TURN)
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW, help="Centered rolling-average window.")
    parser.add_argument("--main-step", type=int, default=MAIN_TEXT_STEP, help="Turn-bin width for main-text online plots.")
    args = parser.parse_args()

    {
        "all": run_all,
        "prism-baselines": run_prism_baselines,
        "prism-claude-eval": run_prism_claude_eval,
        "prism-profile": run_prism_profile,
        "model-ablation": run_model_ablation,
        "prism-ablation": run_prism_ablation,
        "personamem-baselines": run_personamem_baselines,
    }[args.command](args)


if __name__ == "__main__":
    main()
