from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from model import GenerationConfig, load_model
from model.openrouter_model import OpenRouterModel


MODEL_CONFIG_PATHS = [
    "config/model/openrouter-kimi-k2.6.yaml",
    "config/model/openrouter-minimax-m2.7.yaml",
    "config/model/openrouter-glm-5.1.yaml",
    "config/model/openrouter-deepseek-v4-pro.yaml",
]

CASE_PROMPT = """You are testing a preference-tracing model.

Case:
User asks: "I need a restaurant for a team dinner. Keep it quiet, vegetarian-friendly, and within a 15 minute walk."

Candidate A:
"Book the downtown steakhouse. It is lively and famous for dry-aged beef."

Candidate B:
"Choose the nearby Mediterranean place. It has many vegetarian dishes, a quieter dining room, and is a 10 minute walk."

The user chose Candidate B.

Explain the likely user preference in one compact paragraph."""

STRUCTURED_PROMPT = CASE_PROMPT + """

Return only a JSON object with exactly these fields:
- "chosen_candidate": one of "A" or "B"
- "preference_hypothesis": a concise latent user preference string
- "confidence": a number between 0.0 and 1.0

Do not include any other fields. Do not wrap the JSON in markdown."""


class PreferenceProbe(BaseModel):
    chosen_candidate: Literal["A", "B"] = Field(..., description="The candidate selected by the user.")
    preference_hypothesis: str = Field(..., description="A concise latent user preference.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the hypothesis.")


def _load_cfg(path: Path, base_max_tokens: int) -> GenerationConfig:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    cfg = GenerationConfig(**data)
    return replace(cfg, max_tokens=base_max_tokens)


def _usage_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _run_probe(model: OpenRouterModel, cfg: GenerationConfig) -> dict:
    plain_output = model.generate(CASE_PROMPT, cfg=cfg)["output"]

    structured_kwargs = model._build_chat_kwargs(STRUCTURED_PROMPT, cfg, schema=PreferenceProbe)
    wrapper_error = None
    wrapper_output = None
    try:
        wrapper_output = model.generate(STRUCTURED_PROMPT, schema=PreferenceProbe, cfg=cfg)["output"]
    except Exception as exc:
        wrapper_error = f"{type(exc).__name__}: {exc}"

    structured_resp = model.client.chat.completions.create(**structured_kwargs)
    structured_text, _ = model._extract_message(structured_resp)
    structured_error = None
    structured_output = None
    try:
        structured_output = model._parse_schema_output(structured_text, PreferenceProbe)
    except Exception as exc:
        structured_error = f"{type(exc).__name__}: {exc}"

    return {
        "model": cfg.model,
        "sent_max_tokens": structured_kwargs["max_tokens"],
        "plain": {
            "has_think_tag": "<think" in plain_output.lower(),
            "text": plain_output,
        },
        "structured": {
            "wrapper_parsed": wrapper_output,
            "wrapper_error": wrapper_error,
            "usage": _usage_dict(structured_resp),
            "has_think_tag": "<think" in structured_text.lower(),
            "raw_text": structured_text,
            "direct_parsed": structured_output,
            "direct_error": structured_error,
            "plugins": structured_kwargs.get("extra_body", {}).get("plugins", []),
        },
    }


def main() -> None:
    parser = ArgumentParser(description="Smoke test OpenRouter structured output and healing plugin support.")
    parser.add_argument("--config", action="append", default=None, help="Model config path. Repeat to test multiple models.")
    parser.add_argument("--base-max-tokens", type=int, default=1024, help="Base max_tokens before config max_tokens_extra is added.")
    parser.add_argument("--output", type=str, default="result/openrouter_structured_smoke.json", help="Where to save raw probe results.")
    args = parser.parse_args()

    config_paths = [Path(p) for p in (args.config or MODEL_CONFIG_PATHS)]
    results = []
    for path in config_paths:
        cfg = _load_cfg(path, args.base_max_tokens)
        loaded = load_model(backend=cfg.backend, default_cfg=cfg)
        if not isinstance(loaded, OpenRouterModel):
            raise TypeError(f"{path} did not load an OpenRouterModel")
        print(f"Testing {cfg.model} with max_tokens={cfg.max_tokens + cfg.max_tokens_extra}")
        try:
            result = _run_probe(loaded, cfg)
        except Exception as exc:
            result = {
                "model": cfg.model,
                "sent_max_tokens": cfg.max_tokens + cfg.max_tokens_extra,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
