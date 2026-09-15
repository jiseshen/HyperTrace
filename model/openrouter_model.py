from .credentials import require_api_key
import asyncio
import copy
import json
import logging
import os
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Union, Unpack

from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel

from .base import BaseLM, GenerationConfig, GenerationOverrides
from .openai_model import REASONING_BUDGETS, REASONING_PREFIXES

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REASONING_PREFIXES = ("openai/gpt-5", "openai/o", "google/gemini-3")
OPENROUTER_REASONING_MODELS = {
    "deepseek/deepseek-v4-pro",
}
OPENROUTER_DISABLE_THINKING_MODELS = {
    "minimax/minimax-m2.7",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-9b",
    "z-ai/glm-5.1",
}
logger = logging.getLogger(__name__)


class EmptyContentError(ValueError):
    pass


class OpenRouterModel(BaseLM):
    def __init__(self, api_key: Optional[str] = None, default_cfg: Optional[GenerationConfig] = None):
        self.api_key = require_api_key("OPENROUTER_API_KEY", api_key)
        self.default_cfg = default_cfg or GenerationConfig(backend="openrouter")
        self.base_url = self.default_cfg.base_url or os.getenv("OPENROUTER_API_BASE", OPENROUTER_BASE_URL)
        default_headers = self._default_headers()
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=default_headers)
        self._report_lock = threading.Lock()
        self._provider_attempts: list[Dict[str, Any]] = []

    def _make_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=self._default_headers())

    def _default_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        return headers

    def _resolve_cfg(self, cfg: Optional[GenerationConfig], overrides: Dict[str, Any]) -> GenerationConfig:
        cfg = cfg or self.default_cfg
        if overrides:
            return replace(cfg, **overrides)
        return cfg

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        return model.startswith(OPENROUTER_REASONING_PREFIXES) or model.startswith(REASONING_PREFIXES)

    @classmethod
    def _default_extra_body(cls, cfg: GenerationConfig) -> Dict[str, Any]:
        model = cfg.model
        extra_body: Dict[str, Any] = {}
        if cfg.reasoning_effort == "none" and model in OPENROUTER_DISABLE_THINKING_MODELS:
            extra_body["effort"] = "none"
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            extra_body["reasoning"] = {"enabled": False}
            if model == "moonshotai/kimi-k2.6":
                extra_body["chat_template_kwargs"]["thinking"] = False
                extra_body["thinking"] = {"type": "disabled"}
        elif cfg.reasoning_effort != "none" and model in OPENROUTER_REASONING_MODELS:
            extra_body["reasoning"] = {"effort": cfg.reasoning_effort}
        return extra_body

    @staticmethod
    def _schema_name(schema: type[BaseModel]) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", schema.__name__).strip("_") or "structured_output"

    @classmethod
    def _strict_json_schema(cls, schema: type[BaseModel]) -> Dict[str, Any]:
        json_schema = copy.deepcopy(schema.model_json_schema())
        cls._close_object_schemas(json_schema)
        return json_schema

    @classmethod
    def _close_object_schemas(cls, node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
            for value in node.values():
                cls._close_object_schemas(value)
        elif isinstance(node, list):
            for item in node:
                cls._close_object_schemas(item)

    @classmethod
    def _merge_extra_body(cls, base: Dict[str, Any], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not update:
            return base
        for key, value in update.items():
            if isinstance(base.get(key), dict) and isinstance(value, dict):
                cls._merge_extra_body(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    def _build_chat_kwargs(self, prompt: str, cfg: GenerationConfig, schema: Optional[type[BaseModel]] = None) -> Dict[str, Any]:
        model = cfg.model
        reasoning = self._is_reasoning_model(model)
        max_tokens = cfg.max_tokens + cfg.max_tokens_extra
        extra_body = self._default_extra_body(cfg)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if reasoning and cfg.reasoning_effort:
            if cfg.reasoning_effort != "none":
                max_tokens += REASONING_BUDGETS.get(cfg.reasoning_effort, 0)
                kwargs["max_tokens"] = max_tokens
            extra_body["reasoning"] = {
                "effort": cfg.reasoning_effort,
                "exclude": cfg.reasoning_summary is None,
            }
        else:
            kwargs["temperature"] = cfg.temperature

        if not reasoning:
            kwargs["top_p"] = cfg.top_p
            kwargs["presence_penalty"] = cfg.presence_penalty
            extra_body["top_k"] = cfg.top_k
            extra_body["repetition_penalty"] = cfg.repetition_penalty

        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self._schema_name(schema),
                    "strict": True,
                    "schema": self._strict_json_schema(schema),
                },
            }
            extra_body["plugins"] = [{"id": "response-healing"}]

        self._merge_extra_body(extra_body, cfg.extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _parse_schema_output(self, text: str, schema: type[BaseModel]) -> Dict[str, Any]:
        parsed = self._parse_json_text(text)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("Failed to parse OpenRouter structured output as JSON.")
        return schema.model_validate(parsed).model_dump()

    @staticmethod
    def _parse_json_text(text: str) -> Optional[Any]:
        cleaned = text.strip()
        if not cleaned:
            return None
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        first_brace = cleaned.find("{")
        first_bracket = cleaned.find("[")
        starts = [idx for idx in (first_brace, first_bracket) if idx >= 0]
        if not starts:
            return None
        snippet = cleaned[min(starts):]
        for end in range(len(snippet), 0, -1):
            try:
                return json.loads(snippet[:end].strip())
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _extract_message(resp: Any) -> tuple[str, Optional[Any]]:
        message = resp.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_details", None)
        return content, reasoning

    @classmethod
    def _extract_provider(cls, resp: Any) -> str:
        for key, value in cls._iter_response_fields(resp):
            if key in {"provider", "provider_name", "model_provider", "route", "provider_slug"} and value:
                return str(value)
            if key in {"app_infos", "app_info"} and isinstance(value, list) and value:
                provider = cls._provider_from_app_info(value[0])
                if provider:
                    return provider
        return "unknown"

    @classmethod
    def _provider_from_app_info(cls, app_info: Any) -> Optional[str]:
        if isinstance(app_info, dict):
            for key in ("provider", "provider_name", "provider_slug", "slug", "name"):
                value = app_info.get(key)
                if value:
                    return str(value)
        return None

    @classmethod
    def _iter_response_fields(cls, obj: Any, seen: Optional[set[int]] = None):
        if obj is None:
            return
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield str(key), value
                if isinstance(value, (dict, list, tuple)) or hasattr(value, "model_extra"):
                    yield from cls._iter_response_fields(value, seen)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                if isinstance(value, (dict, list, tuple)) or hasattr(value, "model_extra") or not isinstance(value, (str, int, float, bool)):
                    yield from cls._iter_response_fields(value, seen)
            return
        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict):
            yield from cls._iter_response_fields(model_extra, seen)
        for key in ("provider", "provider_name", "model_provider", "route", "provider_slug", "app_infos", "app_info", "choices", "message", "usage"):
            value = getattr(obj, key, None)
            if value is not None:
                yield key, value
                if isinstance(value, (dict, list, tuple)) or hasattr(value, "model_extra") or not isinstance(value, (str, int, float, bool)):
                    yield from cls._iter_response_fields(value, seen)

    @staticmethod
    def _serialize_openai_obj(obj: Any) -> Any:
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {key: OpenRouterModel._serialize_openai_obj(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [OpenRouterModel._serialize_openai_obj(value) for value in obj]
        return str(obj)

    def _response_metadata(self, resp: Any) -> Dict[str, Any]:
        generation_id = getattr(resp, "id", None)
        return {
            "generation_id": generation_id,
            "response_model": getattr(resp, "model", None),
            "usage": self._serialize_openai_obj(getattr(resp, "usage", None)),
            "provider": self._extract_provider(resp),
        }

    def _error_metadata(self, error: BaseException) -> Dict[str, Any]:
        body = getattr(error, "body", None)
        provider = self._extract_provider(body)
        return {
            "provider": provider,
            "error_body": self._serialize_openai_obj(body),
        }

    def _record_provider_attempt(
        self,
        *,
        cfg: GenerationConfig,
        latency_seconds: float,
        success: bool,
        attempt: int,
        schema: bool,
        response_metadata: Optional[Dict[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        response_metadata = response_metadata or {}
        entry = {
            "model": cfg.model,
            "provider": response_metadata.get("provider", "unknown"),
            "generation_id": response_metadata.get("generation_id"),
            "response_model": response_metadata.get("response_model"),
            "usage": response_metadata.get("usage"),
            "error_body": response_metadata.get("error_body"),
            "latency_seconds": latency_seconds,
            "success": success,
            "attempt": attempt,
            "schema": schema,
            "error_type": type(error).__name__ if error else None,
            "error": str(error) if error else None,
            "timestamp": time.time(),
        }
        with self._report_lock:
            self._provider_attempts.append(entry)

    @staticmethod
    def _mean(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def provider_report(self) -> Dict[str, Any]:
        with self._report_lock:
            attempts = list(self._provider_attempts)
        by_provider: Dict[str, Dict[str, Any]] = {}
        for entry in attempts:
            provider = entry["provider"]
            bucket = by_provider.setdefault(
                provider,
                {
                    "attempts": 0,
                    "successes": 0,
                    "errors": 0,
                    "latencies": [],
                    "success_latencies": [],
                    "error_types": {},
                    "models": {},
                },
            )
            bucket["attempts"] += 1
            bucket["models"][entry["model"]] = bucket["models"].get(entry["model"], 0) + 1
            bucket["latencies"].append(entry["latency_seconds"])
            if entry["success"]:
                bucket["successes"] += 1
                bucket["success_latencies"].append(entry["latency_seconds"])
            else:
                bucket["errors"] += 1
                error_type = entry["error_type"] or "UnknownError"
                bucket["error_types"][error_type] = bucket["error_types"].get(error_type, 0) + 1
        for bucket in by_provider.values():
            attempts_count = bucket["attempts"]
            bucket["error_rate"] = bucket["errors"] / attempts_count if attempts_count else None
            bucket["avg_latency_seconds"] = self._mean(bucket.pop("latencies"))
            bucket["avg_success_latency_seconds"] = self._mean(bucket.pop("success_latencies"))
        return {
            "base_url": self.base_url,
            "total_attempts": len(attempts),
            "providers": by_provider,
            "attempts": attempts,
        }

    def dump_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.provider_report(), f, indent=4)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            logger.warning("OpenRouter sync client close failed; ignoring during cleanup.", exc_info=True)

    def generate(
        self,
        prompt: str,
        schema: Optional[type[BaseModel]] = None,
        cfg: Optional[GenerationConfig] = None,
        **overrides: Unpack[GenerationOverrides],
    ) -> Dict[str, Any]:
        cfg = self._resolve_cfg(cfg, overrides)
        retries = cfg.max_retries
        kwargs = self._build_chat_kwargs(prompt, cfg, schema)
        for attempt in range(retries):
            start = time.perf_counter()
            response_metadata: Dict[str, Any] = {}
            try:
                resp = self.client.chat.completions.create(**kwargs)
                response_metadata = self._response_metadata(resp)
                text, reasoning = self._extract_message(resp)
                if not text.strip():
                    raise EmptyContentError("OpenRouter generation returned empty content.")
                output = self._parse_schema_output(text, schema) if schema else text
            except (APIError, RateLimitError, ValueError) as e:
                if not response_metadata:
                    response_metadata = self._error_metadata(e)
                self._record_provider_attempt(
                    cfg=cfg,
                    latency_seconds=time.perf_counter() - start,
                    success=False,
                    attempt=attempt + 1,
                    schema=schema is not None,
                    response_metadata=response_metadata,
                    error=e,
                )
                if attempt == retries - 1:
                    raise
                time.sleep(cfg.retry_delay)
                continue
            self._record_provider_attempt(
                cfg=cfg,
                latency_seconds=time.perf_counter() - start,
                success=True,
                attempt=attempt + 1,
                schema=schema is not None,
                response_metadata=response_metadata,
            )
            result = {"output": output}
            if cfg.reasoning_summary:
                result["reasoning"] = reasoning
            return result
        raise RuntimeError("OpenRouter generation failed without returning a result.")

    async def async_generate(
        self,
        prompts: list[str],
        schema: Optional[type[BaseModel]] = None,
        cfg: Optional[GenerationConfig] = None,
        concurrency: int = 5,
        return_exceptions: bool = True,
        **overrides: Unpack[GenerationOverrides],
    ) -> list[Union[Dict[str, Any], Exception]]:
        cfg = self._resolve_cfg(cfg, overrides)
        sem = asyncio.Semaphore(concurrency)
        async with self._make_async_client() as async_client:
            async def _one(prompt: str) -> Union[Dict[str, Any], Exception]:
                async with sem:
                    retries = cfg.max_retries
                    kwargs = self._build_chat_kwargs(prompt, cfg, schema)
                    for attempt in range(retries):
                        start = time.perf_counter()
                        response_metadata: Dict[str, Any] = {}
                        try:
                            resp = await async_client.chat.completions.create(**kwargs)
                            response_metadata = self._response_metadata(resp)
                            text, reasoning = self._extract_message(resp)
                            if not text.strip():
                                raise EmptyContentError("OpenRouter generation returned empty content.")
                            output = self._parse_schema_output(text, schema) if schema else text
                        except (APIError, RateLimitError, ValueError) as e:
                            if not response_metadata:
                                response_metadata = self._error_metadata(e)
                            self._record_provider_attempt(
                                cfg=cfg,
                                latency_seconds=time.perf_counter() - start,
                                success=False,
                                attempt=attempt + 1,
                                schema=schema is not None,
                                response_metadata=response_metadata,
                                error=e,
                            )
                            if attempt == retries - 1:
                                raise
                            await asyncio.sleep(cfg.retry_delay)
                            continue
                        self._record_provider_attempt(
                            cfg=cfg,
                            latency_seconds=time.perf_counter() - start,
                            success=True,
                            attempt=attempt + 1,
                            schema=schema is not None,
                            response_metadata=response_metadata,
                        )
                        result = {"output": output}
                        if cfg.reasoning_summary:
                            result["reasoning"] = reasoning
                        return result
                raise RuntimeError("OpenRouter generation failed without returning a result.")

            tasks = [_one(p) for p in prompts]
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
