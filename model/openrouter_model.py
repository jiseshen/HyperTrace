import asyncio
import copy
import json
import logging
import os
import re
import time
from dataclasses import replace
from typing import Any, Dict, Optional, Union, Unpack

from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel

from .base import BaseLM, GenerationConfig, GenerationOverrides
from .openai_model import REASONING_BUDGETS, REASONING_PREFIXES

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REASONING_PREFIXES = ("openai/gpt-5", "openai/o", "google/gemini-3")
OPENROUTER_REASONING_MODELS = {
    "deepseek/deepseek-v4-pro",
    "minimax/minimax-m2.7",
}
OPENROUTER_DISABLE_THINKING_MODELS = {
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
}
logger = logging.getLogger(__name__)


class EmptyContentError(ValueError):
    pass


class OpenRouterModel(BaseLM):
    def __init__(self, api_key: Optional[str] = None, default_cfg: Optional[GenerationConfig] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "empty")
        self.default_cfg = default_cfg or GenerationConfig(backend="openrouter")
        self.base_url = self.default_cfg.base_url or os.getenv("OPENROUTER_API_BASE", OPENROUTER_BASE_URL)
        default_headers = self._default_headers()
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=default_headers)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=default_headers)

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

    @classmethod
    def _empty_schema_output(cls, schema: type[BaseModel]) -> Dict[str, Any]:
        root_schema = schema.model_json_schema()
        fallback = cls._empty_value_from_schema(root_schema, root_schema)
        return schema.model_validate(fallback).model_dump()

    @classmethod
    def _empty_value_from_schema(
        cls,
        node: Dict[str, Any],
        root_schema: Dict[str, Any],
        index: int = 0,
        field_name: str = "",
    ) -> Any:
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            node = root_schema.get("$defs", {}).get(ref_name, node)
        if "enum" in node and node["enum"]:
            return node["enum"][0]
        if "const" in node:
            return node["const"]
        if "anyOf" in node:
            non_null = [item for item in node["anyOf"] if item.get("type") != "null"]
            return cls._empty_value_from_schema(non_null[0] if non_null else node["anyOf"][0], root_schema, index, field_name)

        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            return {
                name: cls._empty_value_from_schema(prop, root_schema, index=index, field_name=name)
                for name, prop in node.get("properties", {}).items()
            }
        if node_type == "array":
            min_items = node.get("minItems", 0)
            item_schema = node.get("items", {})
            return [
                cls._empty_value_from_schema(item_schema, root_schema, index=i, field_name=field_name)
                for i in range(min_items)
            ]
        if node_type == "integer":
            return index if field_name == "i" else 0
        if node_type == "number":
            return node.get("minimum", 0)
        if node_type == "boolean":
            return False
        if node_type == "string":
            return f"empty-{index}" if field_name == "id" else "empty"
        return "empty"

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
            try:
                resp = self.client.chat.completions.create(**kwargs)
                text, reasoning = self._extract_message(resp)
                if not text.strip():
                    raise EmptyContentError("OpenRouter generation returned empty content.")
                output = self._parse_schema_output(text, schema) if schema else text
            except (APIError, RateLimitError, ValueError) as e:
                if attempt == retries - 1:
                    if isinstance(e, EmptyContentError):
                        logger.warning("OpenRouter returned empty content after %s attempts; using fallback 'empty'.", retries)
                        output = self._empty_schema_output(schema) if schema else "empty"
                        result = {"output": output}
                        if cfg.reasoning_summary:
                            result["reasoning"] = None
                        return result
                    raise
                time.sleep(cfg.retry_delay)
                continue
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

        async def _one(prompt: str) -> Union[Dict[str, Any], Exception]:
            async with sem:
                retries = cfg.max_retries
                kwargs = self._build_chat_kwargs(prompt, cfg, schema)
                for attempt in range(retries):
                    try:
                        resp = await self.async_client.chat.completions.create(**kwargs)
                        text, reasoning = self._extract_message(resp)
                        if not text.strip():
                            raise EmptyContentError("OpenRouter generation returned empty content.")
                        output = self._parse_schema_output(text, schema) if schema else text
                    except (APIError, RateLimitError, ValueError) as e:
                        if attempt == retries - 1:
                            if isinstance(e, EmptyContentError):
                                logger.warning("OpenRouter returned empty content after %s attempts; using fallback 'empty'.", retries)
                                output = self._empty_schema_output(schema) if schema else "empty"
                                result = {"output": output}
                                if cfg.reasoning_summary:
                                    result["reasoning"] = None
                                return result
                            raise
                        await asyncio.sleep(cfg.retry_delay)
                        continue
                    result = {"output": output}
                    if cfg.reasoning_summary:
                        result["reasoning"] = reasoning
                    return result
            raise RuntimeError("OpenRouter generation failed without returning a result.")

        tasks = [_one(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
