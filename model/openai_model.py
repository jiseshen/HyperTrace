from .credentials import require_api_key
from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError
from .base import BaseLM, GenerationConfig, GenerationOverrides
from pydantic import BaseModel
from dataclasses import replace
from typing import Optional, Union, List, Dict, Any, Unpack
import time
import os
import asyncio
import logging

REASONING_PREFIXES = ("gpt-5", "o")
REASONING_BUDGETS = {"none": 0, "minimal": 128, "low": 1024, "medium": 4096, "high": 16384}
logger = logging.getLogger(__name__)


class OpenAIModel(BaseLM):
    def __init__(self, api_key: Optional[str] = None, default_cfg: Optional[GenerationConfig] = None):
        self.api_key = require_api_key("OPENAI_API_KEY", api_key)
        self.default_cfg = default_cfg or GenerationConfig()
        self.base_url = self.default_cfg.base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _make_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _resolve_cfg(self, cfg: Optional[GenerationConfig], overrides: dict) -> GenerationConfig:
        cfg = cfg or self.default_cfg
        if overrides:
            return replace(cfg, **overrides)
        return cfg
    
    def _build_responses_kwargs(self, prompt: str, cfg: GenerationConfig):
        model = cfg.model
        reasoning = model.startswith(REASONING_PREFIXES)
        kwargs = {
            "model": model,
            "input": prompt,
            "max_output_tokens": cfg.max_tokens,
        }
        if reasoning:
            reasoning_budget = REASONING_BUDGETS.get(cfg.reasoning_effort, 0)
            kwargs["max_output_tokens"] += reasoning_budget
            kwargs["reasoning"] = {
                "effort": cfg.reasoning_effort,
                "summary": cfg.reasoning_summary
            }
            kwargs["text"] = {
                "verbosity": cfg.verbosity
            }
        else:
            kwargs["temperature"] = cfg.temperature
        return kwargs

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None, cfg: Optional[GenerationConfig] = None, **overrides: Unpack[GenerationOverrides]) -> Dict[str, Any]:
        cfg = self._resolve_cfg(cfg, overrides)
        retries = cfg.max_retries
        kwargs = self._build_responses_kwargs(prompt, cfg)
        for attempt in range(retries):
            try:
                if schema:
                    resp = self.client.responses.parse(**kwargs, text_format=schema)
                    output = resp.output_parsed.model_dump()
                else:
                    resp = self.client.responses.create(**kwargs)
                    output = resp.output_text
            except (APIError, RateLimitError):
                if attempt == retries - 1:
                    raise
                time.sleep(cfg.retry_delay)
                continue
            if cfg.reasoning_summary:
                return {"output": output, "reasoning": resp.output[0].summary[0].text}
            return {"output": output}
    
    async def async_generate(self, prompts: list[str], schema: Optional[type[BaseModel]] = None, cfg: Optional[GenerationConfig] = None, concurrency: int = 5, return_exceptions: bool = True, **overrides: Unpack[GenerationOverrides]) -> list[Union[Dict[str, Any], Exception]]:
        cfg = self._resolve_cfg(cfg, overrides)
        sem = asyncio.Semaphore(concurrency)
        async with self._make_async_client() as async_client:
            async def _one(prompt: str) -> Union[Dict[str, Any], Exception]:
                async with sem:
                    retries = cfg.max_retries
                    for attempt in range(retries):
                        try:
                            kwargs = self._build_responses_kwargs(prompt, cfg)
                            if schema:
                                resp = await async_client.responses.parse(**kwargs, text_format=schema)
                                output = resp.output_parsed.model_dump()
                            else:
                                resp = await async_client.responses.create(**kwargs)
                                output = resp.output_text
                        except (APIError, RateLimitError):
                            if attempt == retries - 1:
                                raise
                            await asyncio.sleep(cfg.retry_delay)
                            continue
                        if cfg.reasoning_summary:
                            return {"output": output, "reasoning": resp.output[0].summary[0].text}
                        return {"output": output}
            tasks = [_one(p) for p in prompts]
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            logger.warning("OpenAI sync client close failed; ignoring during cleanup.", exc_info=True)
        
    def _build_batch_line(self, prompt: str, cfg: GenerationConfig, custom_id: str) -> Dict[str, Any]:
        body = self._build_responses_kwargs(prompt, cfg)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
    
