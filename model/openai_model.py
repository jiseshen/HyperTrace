from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError
from .base import BaseLM
from .utils import Parser, ParseError
from pydantic import BaseModel
from dataclasses import dataclass, replace
from typing import Optional, Union, Tuple, List, Dict, Any
import json
import time
import os
import asyncio

REASONING_PREFIXES = ("gpt-5", "o")
REASONING_BUDGETS = {"none": 0, "minimal": 32, "low": 128, "medium": 256, "high": 1024}


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "gpt-5-nano"
    max_tokens: int = 128
    temperature: float = 0.0
    reasoning_effort: str = "minimal"
    reasoning_summary: Optional[str] = None
    verbosity: str = "low"
    max_retries: int = 3
    retry_delay: float = 0.5
    completion_window: str = "24h"
    poll_interval: float = 60.0
    timeout: Optional[float] = None
    

class OpenAIModel(BaseLM):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "empty")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.default_cfg = GenerationConfig(model=model)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _resolve_cfg(self, cfg: Optional[GenerationConfig], overrides: dict) -> GenerationConfig:
        cfg = cfg or self.default_cfg
        if overrides:
            return replace(cfg, **overrides)
        return cfg
    
    def _build_responses_kwargs(self, prompt: str, cfg: GenerationConfig):
        model = cfg.model or self.model
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
    
    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None, cfg: Optional[GenerationConfig] = None, **overrides) -> Dict[str, str]:
        cfg = self._resolve_cfg(cfg, overrides)
        retries = cfg.max_retries
        kwargs = self._build_responses_kwargs(prompt, cfg)
        for attempt in range(retries):
            try:
                resp = self.client.responses.create(**kwargs)
                output = resp.output_text
                if schema:
                    parser = Parser(schema)
                    output = parser.parse(output)
            except (APIError, RateLimitError, ParseError):
                if attempt == retries - 1:
                    raise
                time.sleep(cfg.retry_delay)
                continue
            if cfg.reasoning_summary:
                return {"output": output, "reasoning": resp.output[0].summary[0].text}
            return {"output": output}
    
    async def async_generate(self, prompts: list[str], schema: Optional[type[BaseModel]] = None, cfg: Optional[GenerationConfig] = None, concurrency: int = 5, return_exceptions: bool = True, **overrides) -> list[Union[Dict[str, str], Exception]]:
        cfg = self._resolve_cfg(cfg, overrides)
        sem = asyncio.Semaphore(concurrency)
        async def _one(prompt: str) -> Union[Dict[str, str], Exception]:
            async with sem:
                retries = cfg.max_retries
                for attempt in range(retries):
                    try:
                        kwargs = self._build_responses_kwargs(prompt, cfg)
                        resp = await self.async_client.responses.create(**kwargs)
                        output = resp.output_text
                        if schema:
                            parser = Parser(schema)
                            output = parser.parse(output)
                    except (APIError, RateLimitError, ParseError):
                        if attempt == retries - 1:
                            raise
                        await asyncio.sleep(cfg.retry_delay)
                        continue
                    if cfg.reasoning_summary:
                        return {"output": output, "reasoning": resp.output[0].summary[0].text}
                    return {"output": output}
        tasks = [_one(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        
    def _build_batch_line(self, prompt: str, cfg: GenerationConfig, custom_id: str) -> Dict[str, Any]:
        body = self._build_responses_kwargs(prompt, cfg)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }

    def submit_responses_batch(
        self,
        prompts: List[str],
        cfg: Optional[GenerationConfig] = None,
        custom_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **overrides
    ) -> str:
        """
        Submit a batch job for /v1/responses.

        Returns:
            batch_id (str)
        """
        cfg = self._resolve_cfg(cfg, overrides)
        if custom_ids is None:
            custom_ids = [f"req-{i}" for i in range(len(prompts))]
        if len(custom_ids) != len(prompts):
            raise ValueError("custom_ids must have the same length as prompts.")

        jsonl_lines = [
            json.dumps(self._build_batch_line(p, cfg, cid), ensure_ascii=False)
            for p, cid in zip(prompts, custom_ids)
        ]
        jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")

        in_file = self.client.files.create(
            file=("batch.jsonl", jsonl_bytes),
            purpose="batch",
        )
        batch = self.client.batches.create(
            input_file_id=in_file.id,
            endpoint="/v1/responses",
            completion_window=cfg.completion_window,
            metadata=metadata or {},
        )
        return batch.id

    def poll_batch_until_done(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        timeout: Optional[float] = None
    ):
        """
        Poll batch status until it reaches a terminal state.
        Returns the final batch object.
        """
        start = time.time()
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)

            if status in ("completed", "failed", "cancelled", "expired"):
                return batch

            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Batch {batch_id} polling timed out after {timeout}s")

            time.sleep(poll_interval)

    def fetch_batch_outputs(
        self,
        batch_obj
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Download output_file_id and parse JSONL.
        Returns:
            outputs_by_custom_id: {custom_id: output_text}
            raw_by_custom_id: {custom_id: full_json_line}
        """
        output_file_id = getattr(batch_obj, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError("Batch has no output_file_id. Status may be non-completed or failed.")

        content = self.client.files.content(output_file_id).read().decode("utf-8")
        outputs: Dict[str, str] = {}
        raw: Dict[str, Any] = {}

        for line in content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            raw[cid] = obj

            body = (obj.get("response") or {}).get("body") or {}
            out_text = body.get("output_text")
            if out_text is None:
                out_text = body.get("output", "")
            outputs[cid] = out_text if isinstance(out_text, str) else str(out_text)

        return outputs, raw
    
    def batch_generate(
        self,
        prompts: List[str],
        cfg: Optional[GenerationConfig] = None,
        custom_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **overrides
    ):
        batch_id = self.submit_responses_batch(
            prompts,
            cfg=cfg,
            custom_ids=custom_ids,
            metadata=metadata,
            **overrides
        )
        batch_obj = self.poll_batch_until_done(batch_id, poll_interval=cfg.poll_interval, timeout=cfg.timeout)
        return self.fetch_batch_outputs(batch_obj)