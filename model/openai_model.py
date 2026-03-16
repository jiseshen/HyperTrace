from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError
from .base import BaseLM, GenerationConfig, GenerationOverrides
from pydantic import BaseModel
from dataclasses import replace
from typing import Optional, Union, Tuple, List, Dict, Any, Unpack
import json
import time
import os
import asyncio

REASONING_PREFIXES = ("gpt-5", "o")
REASONING_BUDGETS = {"none": 0, "minimal": 64, "low": 128, "medium": 256, "high": 1024}


class OpenAIModel(BaseLM):
    def __init__(self, api_key: Optional[str] = None, default_cfg: Optional[GenerationConfig] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "empty")
        self.default_cfg = default_cfg or GenerationConfig()
        self.base_url = self.default_cfg.base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
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
        async def _one(prompt: str) -> Union[Dict[str, Any], Exception]:
            async with sem:
                retries = cfg.max_retries
                for attempt in range(retries):
                    try:
                        kwargs = self._build_responses_kwargs(prompt, cfg)
                        if schema:
                            resp = await self.async_client.responses.parse(**kwargs, text_format=schema)
                            output = self._normalize_parsed_output(resp.output_parsed)
                        else:
                            resp = await self.async_client.responses.create(**kwargs)
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
        **overrides: Unpack[GenerationOverrides]
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
        cfg = self._resolve_cfg(cfg, overrides)
        batch_id = self.submit_responses_batch(
            prompts,
            cfg=cfg,
            custom_ids=custom_ids,
            metadata=metadata,
            **overrides
        )
        batch_obj = self.poll_batch_until_done(batch_id, poll_interval=cfg.poll_interval, timeout=cfg.timeout)
        return self.fetch_batch_outputs(batch_obj)
