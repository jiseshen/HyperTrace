import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Union, Unpack

from pydantic import BaseModel
from tqdm import tqdm

from .base import BaseLM, GenerationConfig, GenerationOverrides
from .openai_model import OpenAIModel

logger = logging.getLogger(__name__)


class BatchFlushError(RuntimeError):
    pass


@dataclass
class _QueuedRequest:
    custom_id: str
    prompt: str
    schema: Optional[type[BaseModel]]
    cfg: GenerationConfig
    event: threading.Event
    created_at: float
    user_id: Optional[str] = None
    phase: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class BatchQueueModel(BaseLM):
    """
    Queueing model wrapper with explicit flush barriers.

    Calls to generate/async_generate enqueue requests and block until
    a controller calls flush(). No periodic background flushing.
    """

    def __init__(
        self,
        base_model: OpenAIModel,
        *,
        max_batch_size: int = 1000,
        max_batch_retries: int = 1,
        poll_interval: float = 60.0,
        timeout: Optional[float] = None,
        completion_window: Optional[str] = None,
        retry_delay: float = 0.5,
        cleanup_remote_files: bool = True,
        cleanup_delete_timeout: float = 5.0,
        debug: bool = False,
        debug_track_user: Optional[str] = None,
    ):
        self.base_model = base_model
        self.max_batch_size = max_batch_size
        self.max_batch_retries = max_batch_retries
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.completion_window = completion_window
        self.retry_delay = retry_delay
        self.cleanup_remote_files = cleanup_remote_files
        self.cleanup_delete_timeout = cleanup_delete_timeout
        self.debug = debug
        self.debug_track_user = debug_track_user

        self._cv = threading.Condition()
        self._queue: List[_QueuedRequest] = []
        self._seq = 0
        self._fatal_error: Optional[Exception] = None
        self._waiting_call_count = 0
        self._debug_parse_preview_chars = 2000
        self._local = threading.local()
        self._debug_lock = threading.Lock()

    def set_request_context(self, *, user_id: Optional[str], phase: Optional[str] = None) -> None:
        self._local.user_id = user_id
        self._local.phase = phase

    @property
    def fatal_error(self) -> Optional[Exception]:
        return self._fatal_error

    def shutdown(self) -> None:
        with self._cv:
            pending = self._queue[:]
            self._queue.clear()
        if pending:
            err = RuntimeError("BatchQueueModel shutdown with pending requests.")
            for req in pending:
                req.error = err
                req.event.set()

    def pending_count(self) -> int:
        with self._cv:
            return len(self._queue)

    def waiting_call_count(self) -> int:
        # Number of blocked generate/async_generate call sites (worker-level),
        # not number of queued low-level requests.
        with self._cv:
            return self._waiting_call_count

    def flush(self) -> int:
        if self._fatal_error is not None:
            raise BatchFlushError(str(self._fatal_error))

        flushed_batches = 0
        while True:
            with self._cv:
                if not self._queue:
                    break
                take = min(len(self._queue), self.max_batch_size)
                to_flush = self._queue[:take]
                del self._queue[:take]
            self._flush_requests(to_flush)
            flushed_batches += 1

        if self._fatal_error is not None:
            raise BatchFlushError(str(self._fatal_error))
        return flushed_batches

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def generate(
        self,
        prompt: str,
        schema: Optional[type[BaseModel]] = None,
        cfg: Optional[GenerationConfig] = None,
        **overrides: Unpack[GenerationOverrides],
    ) -> Dict[str, Any]:
        if self._fatal_error is not None:
            raise BatchFlushError(str(self._fatal_error))
        req = self._enqueue(prompt=prompt, schema=schema, cfg=cfg, overrides=overrides)
        with self._cv:
            self._waiting_call_count += 1
        try:
            req.event.wait()
            if req.error is not None:
                raise req.error
            assert req.result is not None
            return req.result
        finally:
            with self._cv:
                self._waiting_call_count -= 1

    async def async_generate(
        self,
        prompts: List[str],
        schema: Optional[type[BaseModel]] = None,
        cfg: Optional[GenerationConfig] = None,
        concurrency: int = 5,
        return_exceptions: bool = True,
        **overrides: Unpack[GenerationOverrides],
    ) -> List[Union[Dict[str, Any], Exception]]:
        if self._fatal_error is not None:
            err = BatchFlushError(str(self._fatal_error))
            if return_exceptions:
                return [err for _ in prompts]
            raise err

        reqs = [self._enqueue(prompt=p, schema=schema, cfg=cfg, overrides=overrides) for p in prompts]

        async def wait_one(req: _QueuedRequest) -> Dict[str, Any]:
            await asyncio.to_thread(req.event.wait)
            if req.error is not None:
                raise req.error
            assert req.result is not None
            return req.result

        with self._cv:
            self._waiting_call_count += 1
        try:
            return await asyncio.gather(*(wait_one(req) for req in reqs), return_exceptions=return_exceptions)
        finally:
            with self._cv:
                self._waiting_call_count -= 1

    def _enqueue(
        self,
        *,
        prompt: str,
        schema: Optional[type[BaseModel]],
        cfg: Optional[GenerationConfig],
        overrides: Dict[str, Any],
    ) -> _QueuedRequest:
        resolved_cfg = self._resolve_cfg(cfg, overrides)
        user_id = getattr(self._local, "user_id", None)
        phase = getattr(self._local, "phase", None)
        with self._cv:
            self._seq += 1
            req = _QueuedRequest(
                custom_id=f"q-{self._seq:010d}",
                prompt=prompt,
                schema=schema,
                cfg=resolved_cfg,
                event=threading.Event(),
                created_at=time.time(),
                user_id=user_id,
                phase=phase,
            )
            self._queue.append(req)
            return req

    def _flush_requests(self, requests: List[_QueuedRequest]) -> None:
        outputs: Optional[Dict[str, str]] = None
        raw_outputs: Optional[Dict[str, Any]] = None
        final_error: Optional[Exception] = None
        batch_obj = None
        input_file_id: Optional[str] = None
        tracked_user = self.debug_track_user or (requests[0].user_id if requests else None)
        tracked_printed_401_hint = False

        tracked_req = next((r for r in requests if r.user_id == tracked_user), None)
        if self.debug and tracked_req is not None:
            preview_req = tracked_req
            self._debug_print(
                f"[batch-debug] submit size={len(requests)} track_user={tracked_user} "
                f"phase={preview_req.phase} in='{self._preview20(preview_req.prompt)}'"
            )

        attempt = 0
        max_attempts = max(1, self.max_batch_retries)
        while attempt < max_attempts:
            try:
                batch_id, input_file_id = self._submit_mixed_batch(requests)
                batch_obj = self._poll_batch_until_done(
                    batch_id,
                    poll_interval=self.poll_interval,
                    timeout=self.timeout,
                )
                outputs, raw_outputs = self._fetch_batch_outputs(batch_obj)
                final_error = None
                break
            except Exception as exc:
                final_error = exc
                attempt += 1
                is_batch_scope_error = self._is_batch_scope_permission_error(exc)
                if is_batch_scope_error:
                    max_attempts = max(max_attempts, 3)
                    if tracked_req is not None and not tracked_printed_401_hint:
                        self._debug_print(
                            f"[batch-debug] track_user={tracked_user} batch scope error, retrying "
                            f"{attempt}/{max_attempts} after {self.retry_delay:.1f}s"
                        )
                        tracked_printed_401_hint = True
                logger.exception(
                    "Batch queue flush attempt failed (%s/%s), size=%s",
                    attempt,
                    max_attempts,
                    len(requests),
                )
                if attempt < max_attempts:
                    if is_batch_scope_error:
                        time.sleep(min(self.retry_delay, 3.0))
                    else:
                        time.sleep(self.retry_delay)
            finally:
                if self.cleanup_remote_files:
                    if batch_obj is not None:
                        for fid in (
                            getattr(batch_obj, "input_file_id", None),
                            getattr(batch_obj, "output_file_id", None),
                            getattr(batch_obj, "error_file_id", None),
                        ):
                            self._safe_delete_file(fid)
                    elif input_file_id is not None:
                        self._safe_delete_file(input_file_id)
                batch_obj = None
                input_file_id = None

        if outputs is None:
            err = final_error or RuntimeError("Batch queue flush failed")
            self._fatal_error = err
            self._mark_failed_requests(requests, err)
            with self._cv:
                pending = self._queue[:]
                self._queue.clear()
            self._mark_failed_requests(pending, err)
            raise BatchFlushError(str(err))

        for req in requests:
            try:
                text = outputs.get(req.custom_id)
                if text is None:
                    raise RuntimeError(f"Missing output for custom_id={req.custom_id}")
                req.result = self._parse_output(text=text, schema=req.schema, cfg=req.cfg)
                if self.debug and tracked_req is not None and req.user_id == tracked_user:
                    self._debug_print(
                        f"[batch-debug] track_user={tracked_user} cid={req.custom_id} parse=ok "
                        f"out='{self._preview20(text)}'"
                    )
            except Exception as exc:
                if req.schema is not None:
                    raw_obj = (raw_outputs or {}).get(req.custom_id)
                    logger.error(
                        "Batch parse failed for custom_id=%s; parse_text_preview=%r; raw_response_preview=%r; err=%s",
                        req.custom_id,
                        self._clip_text(text if text is not None else ""),
                        self._clip_text(json.dumps(raw_obj, ensure_ascii=False) if raw_obj is not None else ""),
                        exc,
                    )
                if self.debug and tracked_req is not None and req.user_id == tracked_user:
                    self._debug_print(
                        f"[batch-debug] track_user={tracked_user} cid={req.custom_id} parse=err "
                        f"out='{self._preview20(text if text is not None else '')}' err={exc}"
                    )
                req.error = exc
            finally:
                req.event.set()

    @staticmethod
    def _mark_failed_requests(requests: List[_QueuedRequest], err: Exception) -> None:
        for req in requests:
            req.error = err
            req.event.set()

    def _submit_mixed_batch(self, requests: List[_QueuedRequest]) -> tuple[str, str]:
        lines = [
            json.dumps(
                self.base_model._build_batch_line(req.prompt, req.cfg, req.custom_id),
                ensure_ascii=False,
            )
            for req in requests
        ]
        payload = ("\n".join(lines) + "\n").encode("utf-8")

        in_file = self.base_model.client.files.create(
            file=("batch.jsonl", payload),
            purpose="batch",
        )

        completion_window = self.completion_window or requests[0].cfg.completion_window
        batch = self.base_model.client.batches.create(
            input_file_id=in_file.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
            metadata={
                "source": "batch_queue_model",
                "size": str(len(requests)),
            },
        )
        return batch.id, in_file.id

    def _safe_delete_file(self, file_id: Optional[str]) -> None:
        if not file_id:
            return
        try:
            # Keep cleanup best-effort and bounded; never block flush for long.
            self.base_model.client.with_options(timeout=self.cleanup_delete_timeout).files.delete(file_id)
        except Exception:
            pass

    def _poll_batch_until_done(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        timeout: Optional[float] = None,
    ):
        start = time.time()
        while True:
            batch = self.base_model.client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)
            if status in ("completed", "failed", "cancelled", "expired"):
                return batch
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Batch {batch_id} polling timed out after {timeout}s")
            time.sleep(poll_interval)

    def _fetch_batch_outputs(self, batch_obj) -> tuple[Dict[str, str], Dict[str, Any]]:
        output_file_id = getattr(batch_obj, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError("Batch has no output_file_id. Status may be non-completed or failed.")

        content = self.base_model.client.files.content(output_file_id).read().decode("utf-8")
        outputs: Dict[str, str] = {}
        raw: Dict[str, Any] = {}

        for line in content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            raw[cid] = obj

            body = (obj.get("response") or {}).get("body") or {}
            outputs[cid] = self._extract_output_text(body)

        return outputs, raw

    @staticmethod
    def _extract_output_text(body: Dict[str, Any]) -> str:
        direct = body.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct

        texts: List[str] = []

        def collect_from_content(content: Any) -> None:
            if not isinstance(content, list):
                return
            for block in content:
                if not isinstance(block, dict):
                    continue
                txt = block.get("text")
                if isinstance(txt, str) and txt:
                    texts.append(txt)

        output = body.get("output")
        if isinstance(output, str) and output.strip():
            return output
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                txt = item.get("text")
                if isinstance(txt, str) and txt:
                    texts.append(txt)
                collect_from_content(item.get("content"))

        collect_from_content(body.get("content"))

        if texts:
            return "\n".join(texts)
        return json.dumps(body, ensure_ascii=False)

    def _resolve_cfg(self, cfg: Optional[GenerationConfig], overrides: Dict[str, Any]) -> GenerationConfig:
        cfg = cfg or self.base_model.default_cfg
        if overrides:
            return replace(cfg, **overrides)
        return cfg

    def _parse_output(self, *, text: str, schema: Optional[type[BaseModel]], cfg: GenerationConfig) -> Dict[str, Any]:
        if schema is None:
            output: Any = text
        else:
            parsed = self._parse_json_text(text)
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Failed to parse batch output as JSON for schema validation.")
            output = schema.model_validate(parsed).model_dump()
        if cfg.reasoning_summary:
            return {"output": output, "reasoning": None}
        return {"output": output}

    def _parse_json_text(self, text: str) -> Optional[Any]:
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
        candidates = [i for i in (first_brace, first_bracket) if i >= 0]
        if not candidates:
            return None
        snippet = cleaned[min(candidates):]
        for end in range(len(snippet), 0, -1):
            chunk = snippet[:end].strip()
            if not chunk:
                continue
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
        return None

    def _clip_text(self, text: str) -> str:
        if len(text) <= self._debug_parse_preview_chars:
            return text
        return text[: self._debug_parse_preview_chars] + "...[truncated]"

    @staticmethod
    def _is_batch_scope_permission_error(exc: Exception) -> bool:
        text = str(exc)
        return "insufficient permissions" in text.lower() and "api.batch." in text.lower()

    @staticmethod
    def _preview20(text: str) -> str:
        clean = text.replace("\n", " ").strip()
        return clean[:20]

    def _debug_print(self, msg: str) -> None:
        if self.debug:
            with self._debug_lock:
                tqdm.write(msg)
