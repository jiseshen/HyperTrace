"""OpenAI Batch API orchestration for the shared command-line runner."""

from .batch_tracer import BatchPreferenceTracer
from model import GenerationConfig, load_model
from model.batch_queue_model import BatchQueueModel


def trace_batch(users, config, tracer_cfg, embed_cfg, prompts, workers):
    if not users:
        return {}
    gen_cfg = GenerationConfig(**config["main_model"])
    base_model = load_model(backend="openai", default_cfg=gen_cfg)
    queue_model = None
    try:
        queue_model = BatchQueueModel(
            base_model=base_model,
            completion_window=gen_cfg.completion_window,
            poll_interval=gen_cfg.poll_interval,
            timeout=gen_cfg.timeout,
            retry_delay=gen_cfg.retry_delay,
        )
        tracer = BatchPreferenceTracer(
            model=queue_model,
            generation_cfg=gen_cfg,
            tracer_cfg=tracer_cfg,
            embed_cfg=embed_cfg,
            stage_workers=workers,
            prompts=prompts,
        )
        return tracer.trace_users(users)
    finally:
        if queue_model is not None:
            queue_model.shutdown()
        base_model.close()
