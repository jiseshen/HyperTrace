import json
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
from data import Turn, UserData
from model import EmbedConfig, GenerationConfig
from model.batch_queue_model import BatchQueueModel
from tqdm import tqdm

from .branch import branch_hypotheses
from .consolidate import consolidate_hypotheses
from .filter import weight_hypothesis
from .hypothesis_set import HypothesisSet
from .initialize import initialize_hypothesis
from .perturb import perturb_hypotheses
from .preprocess import preprocess_candidates
from .response import generate_adapted_response
from .summary import summarize_hypotheses, summarize_profile
from .utils import TracerConfig, TracerContext


class Records(TypedDict, total=False):
    user: str
    turns: list[dict]
    general_profile: Optional[str]
    final_profile: str
    stopped_early: bool
    stop_reason: str


Phase = Literal[
    "start_turn",
    "adapted",
    "preprocess",
    "update",
    "filter",
    "summary",
    "perturb",
    "finalize_turn",
    "done",
]


@dataclass
class _TraceState:
    user_data: UserData
    context: TracerContext
    records: Records
    phase: Phase = "start_turn"
    working_profile: str = ""
    conversation_idx: int = 0
    turn_idx: int = 0
    initialized: bool = False
    conversation_history: List[Turn] = field(default_factory=list)
    turn_record: Dict[str, Any] = field(default_factory=dict)
    candidates_with_choice: Optional[str] = None
    candidates_for_filter: Optional[str] = None


@dataclass
class _WorkEstimate:
    estimated_flushes: int
    estimated_rounds: int
    estimated_users: int
    max_flushes_per_turn: int


class BatchPreferenceTracer:
    """
    Active-user-pool scheduler with explicit flush barriers:
    - each user advances independently through a turn state machine
    - model requests are globally queued
    - flush only when all running workers are blocked on queued model calls
    """

    def __init__(
        self,
        tracer_cfg: TracerConfig,
        model: BatchQueueModel,
        generation_cfg: GenerationConfig,
        embed_cfg: EmbedConfig,
        stage_workers: int = 64,
    ):
        self.model = model
        self.base_generation_config = generation_cfg
        self.embed_config = embed_cfg
        self.tracer_config = tracer_cfg
        self.stage_workers = max(1, stage_workers)
        self._scheduler_tick_seconds = 0.1
        self._active_pool_stop_ratio = 0.1
        self._stop_requested = False

    def trace_users(self, users: List[UserData]) -> Dict[str, Records]:
        states = [self._new_state(user) for user in users]
        ready = deque([state for state in states if state.phase != "done"])
        self._stop_requested = False
        stop_threshold = self._active_stop_threshold(len(states))
        estimate = self._estimate_work(users=users, stop_threshold=stop_threshold)
        pbar = tqdm(
            total=max(estimate.estimated_flushes, 1),
            desc="Tracing preferences (batch)",
            unit="flush",
            dynamic_ncols=True,
        )
        pbar.set_postfix(
            active=len(ready),
            wait=0,
            pending=0,
            max_turn_flushes=estimate.max_flushes_per_turn,
        )

        try:
            with ThreadPoolExecutor(max_workers=self.stage_workers) as executor:
                running: Dict[Future[None], _TraceState] = {}

                while ready or running:
                    active_count = self._count_active(states)
                    if not self._stop_requested and active_count < stop_threshold:
                        self._stop_requested = True
                        for state in ready:
                            self._mark_stopped_early(state)
                        ready.clear()

                    while ready and len(running) < self.stage_workers:
                        state = ready.popleft()
                        if state.phase == "done":
                            continue
                        future = executor.submit(self._advance_one_step, state)
                        running[future] = state

                    if not running:
                        if self.model.pending_count() > 0:
                            flushed = self.model.flush()
                            self._advance_progress(
                                pbar,
                                flushed=flushed,
                                active=self._count_active(states),
                            )
                        continue

                    done, _ = wait(
                        list(running.keys()),
                        timeout=self._scheduler_tick_seconds,
                        return_when=FIRST_COMPLETED,
                    )

                    if done:
                        for future in done:
                            state = running.pop(future)
                            future.result()
                            if self._stop_requested:
                                self._mark_stopped_early(state)
                            elif state.phase != "done":
                                ready.append(state)
                        self._refresh_progress(pbar, active=self._count_active(states))
                        continue

                    pending_count = self.model.pending_count()
                    waiting_count = self.model.waiting_call_count()
                    # Barrier: once all currently running workers are blocked in model calls,
                    # flush all queued requests together, regardless of how many each worker enqueued.
                    if pending_count > 0 and waiting_count >= len(running):
                        flushed = self.model.flush()
                        self._advance_progress(
                            pbar,
                            flushed=flushed,
                            active=self._count_active(states),
                        )
                    else:
                        self._refresh_progress(pbar, active=self._count_active(states))

                if self.model.pending_count() > 0:
                    flushed = self.model.flush()
                    self._advance_progress(
                        pbar,
                        flushed=flushed,
                        active=self._count_active(states),
                    )
        finally:
            pbar.close()

        return {state.user_data.user_id: state.records for state in states}

    def _new_state(self, user: UserData) -> _TraceState:
        hypothesis_set = HypothesisSet(
            n_hypotheses=self.tracer_config.n_hypotheses,
            embed_config=self.embed_config,
        )
        context = TracerContext(
            model=self.model,
            hypothesis_set=hypothesis_set,
            tracer_config=self.tracer_config,
            generation_config=self.base_generation_config,
        )
        records: Records = {
            "user": user.user_id,
            "turns": [],
            "general_profile": None,
        }
        return _TraceState(user_data=user, context=context, records=records)

    def _active_stop_threshold(self, total_users: int) -> int:
        if total_users <= 0:
            return 0
        return max(1, int(total_users * self._active_pool_stop_ratio))

    def _estimate_work(self, users: List[UserData], stop_threshold: int) -> _WorkEstimate:
        if not users:
            return _WorkEstimate(estimated_flushes=0, estimated_rounds=0, estimated_users=0, max_flushes_per_turn=0)
        # Once active users drop below stop_threshold, tracing stops.
        estimated_users = max(1, len(users) - stop_threshold + 1)
        # Empirical average used for ETA only.
        max_flushes_per_turn = 9
        max_flushes_per_conversation_end = 1

        turn_counts = [sum(len(conv.turns) for conv in u.conversations) for u in users]
        p90_turn_cap = max(1, int(np.quantile(turn_counts, 0.9)))
        # Global-round estimate:
        # assume users advance in synchronous rounds, capped at p90 turns.
        estimated_rounds = p90_turn_cap
        estimated_conversations = sum(len(u.conversations) for u in users[:estimated_users])
        estimated_flushes = (
            estimated_rounds * max_flushes_per_turn
            + estimated_conversations * max_flushes_per_conversation_end
        )
        return _WorkEstimate(
            estimated_flushes=estimated_flushes,
            estimated_rounds=estimated_rounds,
            estimated_users=estimated_users,
            max_flushes_per_turn=max_flushes_per_turn,
        )

    @staticmethod
    def _count_active(states: List[_TraceState]) -> int:
        return sum(1 for state in states if state.phase != "done")

    @staticmethod
    def _mark_stopped_early(state: _TraceState) -> None:
        if state.phase == "done":
            return
        state.phase = "done"
        state.records["stopped_early"] = True
        state.records["stop_reason"] = "active_user_pool_below_threshold"

    def _advance_progress(self, pbar: tqdm, flushed: int, active: int) -> None:
        if flushed > 0:
            if pbar.n + flushed > pbar.total:
                pbar.total = pbar.n + flushed
            pbar.update(flushed)
        self._refresh_progress(pbar, active=active)

    def _refresh_progress(self, pbar: tqdm, active: int) -> None:
        pbar.set_postfix(
            active=active,
            wait=self.model.waiting_call_count(),
            pending=self.model.pending_count(),
        )

    def _advance_one_step(self, state: _TraceState) -> None:
        self.model.set_request_context(user_id=state.user_data.user_id, phase=state.phase)
        if state.phase == "done":
            return
        if self._stop_requested:
            self._mark_stopped_early(state)
            return

        if state.phase == "start_turn":
            turn = self._current_turn(state)
            if turn is None:
                state.phase = "done"
                if state.context.current_belief is not None:
                    state.records["final_profile"] = summarize_profile(state.context)
                return
            state.conversation_history.append(turn)
            state.turn_record = {}
            state.candidates_with_choice = None
            state.candidates_for_filter = None
            state.phase = "adapted"
            return

        if state.phase == "adapted":
            state.turn_record["adapted"] = generate_adapted_response(
                conversation_history=state.conversation_history,
                profile=state.working_profile,
                context=state.context,
            )
            state.phase = "preprocess"
            return

        if state.phase == "preprocess":
            structured_candidates, preprocess_status = preprocess_candidates(
                state.conversation_history,
                state.context,
            )
            state.turn_record["preprocess"] = preprocess_status
            if not preprocess_status["success"] or preprocess_status["skip"]:
                state.phase = "finalize_turn"
                return

            state.candidates_with_choice = "[CandidateSet]\n" + json.dumps(
                structured_candidates,
                ensure_ascii=False,
                indent=2,
            )
            state.candidates_for_filter = "[CandidateSet]\n" + json.dumps(
                [
                    {
                        "i": item["i"],
                        "summary": item["summary"],
                        "content": item["content"],
                    }
                    for item in structured_candidates
                ],
                ensure_ascii=False,
                indent=2,
            )
            state.phase = "update"
            return

        if state.phase == "update":
            assert state.candidates_with_choice is not None
            if not state.initialized:
                init_status = initialize_hypothesis(
                    state.conversation_history,
                    state.candidates_with_choice,
                    state.context,
                )
                state.turn_record["initialize"] = init_status
                if not init_status.get("success", False):
                    state.phase = "finalize_turn"
                    return
                state.initialized = True
            else:
                branch_status = branch_hypotheses(
                    state.conversation_history,
                    state.candidates_with_choice,
                    state.context,
                )
                state.turn_record["branch"] = branch_status
            state.phase = "filter"
            return

        if state.phase == "filter":
            assert state.candidates_for_filter is not None
            state.turn_record["weight"] = weight_hypothesis(
                state.conversation_history,
                state.candidates_for_filter,
                state.context,
            )
            state.phase = "summary"
            return

        if state.phase == "summary":
            state.working_profile = summarize_hypotheses(state.context)
            state.phase = "perturb"
            return

        if state.phase == "perturb":
            assert state.candidates_with_choice is not None
            if (ess := state.context.belief.ess()) < self.tracer_config.n_hypotheses / 2:
                similar_groups = state.context.belief.resample()
            else:
                similar_groups = state.context.belief.get_similarity_groups(
                    threshold=self.tracer_config.similarity_threshold
                )
            perturb_status = perturb_hypotheses(
                state.conversation_history,
                state.candidates_with_choice,
                similar_groups,
                state.context,
            )
            perturb_status["ess"] = ess
            state.turn_record["perturb"] = perturb_status
            state.turn_record["hypotheses"] = state.context.belief.log_dict()
            state.turn_record["summary"] = state.working_profile
            state.phase = "finalize_turn"
            return

        if state.phase == "finalize_turn":
            self._finalize_turn(state)
            return

        raise RuntimeError(f"Unknown phase: {state.phase}")

    def _current_turn(self, state: _TraceState) -> Optional[Turn]:
        if state.conversation_idx >= len(state.user_data.conversations):
            return None
        conversation = state.user_data.conversations[state.conversation_idx]
        if state.turn_idx >= len(conversation.turns):
            return None
        return conversation.turns[state.turn_idx]

    def _finalize_turn(self, state: _TraceState) -> None:
        state.records["turns"].append(state.turn_record)

        conversation = state.user_data.conversations[state.conversation_idx]
        is_last_turn = state.turn_idx == len(conversation.turns) - 1

        if is_last_turn:
            if state.conversation_history and state.initialized:
                state.records["turns"][-1]["consolidate"] = consolidate_hypotheses(
                    state.conversation_history,
                    state.context,
                )
            state.conversation_idx += 1
            state.turn_idx = 0
            state.conversation_history = []
            state.initialized = False
        else:
            state.turn_idx += 1

        if self._current_turn(state) is None and state.conversation_idx >= len(state.user_data.conversations):
            state.phase = "done"
            if state.context.current_belief is not None:
                state.records["final_profile"] = summarize_profile(state.context)
        else:
            state.phase = "start_turn"
