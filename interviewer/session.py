from __future__ import annotations

"""
InterviewSession: drives the voice pipeline through a structured interview.

State machine:
  IDLE → INTRO → ASKING → LISTENING → EVALUATING → NEXT → SUMMARY → DONE

The session monkey-patches the pipeline's system prompt and injects question
context on each turn. It hooks into the pipeline's post-turn callback to:
  1. Capture the candidate's answer transcript.
  2. Kick off async evaluation (non-blocking, runs while next question is asked).
  3. Advance to the next question or trigger summary.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from interviewer.evaluator import Evaluator
from interviewer.prompts import interviewer_system, summary_prompt
from interviewer.questions import pick_questions

logger = logging.getLogger(__name__)


class SessionState(Enum):
    IDLE       = auto()
    INTRO      = auto()
    ASKING     = auto()
    LISTENING  = auto()
    EVALUATING = auto()
    SUMMARY    = auto()
    DONE       = auto()


@dataclass
class QuestionRecord:
    index: int
    question: str
    answer: str = ""
    score: Optional[dict] = None
    started_at: float = field(default_factory=time.monotonic)
    answered_at: Optional[float] = None


class InterviewSession:
    def __init__(
        self,
        role: str,
        jd: str,
        gemini_api_key: str,
        total_questions: int = 10,
        on_score_update: Optional[Callable[[int, dict], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
        on_summary_ready: Optional[Callable[[dict], None]] = None,
        model: str = "gemini-2.0-flash-lite",
    ) -> None:
        self.role = role
        self.jd = jd
        self.total_questions = total_questions
        self._on_score_update = on_score_update
        self._on_state_change = on_state_change
        self._on_summary_ready = on_summary_ready

        self.questions: list[str] = pick_questions(total_questions)
        self.records: list[QuestionRecord] = []
        self.scores: list[dict] = []

        self._state = SessionState.IDLE
        self._current_q_idx = 0
        self._evaluator = Evaluator(llm_api_key=gemini_api_key, model=model)
        self._eval_tasks: list[asyncio.Task] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def current_question_number(self) -> int:
        return self._current_q_idx + 1

    @property
    def is_done(self) -> bool:
        return self._state == SessionState.DONE

    def system_prompt(self) -> str:
        return interviewer_system(
            role=self.role,
            jd_snippet=self.jd,
            total_questions=self.total_questions,
        )

    def current_question_text(self) -> Optional[str]:
        if self._current_q_idx < len(self.questions):
            return self.questions[self._current_q_idx]
        return None

    def build_turn_context(self) -> str:
        """
        Builds the user-turn message the pipeline sends to the LLM.
        For INTRO: asks the agent to greet and ask Q1.
        For ASKING: gives the agent the next question to ask.
        For SUMMARY: asks the agent to summarise.
        """
        if self._state == SessionState.INTRO:
            q = self.questions[0]
            return (
                f"[SYSTEM: Start the interview. Greet the candidate warmly in 1-2 sentences, "
                f"then ask this first question:]\n\n{q}"
            )
        elif self._state == SessionState.ASKING:
            q = self.current_question_text()
            n = self.current_question_number
            return (
                f"[SYSTEM: Give 1-sentence feedback on their previous answer, "
                f"then say 'Let's move to question {n}.' and ask:]\n\n{q}"
            )
        elif self._state == SessionState.SUMMARY:
            return "[SYSTEM: The interview is complete. Deliver the final spoken verdict now.]"
        return ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._set_state(SessionState.INTRO)

    def on_answer_received(self, transcript: str) -> None:
        """Called by pipeline after the user finishes speaking."""
        if self._state not in (SessionState.LISTENING,):
            return

        q_idx = self._current_q_idx
        q_text = self.questions[q_idx]

        record = QuestionRecord(
            index=q_idx,
            question=q_text,
            answer=transcript,
            answered_at=time.monotonic(),
        )
        self.records.append(record)
        logger.info("Q%d answer captured (%d chars)", q_idx + 1, len(transcript))

        # Kick off async evaluation — don't block the pipeline waiting for it
        task = asyncio.create_task(self._evaluate_answer(record))
        self._eval_tasks.append(task)

        self._current_q_idx += 1

        if self._current_q_idx >= self.total_questions:
            self._set_state(SessionState.SUMMARY)
        else:
            self._set_state(SessionState.ASKING)

    def on_agent_spoke(self) -> None:
        """Called after the agent finishes its turn — now we're listening for user answer."""
        if self._state in (SessionState.INTRO, SessionState.ASKING):
            self._set_state(SessionState.LISTENING)
        elif self._state == SessionState.SUMMARY:
            self._set_state(SessionState.DONE)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_state(self, new: SessionState) -> None:
        old = self._state
        self._state = new
        logger.info("Session: %s → %s", old.name, new.name)
        if self._on_state_change:
            self._on_state_change(new.name)

    async def _evaluate_answer(self, record: QuestionRecord) -> None:
        try:
            score = await self._evaluator.evaluate(record.question, record.answer)
            record.score = score
            self.scores.append(score)
            logger.info("Q%d scored: overall=%s", record.index + 1, score.get("overall"))
            if self._on_score_update:
                self._on_score_update(record.index, score)
            # Fire summary once all answers are scored, regardless of current state
            # (state may not yet be SUMMARY if this evaluation finished very fast)
            if len(self.scores) == self.total_questions:
                logger.info("All %d answers scored — firing summary", self.total_questions)
                await self._fire_summary()
        except Exception as e:
            logger.error("Evaluation task error: %s", e, exc_info=True)

    async def _fire_summary(self) -> None:
        if self._on_summary_ready and self.records and self.scores:
            qa_pairs = [{"question": r.question, "answer": r.answer} for r in self.records]
            payload = {
                "qa_pairs": qa_pairs,
                "scores": self.scores,
                "averages": self._compute_averages(),
            }
            self._on_summary_ready(payload)

    def _compute_averages(self) -> dict:
        if not self.scores:
            return {}
        dims = ["communication", "structure", "content_depth", "relevance"]
        avgs: dict[str, float] = {}
        for d in dims:
            vals = [s[d]["score"] for s in self.scores if d in s and "score" in s[d]]
            avgs[d] = round(sum(vals) / len(vals), 1) if vals else 0.0
        overall = [s.get("overall", 0) for s in self.scores]
        avgs["overall"] = round(sum(overall) / len(overall), 1) if overall else 0.0
        return avgs

    def scorecard(self) -> dict:
        return {
            "role": self.role,
            "total_questions": self.total_questions,
            "answered": len(self.records),
            "scores": self.scores,
            "averages": self._compute_averages(),
            "records": [
                {
                    "index": r.index,
                    "question": r.question,
                    "answer": r.answer,
                    "score": r.score,
                }
                for r in self.records
            ],
        }
