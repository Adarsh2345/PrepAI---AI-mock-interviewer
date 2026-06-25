from __future__ import annotations

"""
InterviewPipeline: extends VoicePipeline with session-aware behaviour.

Overrides:
  - _command_loop: handles 'start_interview' command from frontend
  - _llm_and_tts_turn: injects session context into every LLM call
  - Post-turn hook: captures answers, fires scoring, advances questions
"""

import asyncio
import logging
import os
from typing import Optional

from core.instrumentation import LatencyLogger
from core.pipeline import VoicePipeline
from core.server import EventServer
from interfaces.llm import LLMProvider, Message, Role
from interfaces.stt import STTProvider
from interfaces.tts import TTSProvider
from interfaces.vad import VADProvider
from interviewer.session import InterviewSession, SessionState

logger = logging.getLogger(__name__)


class InterviewPipeline(VoicePipeline):
    def __init__(self, gemini_model: str = "gemini-2.0-flash-lite", **kwargs) -> None:
        super().__init__(**kwargs)
        self._session: Optional[InterviewSession] = None
        self._gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self._gemini_model = gemini_model

    # ── Command loop override ─────────────────────────────────────────────────

    async def _command_loop(self) -> None:
        if self._ev is None:
            return
        while True:
            cmd = await self._ev.commands.get()
            action = cmd.get("action")

            if action == "toggle_mic":
                if self._armed.is_set():
                    self._armed.clear()
                    self._speech_active = False
                    self._speech_buffer.clear()
                    await self._emit({"type": "state", "state": "IDLE"})
                else:
                    self._armed.set()
                    await self._emit({"type": "state", "state": "LISTENING"})

            elif action == "start_interview":
                role  = cmd.get("role", "Software Engineer")
                jd    = cmd.get("jd", "")
                name  = cmd.get("name", "Candidate")
                total = int(cmd.get("total_questions", 10))

                self._session = InterviewSession(
                    role=role,
                    jd=jd,
                    gemini_api_key=self._gemini_key,
                    total_questions=total,
                    on_score_update=self._on_score_update,
                    on_state_change=self._on_session_state_change,
                    on_summary_ready=self._on_summary_ready,
                    model=self._gemini_model,
                )
                self._session.start()

                # Push system prompt update and question list to pipeline
                self._history = [
                    Message(role=Role.SYSTEM, content=self._session.system_prompt())
                ]

                # Broadcast question list to UI
                await self._emit({
                    "type": "questions_list",
                    "questions": self._session.questions,
                })

                # Auto-arm mic and kick off the intro turn
                self._armed.set()
                await self._emit({"type": "state", "state": "LISTENING"})
                logger.info("Interview started: role=%r, questions=%d", role, total)

                # Trigger the intro immediately (no user speech needed to start)
                await self._trigger_agent_turn()

    async def _trigger_agent_turn(self) -> None:
        """Inject a synthetic turn to make the agent speak (intro / next question)."""
        if self._session is None:
            return
        context = self._session.build_turn_context()
        if not context:
            return

        # Emit the current question index to UI
        q_idx = self._session._current_q_idx
        if q_idx < len(self._session.questions) and self._session.state != SessionState.SUMMARY:
            await self._emit({
                "type": "question",
                "index": q_idx,
                "text": self._session.questions[q_idx],
            })
        elif self._session.state == SessionState.SUMMARY:
            await self._emit({"type": "session_state", "state": "SUMMARY"})

        from core.instrumentation import TurnLatency
        import time
        record = self.latency_logger.new_turn()
        record.t_endpoint = time.monotonic()
        record.transcript = "[agent-initiated]"

        # Push context as user message so LLM knows what to say
        self._history.append(Message(role=Role.USER, content=context))

        # Reset cancel flags only — don't change state machine here, let
        # on_agent_start_speaking() in _llm_and_tts_turn_raw handle the transition
        self.barge_in.cancel_llm.clear()
        self.barge_in.cancel_tts.clear()
        await self._llm_and_tts_turn_raw(record)

        # After agent speaks: mark session as listening for answer
        if self._session:
            self._session.on_agent_spoke()

        self.latency_logger.commit(record)

    # ── Override _llm_and_tts_turn to hook answer capture ────────────────────

    async def _llm_and_tts_turn(self, transcript: str, record) -> None:
        """Called after user speaks. Capture answer, then let agent respond."""
        if self._session and self._session.state == SessionState.LISTENING:
            self._session.on_answer_received(transcript)

        # Build combined user message: transcript + session directive as one USER turn.
        # This keeps the alternating user/assistant role requirement of the Gemini API.
        context = self._session.build_turn_context() if self._session else None
        if context:
            combined = f"{transcript}\n\n{context}"
        else:
            combined = transcript

        self._history.append(Message(role=Role.USER, content=combined))

        if len(self._history) > self.max_history_turns * 2 + 1:
            self._history = self._history[:1] + self._history[-(self.max_history_turns * 2):]

        # Emit question update to UI
        if self._session and self._session.state == SessionState.ASKING:
            q_idx = self._session._current_q_idx
            if q_idx < len(self._session.questions):
                await self._emit({
                    "type": "question",
                    "index": q_idx,
                    "text": self._session.questions[q_idx],
                })
        elif self._session and self._session.state == SessionState.SUMMARY:
            await self._emit({"type": "session_state", "state": "SUMMARY"})

        await self._llm_and_tts_turn_raw(record)

        if self._session:
            self._session.on_agent_spoke()
            if self._session.state == SessionState.DONE:
                await self._emit({"type": "session_state", "state": "DONE"})

    async def _llm_and_tts_turn_raw(self, record) -> None:
        """The actual LLM+TTS pipeline logic (from base class, inlined to avoid history double-append)."""
        import asyncio
        import time
        from typing import Optional as Opt

        sentence_q: asyncio.Queue[Opt[str]] = asyncio.Queue(maxsize=50)

        from core.pipeline import _split_sentences
        from core.pronunciation import normalise as pronunciation_normalise

        async def llm_stage() -> None:
            full_response = ""
            pending = ""
            first_token = True

            async for llm_event in self.llm.stream(list(self._history), self.barge_in.cancel_llm):
                if self.barge_in.cancel_llm.is_set():
                    break
                token = llm_event.token
                if first_token:
                    record.t_llm_first_token = time.monotonic()
                    first_token = False
                    await self._emit({"type": "state", "state": "AGENT_SPEAKING"})

                full_response += token
                pending += token

                await self._emit({"type": "transcript", "role": "agent", "text": full_response, "partial": True})

                sentences = _split_sentences(pending)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        if s:
                            await sentence_q.put(pronunciation_normalise(s))
                    pending = sentences[-1]

                if llm_event.is_final:
                    if pending.strip():
                        await sentence_q.put(pronunciation_normalise(pending.strip()))
                    break

            await sentence_q.put(None)

            if full_response and not self.barge_in.cancel_llm.is_set():
                self._history.append(Message(role=Role.ASSISTANT, content=full_response))
                record.response_preview = full_response[:80]
                await self._emit({"type": "transcript", "role": "agent", "text": full_response, "partial": False})

        async def tts_stage() -> None:
            async def sentence_iter():
                while True:
                    s = await sentence_q.get()
                    if s is None:
                        return
                    yield s

            first_audio = True
            try:
                async for tts_chunk in self.tts.synthesize(sentence_iter(), self.barge_in.cancel_tts):
                    if self.barge_in.cancel_tts.is_set():
                        await self.player.stop()
                        break
                    if first_audio:
                        record.t_tts_first_audio = time.monotonic()
                        first_audio = False
                    await self.player.write(tts_chunk.audio)
                    if record.t_playback_start is None:
                        record.t_playback_start = time.monotonic()
                    if tts_chunk.is_final:
                        record.t_playback_end = time.monotonic()
            except Exception as e:
                logger.error("TTS stage error: %s", e, exc_info=True)

        await self.barge_in.on_agent_start_speaking()
        try:
            await asyncio.gather(llm_stage(), tts_stage())
            # Wait for the drain thread to finish playing all queued audio before
            # transitioning to LISTENING — without this the state flips mid-sentence.
            if not self.barge_in.cancel_tts.is_set():
                await self.player.drain()
        finally:
            await self.barge_in.on_agent_done_speaking()
            self._speech_buffer.clear()
            self._utterance_ready.clear()
            await asyncio.sleep(0.4)
            self._speech_buffer.clear()
            self._utterance_ready.clear()
            if hasattr(self.vad, "reset"):
                self.vad.reset()
            await self.barge_in.on_agent_cooldown_done()
            await self._emit({"type": "state", "state": "LISTENING"})

    # ── Session callbacks (called from background eval tasks) ─────────────────

    def _on_score_update(self, question_index: int, score: dict) -> None:
        # Called from _evaluate_answer (asyncio Task on the event loop) — use ensure_future directly
        asyncio.ensure_future(self._emit({
            "type": "score_update",
            "question_index": question_index,
            "score": score,
        }))

    def _on_session_state_change(self, state_name: str) -> None:
        asyncio.ensure_future(self._emit({
            "type": "session_state",
            "state": state_name,
        }))

    def _on_summary_ready(self, payload: dict) -> None:
        asyncio.ensure_future(self._emit({
            "type": "summary",
            "averages": payload.get("averages", {}),
            "scores": payload.get("scores", []),
            "text": self._generate_summary_text(payload),
        }))

    def _generate_summary_text(self, payload: dict) -> str:
        avgs = payload.get("averages", {})
        overall = avgs.get("overall", 0)
        if overall >= 8:
            verdict = "strong pass"
        elif overall >= 6:
            verdict = "pass"
        elif overall >= 4:
            verdict = "borderline"
        else:
            verdict = "not ready"
        return (
            f"Overall score: {overall}/10 — {verdict}. "
            f"Communication: {avgs.get('communication', '?')}, "
            f"Structure: {avgs.get('structure', '?')}, "
            f"Content depth: {avgs.get('content_depth', '?')}."
        )
