from __future__ import annotations

"""
Barge-in state machine.

States
──────
IDLE          → no user speech, agent silent
LISTENING     → user is (potentially) speaking, capturing audio for STT
AGENT_SPEAKING → agent's TTS audio is playing
INTERRUPTED   → user spoke over agent; pipeline cancelling in-flight work

Transitions
───────────
IDLE          --[VAD: SPEECH_START]--> LISTENING
LISTENING     --[VAD: SPEECH_END]----> IDLE (endpoint fires → triggers LLM)
AGENT_SPEAKING--[VAD: BARGE_IN]------> INTERRUPTED
INTERRUPTED   --[pipeline settled]---> LISTENING
LISTENING     --[agent starts TTS]--> AGENT_SPEAKING
AGENT_SPEAKING--[TTS finished]-------> IDLE
"""

import asyncio
import logging
import time
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = auto()
    LISTENING = auto()
    AGENT_SPEAKING = auto()
    INTERRUPTED = auto()


class BargeInController:
    """
    Thread-safe state machine + cancel signals for the voice pipeline.

    The pipeline checks `cancel_llm` and `cancel_tts` Events to abort in-flight
    work.  `agent_speaking` is an Event that VAD reads to switch detection mode.
    """

    def __init__(self) -> None:
        self._state = AgentState.IDLE
        self._lock = asyncio.Lock()

        # Signals consumed by pipeline stages
        self.cancel_llm: asyncio.Event = asyncio.Event()
        self.cancel_tts: asyncio.Event = asyncio.Event()
        self.agent_speaking: asyncio.Event = asyncio.Event()

        # Set by pipeline when an endpoint fires — pipeline awaits this
        self.endpoint_ready: asyncio.Event = asyncio.Event()

        self._state_change_callbacks: list = []

    @property
    def state(self) -> AgentState:
        return self._state

    def _set_state(self, new: AgentState) -> None:
        old = self._state
        self._state = new
        logger.debug("State: %s → %s", old.name, new.name)
        for cb in self._state_change_callbacks:
            cb(old, new)

    def on_state_change(self, callback) -> None:
        self._state_change_callbacks.append(callback)

    # ── Transitions called by VAD ─────────────────────────────────────────────

    async def on_speech_start(self) -> None:
        async with self._lock:
            if self._state == AgentState.IDLE:
                self._set_state(AgentState.LISTENING)

    async def on_endpoint(self) -> None:
        """User has stopped speaking — endpoint detected."""
        async with self._lock:
            if self._state == AgentState.LISTENING:
                self.endpoint_ready.set()
                # stays LISTENING until agent begins speaking

    async def on_barge_in(self) -> None:
        """User spoke while agent was speaking — trigger barge-in."""
        async with self._lock:
            if self._state == AgentState.AGENT_SPEAKING:
                logger.info("Barge-in detected — cancelling in-flight LLM + TTS")
                self._set_state(AgentState.INTERRUPTED)
                self.cancel_llm.set()
                self.cancel_tts.set()
                self.agent_speaking.clear()
            else:
                logger.debug("on_barge_in called but state=%s — ignoring", self._state.name)

    # ── Transitions called by pipeline ───────────────────────────────────────

    async def on_agent_start_speaking(self) -> None:
        async with self._lock:
            self._set_state(AgentState.AGENT_SPEAKING)
            self.agent_speaking.set()

    async def on_agent_done_speaking(self) -> None:
        """Audio finished playing — enter cooldown but keep agent_speaking set
        so VAD continues to suppress mic bleed from the speaker ringing out."""
        async with self._lock:
            if self._state == AgentState.AGENT_SPEAKING:
                self._set_state(AgentState.IDLE)
                # agent_speaking intentionally NOT cleared here — pipeline
                # calls on_agent_cooldown_done() after the sleep.

    async def on_agent_cooldown_done(self) -> None:
        """Called after the post-turn cooldown sleep — now safe to re-arm VAD."""
        async with self._lock:
            self.agent_speaking.clear()

    async def reset_for_next_turn(self) -> None:
        """Clear cancel signals and prepare for the next utterance."""
        async with self._lock:
            self.cancel_llm.clear()
            self.cancel_tts.clear()
            self.endpoint_ready.clear()
            self._set_state(AgentState.LISTENING)
