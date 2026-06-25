from __future__ import annotations

"""
Async voice pipeline orchestrator.

Flow per turn:
  1. VAD detects SPEECH_START  → start buffering mic audio
  2. VAD detects SPEECH_END    → endpoint fires, send buffered audio to STT
  3. STT transcribes buffer    → transcript ready
  4. LLM streams tokens        → sentence splitter feeds TTS
  5. TTS synthesizes per sentence → AudioPlayer plays immediately
  6. Barge-in: VAD detects SPEECH_START while agent speaking → cancel LLM+TTS
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.audio_utils import AudioPlayer, MicrophoneStream
from core.barge_in import AgentState, BargeInController
from core.instrumentation import LatencyLogger, TurnLatency
from core.pronunciation import normalise as pronunciation_normalise
from core.stt_corrections import correct as stt_correct
from interfaces.llm import LLMProvider, Message, Role
from interfaces.stt import STTEventType, STTProvider
from interfaces.tts import TTSProvider
from interfaces.vad import VADEventType, VADProvider

if TYPE_CHECKING:
    from core.server import EventServer

logger = logging.getLogger(__name__)

# Minimum RMS amplitude (0–1 scale) for an audio buffer to be considered real
# speech rather than mic bleed from the speaker.  Bleed through a laptop mic
# is typically 5–15% of direct speech amplitude.  0.02 rejects near-silence
# and quiet bleed while accepting any real voice input.
_MIN_RMS = 0.02

_SENTENCE_ENDS = {".", "!", "?", "…"}


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_ENDS and len(buf.strip()) > 4:
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


class VoicePipeline:
    def __init__(
        self,
        vad: VADProvider,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        latency_logger: LatencyLogger,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        mic_device: Optional[int] = None,
        system_prompt: str = "You are a helpful voice assistant. Be concise.",
        max_history_turns: int = 10,
        event_server: Optional["EventServer"] = None,
    ) -> None:
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.latency_logger = latency_logger
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.mic_device = mic_device
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self._ev = event_server

        self.barge_in = BargeInController()
        self.player = AudioPlayer(sample_rate=tts.sample_rate)
        self._armed = asyncio.Event()

        self._history: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt)
        ]

        # VAD gets all mic audio continuously
        self._vad_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=1000)

        # Speech buffer: filled only while VAD says user is speaking
        self._speech_buffer: list[bytes] = []
        self._speech_active = False

        # Signals the main loop that an endpoint fired and buffer is ready
        self._utterance_ready: asyncio.Event = asyncio.Event()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _emit(self, event: dict) -> None:
        if self._ev is not None:
            await self._ev.broadcast(event)

    async def _mic_fanout(self, mic: MicrophoneStream) -> None:
        """Feed mic audio into VAD queue and, when speech is active, the speech buffer."""
        async for chunk in mic:
            await self._vad_q.put(chunk)
            if self._speech_active:
                self._speech_buffer.append(chunk)
        await self._vad_q.put(None)

    async def _queue_iter(self, q: asyncio.Queue):
        while True:
            item = await q.get()
            if item is None:
                return
            yield item

    # ── Command loop ──────────────────────────────────────────────────────────

    async def _command_loop(self) -> None:
        if self._ev is None:
            return
        while True:
            cmd = await self._ev.commands.get()
            if cmd.get("action") == "toggle_mic":
                if self._armed.is_set():
                    self._armed.clear()
                    self._speech_active = False
                    self._speech_buffer.clear()
                    await self._emit({"type": "state", "state": "IDLE"})
                    logger.info("Mic disarmed")
                else:
                    self._armed.set()
                    await self._emit({"type": "state", "state": "LISTENING"})
                    logger.info("Mic armed — listening")

    # ── VAD loop ──────────────────────────────────────────────────────────────

    async def _vad_loop(self) -> None:
        async for event in self.vad.stream(
            self._queue_iter(self._vad_q),
            self.barge_in.agent_speaking,
        ):
            if not self._armed.is_set():
                continue

            if event.type == VADEventType.SPEECH_START:
                if not self.barge_in.agent_speaking.is_set():
                    # Clear stale ready flag so each utterance starts fresh
                    self._utterance_ready.clear()
                    self._speech_buffer.clear()
                    self._speech_active = True
                    await self.barge_in.on_speech_start()
                    await self._emit({"type": "state", "state": "LISTENING"})
                    logger.debug("VAD: speech start")
                else:
                    logger.debug("VAD: speech start ignored — agent is speaking (bleed suppressed)")

            elif event.type == VADEventType.SPEECH_END:
                if self._speech_active and not self.barge_in.agent_speaking.is_set():
                    self._speech_active = False
                    await self.barge_in.on_endpoint()
                    self._utterance_ready.set()
                    logger.debug("VAD: endpoint — %d bytes buffered", sum(len(c) for c in self._speech_buffer))
                elif self._speech_active:
                    self._speech_active = False
                    self._speech_buffer.clear()
                    logger.debug("VAD: speech end discarded — agent speaking")

            elif event.type == VADEventType.BARGE_IN:
                # Only act if agent is actually speaking — ignore startup noise
                if self.barge_in.agent_speaking.is_set():
                    self._speech_buffer.clear()
                    self._speech_active = True
                    await self.barge_in.on_barge_in()
                    await self._emit({"type": "barge_in"})
                    await self._emit({"type": "state", "state": "LISTENING"})
                    logger.info("Barge-in detected")
                else:
                    logger.debug("VAD: BARGE_IN ignored — agent not speaking (state=%s)", self.barge_in.state.name)

    # ── STT ───────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio_buffer: list[bytes]) -> str:
        """Run STT on accumulated speech buffer, return transcript."""
        audio = b"".join(audio_buffer)
        if not audio:
            return ""

        async def _buf_iter():
            yield audio  # single chunk containing the full utterance

        result = ""
        async for event in self.stt.stream(_buf_iter()):
            if event.type == STTEventType.FINAL:
                result = event.text
                break
        return result

    # ── LLM + TTS turn ───────────────────────────────────────────────────────

    async def _llm_and_tts_turn(self, transcript: str, record: TurnLatency) -> None:
        self._history.append(Message(role=Role.USER, content=transcript))
        if len(self._history) > self.max_history_turns * 2 + 1:
            self._history = self._history[:1] + self._history[-(self.max_history_turns * 2):]

        sentence_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=50)

        async def llm_stage() -> None:
            full_response = ""
            pending = ""
            first_token = True

            logger.info("Calling LLM (cancel_llm=%s)…", self.barge_in.cancel_llm.is_set())
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
            chunk_count = 0
            try:
                async for tts_chunk in self.tts.synthesize(sentence_iter(), self.barge_in.cancel_tts):
                    if self.barge_in.cancel_tts.is_set():
                        await self.player.stop()
                        break
                    if first_audio:
                        record.t_tts_first_audio = time.monotonic()
                        first_audio = False
                        logger.info("TTS first audio — writing to player")
                    chunk_count += 1
                    await self.player.write(tts_chunk.audio)
                    if record.t_playback_start is None:
                        record.t_playback_start = time.monotonic()
                    if tts_chunk.is_final:
                        record.t_playback_end = time.monotonic()
            except Exception as e:
                logger.error("TTS stage error: %s", e, exc_info=True)
            logger.info("TTS stage done — %d chunks written", chunk_count)

        await self.barge_in.on_agent_start_speaking()
        try:
            await asyncio.gather(llm_stage(), tts_stage())
            # Wait for all queued audio to finish playing before transitioning.
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

    # ── Warmup ────────────────────────────────────────────────────────────────

    async def _warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await self._emit({"type": "state", "state": "LOADING"})
        logger.info("Loading VAD model…")
        await loop.run_in_executor(None, self.vad._load_model)
        logger.info("Loading STT model…")
        await loop.run_in_executor(None, self.stt._load_model)
        if hasattr(self.tts, "_load_model"):
            logger.info("Loading TTS model…")
            await loop.run_in_executor(None, self.tts._load_model)
        elif hasattr(self.tts, "_get_proc"):
            logger.info("Starting TTS process…")
            await self.tts._get_proc()
        logger.info("All models ready.")

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.player.start()
        await self._warmup()
        await self._emit({"type": "state", "state": "IDLE"})
        logger.info("Pipeline ready — click the orb to start.")

        async with MicrophoneStream(sample_rate=self.sample_rate, chunk_ms=self.chunk_ms, device=self.mic_device) as mic:
            fanout_task = asyncio.create_task(self._mic_fanout(mic))
            vad_task = asyncio.create_task(self._vad_loop())
            cmd_task = asyncio.create_task(self._command_loop())

            try:
                while True:
                    # Wait for VAD to signal an endpoint
                    await self._utterance_ready.wait()
                    self._utterance_ready.clear()

                    # Snapshot the buffer — VAD may start filling a new one immediately
                    audio_snapshot = list(self._speech_buffer)
                    self._speech_buffer.clear()

                    if not audio_snapshot:
                        logger.debug("Empty audio buffer — skipping")
                        await self.barge_in.reset_for_next_turn()
                        continue

                    # Reject low-energy buffers — speaker bleed through the mic
                    # is much quieter than direct speech and causes ghost turns.
                    raw = b"".join(audio_snapshot)
                    rms = float(np.sqrt(np.mean(
                        np.frombuffer(raw, dtype=np.int16).astype(np.float32) ** 2
                    ))) / 32768.0
                    if rms < _MIN_RMS:
                        logger.debug("Audio RMS %.4f below threshold — discarding (likely bleed)", rms)
                        await self.barge_in.reset_for_next_turn()
                        continue

                    record = self.latency_logger.new_turn()
                    record.t_endpoint = time.monotonic()

                    # Transcribe
                    logger.info("Transcribing utterance (%d bytes)…", sum(len(c) for c in audio_snapshot))
                    transcript = await self._transcribe(audio_snapshot)
                    record.t_stt_final = time.monotonic()

                    if not transcript or not transcript.strip():
                        logger.info("Empty transcript — skipping turn")
                        await self.barge_in.reset_for_next_turn()
                        await self._emit({"type": "state", "state": "LISTENING"})
                        continue

                    transcript = stt_correct(transcript)
                    record.transcript = transcript
                    logger.info("Transcript: %r  (STT: %.0f ms)", transcript,
                                (record.t_stt_final - record.t_endpoint) * 1000)
                    await self._emit({"type": "transcript", "role": "user", "text": transcript, "partial": False})

                    # Clear any stale cancel flags before calling LLM
                    await self.barge_in.reset_for_next_turn()

                    # LLM + TTS
                    try:
                        await self._llm_and_tts_turn(transcript, record)
                    except asyncio.CancelledError:
                        record.barge_in = True
                        raise

                    if self.barge_in.state == AgentState.INTERRUPTED:
                        record.barge_in = True

                    self.latency_logger.commit(record)
                    await self._emit({
                        "type": "latency",
                        "turn_id": record.turn_id,
                        "barge_in": record.barge_in,
                        "stt_lag_ms": record.stt_lag_ms,
                        "endpoint_to_llm_first_ms": record.endpoint_to_llm_first_ms,
                        "llm_first_to_tts_first_ms": record.llm_first_to_tts_first_ms,
                        "tts_first_to_playback_ms": record.tts_first_to_playback_ms,
                        "total_response_latency_ms": record.total_response_latency_ms,
                    })

                    # Reset state machine so next turn starts clean
                    await self.barge_in.reset_for_next_turn()
                    self._speech_buffer.clear()
                    self._utterance_ready.clear()

            except KeyboardInterrupt:
                logger.info("Shutting down.")
            finally:
                fanout_task.cancel()
                vad_task.cancel()
                cmd_task.cancel()
                await self.player.close()
                await self.vad.aclose()
                await self.stt.aclose()
                await self.llm.aclose()
                await self.tts.aclose()
