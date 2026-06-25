from __future__ import annotations

"""
Silero VAD provider — uses silero-vad v6 API (load_silero_vad + VADIterator).

VADIterator already implements the silence-counting state machine and returns
{'start': N} / {'end': N} dicts, so we just translate those to VADEvents.

Barge-in: when agent_speaking is set and a 'start' event fires, we emit
BARGE_IN instead of SPEECH_START.  'end' events during agent speech are
suppressed (we don't want a spurious endpoint while the agent is talking).
"""

import asyncio
import logging
from typing import AsyncIterator

import numpy as np
import torch

from interfaces.vad import VADEvent, VADEventType, VADProvider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512   # 32 ms @ 16 kHz — required by silero-vad


class SileroVAD(VADProvider):
    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 400,
        speech_pad_ms: int = 100,
    ) -> None:
        self._threshold = threshold
        self._silence_ms = silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._model = None
        self._vad_iter = None   # current iterator — reset after each agent turn

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from silero_vad import load_silero_vad
        logger.info("Loading Silero VAD model …")
        self._model = load_silero_vad()
        logger.info("Silero VAD model ready")

    def _make_iterator(self):
        from silero_vad import VADIterator
        return VADIterator(
            model=self._model,
            threshold=self._threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=self._silence_ms,
            speech_pad_ms=self._speech_pad_ms,
        )

    def _run_chunk(self, vad_iter, window_bytes: bytes):
        """Run one 512-sample window through VADIterator. Returns dict or None."""
        arr = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(arr)
        return vad_iter(tensor)

    def reset(self) -> None:
        """Reset VADIterator state — call after each agent turn so the iterator
        starts fresh and doesn't carry over state from the agent's audio bleed."""
        self._vad_iter = self._make_iterator() if self._model is not None else None
        logger.debug("VADIterator reset")

    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        agent_speaking: asyncio.Event,
    ) -> AsyncIterator[VADEvent]:
        loop = asyncio.get_running_loop()
        self._load_model()
        self._vad_iter = self._make_iterator()

        buffer = b""
        prev_agent_speaking = agent_speaking.is_set()

        async for chunk in audio_chunks:
            buffer += chunk

            # Reset iterator when agent finishes speaking — clears stale speech
            # state from bleed audio so next SPEECH_START fires cleanly.
            cur_agent_speaking = agent_speaking.is_set()
            if prev_agent_speaking and not cur_agent_speaking:
                self._vad_iter = self._make_iterator()
                buffer = b""
                logger.debug("VADIterator reset on agent-done transition")
            prev_agent_speaking = cur_agent_speaking

            # Process in 512-sample (1024-byte) windows
            while len(buffer) >= CHUNK_SAMPLES * 2:
                window = buffer[: CHUNK_SAMPLES * 2]
                buffer = buffer[CHUNK_SAMPLES * 2 :]

                result = await loop.run_in_executor(
                    None, self._run_chunk, self._vad_iter, window
                )

                if result is None:
                    continue

                if "start" in result:
                    if agent_speaking.is_set():
                        yield VADEvent(type=VADEventType.BARGE_IN, confidence=1.0)
                    else:
                        yield VADEvent(type=VADEventType.SPEECH_START, confidence=1.0)

                elif "end" in result:
                    if not agent_speaking.is_set():
                        yield VADEvent(type=VADEventType.SPEECH_END, confidence=1.0)

    async def aclose(self) -> None:
        self._model = None
