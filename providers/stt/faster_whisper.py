from __future__ import annotations

"""
STT provider: faster-whisper (local, CPU/CUDA).

Strategy for streaming latency:
  faster-whisper is not a true streaming model — it transcribes full audio
  segments.  We use a sliding buffer approach:
    1. Accumulate audio while VAD says speech is active.
    2. On endpoint (the STT queue is closed from above or we detect a pause
       internally via energy), transcribe the full utterance buffer.
    3. Emit a single FINAL event.

  This gives us one RTT for transcription after the endpoint fires.
  On tiny.en / CPU the transcription of a 5-second utterance takes ~300 ms;
  on GPU it's ~50 ms.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import numpy as np

from interfaces.stt import STTEvent, STTEventType, STTProvider

logger = logging.getLogger(__name__)


class FasterWhisperSTT(STTProvider):
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 3,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._model = None

    @property
    def sample_rate(self) -> int:
        return 16000

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model %r on %s …", self._model_size, self._device)
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("faster-whisper model ready")

    def _preprocess(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert PCM to float32, normalize, and high-pass filter."""
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # High-pass filter: remove DC offset and low-frequency rumble
        if len(arr) > 1:
            arr = arr - np.mean(arr)

        # Normalize to use full dynamic range (helps with quiet microphones)
        peak = np.max(np.abs(arr))
        if peak > 0.01:
            arr = arr / peak * 0.95

        return arr

    # Common Whisper hallucinations on silence/noise
    _HALLUCINATIONS = {
        "you", "thank you", "thank you.", "thanks for watching.",
        "thanks for watching", "bye.", "bye", ".",  "...", "goodbye.",
        "please subscribe.", "subtitles by", "we'll see you next time.",
        "i'll see you in the next one.", "see you next time.",
    }

    def _transcribe(self, pcm_bytes: bytes) -> str:
        arr = self._preprocess(pcm_bytes)
        segments, info = self._model.transcribe(
            arr,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        parts = []
        for seg in segments:
            # Skip segment if model thinks it's likely not speech
            if seg.no_speech_prob > 0.5:
                logger.debug("Skipping segment (no_speech_prob=%.2f): %r", seg.no_speech_prob, seg.text)
                continue
            parts.append(seg.text.strip())

        text = " ".join(parts).strip()

        # Reject known hallucinations
        if text.lower() in self._HALLUCINATIONS:
            logger.debug("Rejecting hallucination: %r", text)
            return ""

        # Reject single characters or empty
        if len(text) < 2:
            return ""

        return text

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[STTEvent]:
        loop = asyncio.get_running_loop()
        self._load_model()

        # Accumulate the full utterance while audio arrives
        buffer = b""
        async for chunk in audio_chunks:
            buffer += chunk

        if not buffer:
            return

        logger.debug("Transcribing %d bytes of audio …", len(buffer))
        t0 = time.monotonic()
        text = await loop.run_in_executor(None, self._transcribe, buffer)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("Transcription took %.1f ms: %r", elapsed_ms, text)

        if text:
            yield STTEvent(type=STTEventType.FINAL, text=text)

    async def aclose(self) -> None:
        self._model = None
