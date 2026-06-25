from __future__ import annotations

"""
TTS provider: Kokoro (local, ONNX, higher quality than Piper).

Install: pip install kokoro-onnx
Models:  download from https://github.com/thewh1teagle/kokoro-onnx/releases
         and set KOKORO_MODEL_PATH to the directory containing kokoro.onnx + voices.bin

Output:  24000 Hz, mono, 16-bit signed LE PCM.
"""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

import numpy as np

from interfaces.tts import TTSChunk, TTSProvider

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000


class KokoroTTS(TTSProvider):
    def __init__(
        self,
        model_path: Optional[str] = None,
        voice: str = "af_heart",  # see kokoro-onnx docs for voice list
        speed: float = 1.0,
    ) -> None:
        self._model_path = model_path or os.environ.get(
            "KOKORO_MODEL_PATH", "./models/kokoro"
        )
        self._voice = voice
        self._speed = speed
        self._kokoro = None

    @property
    def sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE

    def _load_model(self) -> None:
        if self._kokoro is not None:
            return
        from kokoro_onnx import Kokoro
        model_file = os.path.join(self._model_path, "kokoro.onnx")
        voices_file = os.path.join(self._model_path, "voices.bin")
        self._kokoro = Kokoro(model_file, voices_file)
        logger.info("Kokoro TTS model loaded from %s", self._model_path)
        # Warm up: run a silent synthesis so ONNX JIT compilation happens now,
        # not on the first real turn where it would add ~1-2s to perceived latency.
        try:
            self._kokoro.create("Hello.", voice=self._voice, speed=self._speed, lang="en-us")
            logger.info("Kokoro warmup complete")
        except Exception as e:
            logger.warning("Kokoro warmup failed (non-fatal): %s", e)

    def _synthesize_sync(self, text: str) -> bytes:
        samples, rate = self._kokoro.create(text, voice=self._voice, speed=self._speed, lang="en-us")
        # samples is float32 [-1, 1] — convert to int16 PCM
        arr = (np.array(samples) * 32767).clip(-32768, 32767).astype(np.int16)
        return arr.tobytes()

    async def synthesize(
        self,
        text_chunks: AsyncIterator[str],
        cancel: asyncio.Event,
    ) -> AsyncIterator[TTSChunk]:
        loop = asyncio.get_running_loop()
        self._load_model()

        # Collect sentences into a list so we can peek at whether each is the last.
        # We synthesize eagerly as each sentence arrives from the LLM rather than
        # waiting for all text — but we need one sentence of lookahead to set is_final.
        pending_pcm: Optional[bytes] = None
        pending_sentence: Optional[str] = None

        async for sentence in text_chunks:
            if not sentence.strip():
                continue
            if cancel.is_set():
                return

            # Synthesize the previous sentence now (we know it's not final)
            if pending_sentence is not None:
                pcm = await loop.run_in_executor(None, self._synthesize_sync, pending_sentence)
                if pcm and not cancel.is_set():
                    yield TTSChunk(audio=pcm, is_final=False)

            pending_sentence = sentence

        # Yield the last sentence marked final
        if pending_sentence is not None and not cancel.is_set():
            pcm = await loop.run_in_executor(None, self._synthesize_sync, pending_sentence)
            if pcm:
                yield TTSChunk(audio=pcm, is_final=True)

    async def aclose(self) -> None:
        self._kokoro = None
