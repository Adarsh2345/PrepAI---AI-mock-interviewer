from __future__ import annotations

"""
TTS provider: Piper (local neural TTS).

Spawns one subprocess per sentence (simpler and more reliable than a persistent
process with timeout-based flushing). The process overhead is acceptable because
we overlap synthesis with LLM streaming — by the time the first sentence arrives
from the LLM, Piper is already running.

Output: 22050 Hz, mono, 16-bit signed LE PCM.
"""

import asyncio
import logging
import os
import sys
from typing import AsyncIterator, Optional

from interfaces.tts import TTSChunk, TTSProvider

logger = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22050


class PiperTTS(TTSProvider):
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        length_scale: float = 1.0,
    ) -> None:
        self._model_path = model_path or os.environ.get(
            "PIPER_MODEL_PATH", "./models/piper/en_US-lessac-medium.onnx"
        )
        self._config_path = config_path or os.environ.get(
            "PIPER_CONFIG_PATH",
            (self._model_path + ".json") if self._model_path else "",
        )
        self._length_scale = length_scale

    @property
    def sample_rate(self) -> int:
        return PIPER_SAMPLE_RATE

    def _build_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "piper",
            "--model", self._model_path,
            "--output_raw",
            "--length_scale", str(self._length_scale),
        ]
        if self._config_path and os.path.exists(self._config_path):
            cmd += ["--config", self._config_path]
        return cmd

    async def _synthesize_sentence(self, sentence: str) -> bytes:
        """Synthesize one sentence, return raw PCM bytes."""
        proc = await asyncio.create_subprocess_exec(
            *self._build_cmd(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=(sentence.strip() + "\n").encode())
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            logger.error("Piper error (rc=%d): %s", proc.returncode, err)
            return b""
        if not stdout:
            logger.warning("Piper returned empty audio for: %r", sentence)
        else:
            logger.debug("Piper synthesized %d bytes for: %r", len(stdout), sentence)
        return stdout

    async def _get_proc(self) -> None:
        """No-op warmup shim — kept so pipeline._warmup() can call it safely."""
        pass

    async def synthesize(
        self,
        text_chunks: AsyncIterator[str],
        cancel: asyncio.Event,
    ) -> AsyncIterator[TTSChunk]:
        pending_sentence: Optional[str] = None
        pending_pcm: Optional[bytes] = None

        async for sentence in text_chunks:
            if not sentence.strip():
                continue
            if cancel.is_set():
                return

            # Synthesize the previous sentence now (we know it's not the last)
            if pending_sentence is not None:
                pcm = await self._synthesize_sentence(pending_sentence)
                if pcm and not cancel.is_set():
                    yield TTSChunk(audio=pcm, is_final=False)

            pending_sentence = sentence

        if pending_sentence is not None and not cancel.is_set():
            pcm = await self._synthesize_sentence(pending_sentence)
            if pcm:
                yield TTSChunk(audio=pcm, is_final=True)

    async def aclose(self) -> None:
        pass
