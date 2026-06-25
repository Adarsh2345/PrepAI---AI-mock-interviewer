from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class TTSChunk:
    """A chunk of synthesized PCM audio (mono, 16-bit LE, provider's sample_rate)."""
    audio: bytes
    # True on the last chunk for this synthesis request
    is_final: bool = False
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())


class TTSProvider(ABC):
    """
    Contract: consume an async stream of text chunks (sentences or sentence
    fragments) and yield TTSChunks of PCM audio as soon as they are available.

    Implementations MUST start yielding audio before the full text is received
    (sentence-level pipelining).  They MUST honour the cancel event and stop
    synthesizing immediately when it is set.

    Output is raw PCM: mono, 16-bit signed little-endian, at sample_rate Hz.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output PCM sample rate."""

    @abstractmethod
    async def synthesize(
        self,
        text_chunks: AsyncIterator[str],
        cancel: asyncio.Event,
    ) -> AsyncIterator[TTSChunk]:
        """
        Yield TTSChunks as audio is synthesized.
        The last chunk must have is_final=True.
        """
        # pragma: no cover
        yield  # type: ignore[misc]

    async def aclose(self) -> None:
        """Optional cleanup (close model, release GPU memory, etc.)."""
