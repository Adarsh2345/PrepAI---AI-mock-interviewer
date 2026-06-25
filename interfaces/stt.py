from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class STTEventType(str, Enum):
    PARTIAL = "partial"   # interim transcript, may change
    FINAL = "final"       # committed transcript for this utterance


@dataclass
class STTEvent:
    type: STTEventType
    text: str
    confidence: float = 1.0
    # wall-clock timestamp (time.monotonic()) when this event was produced
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())


class STTProvider(ABC):
    """
    Contract: consume a stream of raw PCM audio chunks (bytes, mono 16-bit
    little-endian at the sample rate declared by sample_rate) and yield STTEvents.

    The provider MUST yield a FINAL event when it has committed a transcript.
    It SHOULD yield PARTIAL events as intermediate results.

    The caller closes the audio_stream (raises StopAsyncIteration) to signal
    end of session; the provider should flush any buffered audio and return.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Input sample rate this provider expects (typically 16000)."""

    @abstractmethod
    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[STTEvent]:
        """Yield STTEvents as audio arrives. Must be an async generator."""
        # pragma: no cover — ABC stub
        yield  # type: ignore[misc]

    async def aclose(self) -> None:
        """Optional cleanup (close websocket, release model, etc.)."""
