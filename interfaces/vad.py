from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class VADEventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"     # endpointing: user has stopped speaking
    BARGE_IN = "barge_in"         # speech detected while agent was speaking


@dataclass
class VADEvent:
    type: VADEventType
    # speech probability reported by the model (0.0–1.0)
    confidence: float = 1.0
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())


class VADProvider(ABC):
    """
    Contract: consume raw PCM audio chunks and emit VADEvents.

    The provider runs in two modes selected by agent_speaking:
      - False (listening mode): emit SPEECH_START / SPEECH_END for endpointing
      - True  (playback mode):  emit BARGE_IN when user speech is detected
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected input sample rate."""

    @abstractmethod
    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        agent_speaking: "asyncio.Event",  # noqa: F821
    ) -> AsyncIterator[VADEvent]:
        """Yield VADEvents as audio arrives."""
        # pragma: no cover
        yield  # type: ignore[misc]

    async def aclose(self) -> None:
        """Optional cleanup."""
