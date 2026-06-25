from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class LLMEvent:
    """A single streamed chunk from the LLM."""
    token: str
    # True on the last chunk of a complete response
    is_final: bool = False
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())


class LLMProvider(ABC):
    """
    Contract: given a conversation history, stream response tokens one chunk at
    a time.  The cancel event is set externally (barge-in) to signal that the
    caller wants to abort; implementations MUST check it and stop yielding ASAP.

    Implementations should NOT buffer the full response before yielding — the
    pipeline depends on first-token latency.
    """

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        cancel: asyncio.Event,
    ) -> AsyncIterator[LLMEvent]:
        """
        Yield LLMEvent chunks until the response is complete or cancel is set.
        The final chunk must have is_final=True.
        """
        # pragma: no cover
        yield  # type: ignore[misc]

    async def aclose(self) -> None:
        """Optional cleanup."""
