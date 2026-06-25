from __future__ import annotations

"""
LLM provider: Google Gemini via google-genai SDK (async streaming).
Migrated from deprecated google.generativeai to google.genai.
"""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from interfaces.llm import LLMEvent, LLMProvider, Message, Role

logger = logging.getLogger(__name__)

_ROLE_MAP = {
    Role.USER: "user",
    Role.ASSISTANT: "model",
}


class GeminiLLM(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.7,
        max_output_tokens: int = 512,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_contents(self, messages: list[Message]):
        """Split messages into system prompt + conversation contents."""
        from google.genai import types
        system_text = None
        contents = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_text = msg.content
                continue
            contents.append(
                types.Content(
                    role=_ROLE_MAP[msg.role],
                    parts=[types.Part(text=msg.content)],
                )
            )
        return system_text, contents

    async def stream(
        self,
        messages: list[Message],
        cancel: asyncio.Event,
    ) -> AsyncIterator[LLMEvent]:
        from google.genai import types

        client = self._get_client()
        system_text, contents = self._build_contents(messages)

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            system_instruction=system_text,
        )

        loop = asyncio.get_running_loop()
        chunk_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=200)

        def _sync_stream():
            try:
                response = client.models.generate_content_stream(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                for chunk in response:
                    if cancel.is_set():
                        break
                    if chunk.text:
                        loop.call_soon_threadsafe(chunk_q.put_nowait, chunk.text)
            except Exception as e:
                logger.error("Gemini stream error: %s", e)
            finally:
                loop.call_soon_threadsafe(chunk_q.put_nowait, None)

        stream_task = loop.run_in_executor(None, _sync_stream)

        try:
            while True:
                if cancel.is_set():
                    break
                token = await chunk_q.get()
                if token is None:
                    yield LLMEvent(token="", is_final=True)
                    break
                yield LLMEvent(token=token, is_final=False)
        finally:
            await stream_task

    async def aclose(self) -> None:
        self._client = None
