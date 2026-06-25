from __future__ import annotations

"""
STT provider: Deepgram Nova-2 (streaming WebSocket).

Unlike faster-whisper, Deepgram is a true streaming STT — it sends back
partial and final transcripts as audio arrives.  This is the preferred path
for lowest latency because STT final can arrive *before* or *simultaneously*
with the endpoint event, removing the STT step from the critical path entirely.

Free tier: 200 hours/month.
"""

import asyncio
import logging
import os
from typing import AsyncIterator

from interfaces.stt import STTEvent, STTEventType, STTProvider

logger = logging.getLogger(__name__)


class DeepgramSTT(STTProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-2",
        language: str = "en",
        endpointing_ms: int = 300,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self._model = model
        self._language = language
        self._endpointing_ms = endpointing_ms

    @property
    def sample_rate(self) -> int:
        return 16000

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[STTEvent]:
        from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

        dg = DeepgramClient(api_key=self._api_key)
        options = LiveOptions(
            model=self._model,
            language=self._language,
            sample_rate=self.sample_rate,
            channels=1,
            encoding="linear16",
            endpointing=self._endpointing_ms,
            interim_results=True,
            utterance_end_ms=1000,
        )

        result_q: asyncio.Queue[Optional[STTEvent]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        conn = dg.listen.asynclive.v("1")

        async def on_message(self_inner, result, **kwargs):
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                return
            if result.is_final:
                loop.call_soon_threadsafe(
                    result_q.put_nowait,
                    STTEvent(type=STTEventType.FINAL, text=text, confidence=alt.confidence),
                )
            else:
                loop.call_soon_threadsafe(
                    result_q.put_nowait,
                    STTEvent(type=STTEventType.PARTIAL, text=text),
                )

        conn.on(LiveTranscriptionEvents.Transcript, on_message)
        await conn.start(options)

        # Feed audio to Deepgram while consuming events
        async def feed():
            async for chunk in audio_chunks:
                await conn.send(chunk)
            await conn.finish()
            result_q.put_nowait(None)  # sentinel

        feed_task = asyncio.create_task(feed())

        try:
            while True:
                event = await result_q.get()
                if event is None:
                    break
                yield event
                if event.type == STTEventType.FINAL:
                    break
        finally:
            feed_task.cancel()
            try:
                await conn.finish()
            except Exception:
                pass

    async def aclose(self) -> None:
        pass
