from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator, Iterator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class MicrophoneStream:
    """
    Captures microphone audio and exposes it as an async generator of raw
    PCM bytes (mono, 16-bit signed little-endian).

    Uses a thread-safe queue.Queue as the bridge from the sounddevice callback
    thread to the async consumer — avoids asyncio.Queue.put_nowait being called
    across threads, which raises QueueFull inside the event loop with no way to
    catch it from the caller.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_ms / 1000)
        self.device = device
        self._thread_q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=500)
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Sounddevice input status: %s", status)
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        try:
            self._thread_q.put_nowait(pcm)
        except queue.Full:
            pass  # drop chunk rather than crash

    async def __aenter__(self) -> "MicrophoneStream":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_frames,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.debug("Microphone stream started (rate=%d, chunk=%d frames)", self.sample_rate, self.chunk_frames)
        return self

    async def __aexit__(self, *args) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._thread_q.put_nowait(None)  # sentinel

    async def __aiter__(self) -> AsyncIterator[bytes]:
        # Bridge: sounddevice callback → thread queue → asyncio queue → async consumer.
        # Using call_soon_threadsafe avoids blocking an executor thread on queue.get()
        # and delivers chunks to the event loop as soon as they arrive (~30 ms cadence).
        loop = asyncio.get_running_loop()
        async_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=500)

        def _bridge() -> None:
            while True:
                chunk = self._thread_q.get()
                loop.call_soon_threadsafe(async_q.put_nowait, chunk)
                if chunk is None:
                    return

        bridge_thread = threading.Thread(target=_bridge, daemon=True)
        bridge_thread.start()

        while True:
            chunk = await async_q.get()
            if chunk is None:
                return
            yield chunk


class AudioPlayer:
    """
    Plays PCM audio (mono, 16-bit LE) through the default output device.

    The drain loop runs in a dedicated background thread so that
    sounddevice.write() (a blocking call) never stalls the asyncio event loop.
    Chunks are passed via a thread-safe queue.Queue from async producers.
    """

    _STOP = object()  # sentinel: flush and keep running
    _CLOSE = object()  # sentinel: exit thread
    _DRAIN = object()  # sentinel: signal drain complete

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self._thread_q: queue.Queue = queue.Queue(maxsize=200)
        self._drain_event: threading.Event = threading.Event()
        self._stream: Optional[sd.OutputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._playing = asyncio.Event()

    async def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._drain_thread, daemon=True)
        self._thread.start()
        logger.debug("Audio player started (rate=%d)", self.sample_rate)

    def _drain_thread(self) -> None:
        """Blocking drain loop — runs entirely off the event loop."""
        while True:
            item = self._thread_q.get()
            if item is self._CLOSE:
                return
            if item is self._STOP:
                # Flush remaining items without playing them
                while True:
                    try:
                        nxt = self._thread_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is self._CLOSE:
                        return
                    if nxt is self._STOP:
                        break
                    # discard audio
                continue
            if item is self._DRAIN:
                # All queued audio has been played — signal the waiter
                self._drain_event.set()
                continue
            # item is raw PCM bytes
            if self._stream and self._stream.active:
                arr = np.frombuffer(item, dtype=np.int16)
                self._stream.write(arr)  # blocks until device consumes — fine in thread

    async def write(self, pcm: bytes) -> None:
        try:
            self._thread_q.put_nowait(pcm)
        except queue.Full:
            logger.warning("AudioPlayer queue full — dropping chunk")

    async def drain(self) -> None:
        """Wait until all queued audio has finished playing out of the speaker."""
        loop = asyncio.get_running_loop()
        self._drain_event.clear()
        self._thread_q.put_nowait(self._DRAIN)
        # Wait on the threading.Event from a thread so we don't block the loop
        await loop.run_in_executor(None, self._drain_event.wait)

    async def stop(self) -> None:
        """Immediately silence playback and flush the queue."""
        self._thread_q.put_nowait(self._STOP)
        self._playing.clear()
        logger.debug("Audio player flushed")

    async def close(self) -> None:
        self._thread_q.put(self._CLOSE)
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop()
            self._stream.close()

    @property
    def is_playing(self) -> bool:
        return not self._thread_q.empty()


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(arr: np.ndarray) -> bytes:
    return (arr * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
