from __future__ import annotations

"""
Entry point for the real-time voice agent.

Reads configuration from .env, wires up providers, and starts the pipeline.
All providers are swappable — change STT_PROVIDER / TTS_PROVIDER in .env.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Provider factories ────────────────────────────────────────────────────────

def build_vad():
    from providers.vad.silero import SileroVAD
    return SileroVAD(
        threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
        silence_ms=int(os.environ.get("VAD_SILENCE_MS", "400")),
        speech_pad_ms=int(os.environ.get("VAD_SPEECH_PAD_MS", "100")),
    )


def build_stt():
    provider = os.environ.get("STT_PROVIDER", "faster_whisper").lower()
    if provider == "faster_whisper":
        from providers.stt.faster_whisper import FasterWhisperSTT
        return FasterWhisperSTT(
            model_size=os.environ.get("WHISPER_MODEL", "tiny.en"),
            device=os.environ.get("WHISPER_DEVICE", "cpu"),
        )
    elif provider == "deepgram":
        from providers.stt.deepgram import DeepgramSTT
        return DeepgramSTT(api_key=os.environ.get("DEEPGRAM_API_KEY"))
    else:
        raise ValueError(f"Unknown STT_PROVIDER: {provider!r}. Choose: faster_whisper, deepgram")


def build_llm():
    from providers.llm.gemini import GeminiLLM
    return GeminiLLM(
        api_key=os.environ.get("GEMINI_API_KEY"),
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
    )


def build_tts():
    provider = os.environ.get("TTS_PROVIDER", "piper").lower()
    if provider == "piper":
        from providers.tts.piper import PiperTTS
        return PiperTTS(
            model_path=os.environ.get("PIPER_MODEL_PATH"),
            config_path=os.environ.get("PIPER_CONFIG_PATH"),
        )
    elif provider == "kokoro":
        from providers.tts.kokoro import KokoroTTS
        return KokoroTTS(
            model_path=os.environ.get("KOKORO_MODEL_PATH"),
        )
    elif provider == "elevenlabs":
        raise NotImplementedError("ElevenLabs provider not yet implemented")
    elif provider == "cartesia":
        raise NotImplementedError("Cartesia provider not yet implemented")
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {provider!r}. Choose: piper, kokoro")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    from core.instrumentation import LatencyLogger
    from core.server import EventServer
    from interviewer.interview_pipeline import InterviewPipeline

    logger.info("=== PrepAI — AI Mock Interviewer ===")
    logger.info("STT: %s | LLM: Gemini (%s) | TTS: %s",
                os.environ.get("STT_PROVIDER", "faster_whisper"),
                os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
                os.environ.get("TTS_PROVIDER", "piper"))

    # ── Frontend server ───────────────────────────────────────────────────────
    frontend_html = (Path(__file__).parent / "frontend" / "index.html").read_text(encoding="utf-8")
    host = os.environ.get("FRONTEND_HOST", "127.0.0.1")
    port = int(os.environ.get("FRONTEND_PORT", "8765"))
    event_server = EventServer(host=host, port=port)
    await event_server.start(frontend_html)

    logger.info("Open http://%s:%d in your browser, fill in the form, and click Start Interview.", host, port)
    logger.info("Press Ctrl+C to quit.\n")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    vad = build_vad()
    stt = build_stt()
    llm = build_llm()
    tts = build_tts()
    latency_logger = LatencyLogger(
        log_path=os.environ.get("LATENCY_LOG_PATH", "./logs/latency.jsonl")
    )

    mic_device_env = os.environ.get("MIC_DEVICE", "").strip()
    mic_device = int(mic_device_env) if mic_device_env else None

    pipeline = InterviewPipeline(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        latency_logger=latency_logger,
        sample_rate=int(os.environ.get("SAMPLE_RATE", "16000")),
        chunk_ms=int(os.environ.get("CHUNK_MS", "30")),
        mic_device=mic_device,
        system_prompt=(
            "You are Alex, a senior engineering interviewer. "
            "Speak naturally and concisely. No markdown or bullet points."
        ),
        max_history_turns=int(os.environ.get("MAX_HISTORY_TURNS", "20")),
        event_server=event_server,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
    )

    try:
        await pipeline.run()
    finally:
        await event_server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye.")
