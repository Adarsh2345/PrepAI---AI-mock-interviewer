# PrepAI — AI Mock Interviewer

> A voice-first behavioral interview coach. Speak your answers out loud, get real-time spoken feedback from an AI interviewer, and receive a detailed scorecard at the end — all running locally with no cloud STT or TTS.

**Built from scratch in Python using raw `asyncio` — no voice agent frameworks.**

---

## Screenshots

| Setup | Live interview |
|---|---|
| ![Setup screen](image.png) | ![Interview in progress](interview.png) |

---

## Why this is interesting to build

Most "voice AI" projects glue together a few APIs with no thought for timing. The hard part here is making the conversation feel natural:

- **Barge-in** — if Alex is mid-sentence and you start talking, he stops *immediately*. This requires cancelling in-flight LLM streaming and TTS synthesis simultaneously, with precise event ordering to avoid race conditions.
- **Overlapping pipeline stages** — LLM starts generating before STT is done; TTS starts synthesising the first sentence before the LLM finishes. Each stage runs concurrently via `asyncio` with a sentence queue as the handoff point.
- **Echo suppression** — the microphone picks up Alex's own voice through the speaker. The VAD must stay blind during playback and through a cooldown period after it ends, or it triggers ghost turns on its own audio.
- **Async scoring** — each answer is scored by an LLM call in a thread executor while the next question is already being asked. No waiting.

---

## Architecture

```
Microphone (16 kHz PCM)
        │
        ├──► Silero VAD ──► SPEECH_START / SPEECH_END / BARGE_IN
        │         │
        │    BargeInController (state machine)
        │         │
        └──► STT buffer ──► faster-whisper ──► transcript
                                                    │
                                          ┌─────────▼──────────┐
                                          │   LLM (streaming)  │
                                          └─────────┬──────────┘
                                                    │ tokens
                                            sentence splitter
                                                    │ sentences
                                          ┌─────────▼──────────┐
                                          │     Piper TTS      │
                                          └─────────┬──────────┘
                                                    │ PCM chunks
                                          ┌─────────▼──────────┐
                                          │   AudioPlayer      │
                                          │  (drain thread)    │
                                          └────────────────────┘

Background — per answer, non-blocking:
  answer + question ──► LLM eval ──► JSON scores ──► WebSocket ──► browser
```

### Key engineering decisions

| Decision | Why |
|---|---|
| Raw `asyncio`, no framework | Barge-in requires cancelling LLM and TTS mid-flight with exact timing — every framework I looked at abstracts this in ways that make it impossible |
| Sentence-level TTS streaming | First audio out of the speaker happens after the first sentence (~1 sentence), not after the full LLM response |
| Dedicated `AudioPlayer` drain thread | `sounddevice.write()` blocks — calling it from the event loop via `run_in_executor` caused 6–7 s delays; a dedicated thread with a queue eliminates this entirely |
| `agent_speaking` held through cooldown | Clearing it immediately after playback caused the VAD to pick up speaker ring-out as user speech, creating infinite echo loops |
| VADIterator reset after each turn | Silero VAD is stateful — without a reset between turns, bleed audio from the speaker left the iterator in a mid-speech state that suppressed the next `SPEECH_START` |
| Async evaluator in thread executor | LLM scoring calls are synchronous; running them in `run_in_executor` keeps the event loop free so the next question starts immediately |

---

## Features

- **10-question behavioral interview** over voice with a real AI interviewer persona (Alex)
- **Paste any job description** — questions are tailored to the role
- **Barge-in** — interrupt Alex mid-sentence and he stops immediately
- **Per-answer scoring** across 4 dimensions: Communication, Structure, Content Depth, Relevance
- **Full scorecard** revealed at the end with per-question breakdown
- **Fully local** — STT (faster-whisper) and TTS (Piper) run on-device; only the LLM call is remote
- **Swappable providers** — every component is behind an interface; change STT/TTS/LLM with one env var

---

## Project structure

```
prepai/
├── main.py                      # Entry point — wires providers, starts server
├── .env.example                 # All config options with inline docs
│
├── interfaces/                  # Abstract contracts for every provider
│   ├── llm.py, stt.py, tts.py, vad.py
│
├── core/
│   ├── pipeline.py              # VoicePipeline — async orchestrator (~420 lines)
│   ├── barge_in.py              # BargeInController state machine
│   ├── audio_utils.py           # MicrophoneStream, AudioPlayer (drain thread)
│   ├── instrumentation.py       # Per-turn latency logging to JSONL
│   ├── pronunciation.py         # Pre-TTS text normalisation
│   └── stt_corrections.py       # Post-STT corrections (name mishearings etc.)
│
├── providers/
│   ├── vad/silero.py            # Silero VAD v5
│   ├── stt/faster_whisper.py    # faster-whisper (local, default)
│   ├── stt/deepgram.py          # Deepgram Nova-2 (cloud, swap-in)
│   ├── llm/gemini.py            # Streaming LLM provider
│   ├── tts/piper.py             # Piper TTS (local, default)
│   └── tts/kokoro.py            # Kokoro ONNX (local, higher quality)
│
└── interviewer/
    ├── interview_pipeline.py    # Extends VoicePipeline with session logic
    ├── session.py               # InterviewSession state machine
    ├── evaluator.py             # Async per-answer scorer with JSON extraction
    ├── prompts.py               # All LLM prompts centralised
    └── questions.py             # 40+ behavioral questions across 10 categories
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

# CPU-only PyTorch (avoids the 2 GB CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Optional providers (Deepgram STT, Kokoro TTS):
```bash
pip install -r requirements-optional.txt
```

### 2. Download the Piper voice model

```bash
mkdir -p models/piper
# Download en_US-lessac-medium.onnx and .onnx.json from:
# https://github.com/rhasspy/piper/releases
# Place both files in models/piper/
```

### 3. Configure

```bash
cp .env.example .env
# Set GEMINI_API_KEY — free tier at https://aistudio.google.com
```

### 4. Run

```bash
python main.py
# Open http://127.0.0.1:8765
```

---

## Swapping providers

Every provider implements an abstract interface in `interfaces/`. To swap:

```bash
# Lower-latency cloud STT
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your_key

# Higher quality local TTS
TTS_PROVIDER=kokoro
KOKORO_MODEL_PATH=./models/kokoro
```

To add a new provider: subclass the interface, implement the required methods, add a branch to the factory in `main.py`.

---

## Diagnostic scripts

```bash
python test_devices.py    # Measure RMS across mic devices — find the best one
python test_pipeline.py   # End-to-end VAD → STT test from mic
python test_scoring.py    # Fire 3 Q&A pairs through the evaluator, print scores
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No audio output | Check `TTS_PROVIDER` and model path in `.env` |
| Agent speaks over itself / echo loops | `VAD_SILENCE_MS` too low — set to `400`; run `test_devices.py` to confirm mic device |
| VAD fires on background noise | Raise `VAD_THRESHOLD` to `0.6`–`0.7` |
| All scores show 5 | LLM API key missing or invalid — check `.env` |
| High transcription latency | Switch to `WHISPER_MODEL=tiny.en` or `STT_PROVIDER=deepgram` |
| Mic returns silence | Run `test_devices.py` and set `MIC_DEVICE=<n>` in `.env` |
