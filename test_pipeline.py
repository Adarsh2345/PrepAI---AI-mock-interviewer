"""
Diagnostic test: speaks into the mic for 5 seconds and prints what each stage sees.
Run with: python test_pipeline.py

Checks in order:
  1. Mic audio is arriving (RMS levels printed every 0.5s)
  2. VAD fires SPEECH_START / SPEECH_END
  3. STT transcribes the captured audio
  4. Each stage prints a timestamped line so you can see exactly where it stalls
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pipeline")


async def test_mic():
    """Stage 1: confirm raw audio arrives from sounddevice."""
    import queue
    import threading
    import sounddevice as sd

    print("\n" + "="*60)
    print("STAGE 1: Microphone — speak now, 5 seconds")
    print("="*60)

    q: queue.Queue = queue.Queue()
    rms_values = []

    def cb(indata, frames, time_info, status):
        if status:
            print(f"  [sounddevice status] {status}")
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        q.put_nowait(pcm.tobytes())

    mic_dev = int(os.environ.get("MIC_DEVICE", "5"))
    stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=480, device=mic_dev, callback=cb)
    stream.start()

    deadline = time.monotonic() + 5.0
    last_print = time.monotonic()
    while time.monotonic() < deadline:
        try:
            chunk = q.get(timeout=0.1)
            arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(arr ** 2)))
            rms_values.append(rms)
            if time.monotonic() - last_print >= 0.5:
                bar = "#" * int(rms * 200)
                print(f"  Mic RMS: {rms:.4f}  |{bar}")
                last_print = time.monotonic()
        except queue.Empty:
            pass

    stream.stop()
    stream.close()

    max_rms = max(rms_values) if rms_values else 0
    avg_rms = sum(rms_values) / len(rms_values) if rms_values else 0
    print(f"\n  Max RMS: {max_rms:.4f}  Avg RMS: {avg_rms:.4f}")
    if max_rms < 0.01:
        print("  FAIL: No audio detected. Check mic permissions / default input device.")
        return False
    elif max_rms < 0.02:
        print("  WARN: Very quiet. Pipeline RMS gate (_MIN_RMS=0.02) may reject this.")
    else:
        print("  PASS: Mic audio OK")
    return True


async def test_vad():
    """Stage 2: confirm VAD fires SPEECH_START and SPEECH_END."""
    import queue as qlib
    import sounddevice as sd
    import torch
    from silero_vad import load_silero_vad, VADIterator

    print("\n" + "="*60)
    print("STAGE 2: VAD — speak a sentence, then stop")
    print("="*60)

    model = load_silero_vad()
    vad_iter = VADIterator(model, threshold=0.5, sampling_rate=16000,
                           min_silence_duration_ms=400, speech_pad_ms=100)

    q: qlib.Queue = qlib.Queue()

    def cb(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        q.put_nowait(pcm)

    stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=512, callback=cb)
    stream.start()

    events = []
    buffer = b""
    deadline = time.monotonic() + 8.0
    print("  Listening for VAD events for 8 seconds...")

    while time.monotonic() < deadline:
        try:
            chunk = q.get(timeout=0.1)
            buffer += chunk
            while len(buffer) >= 1024:
                window = buffer[:1024]
                buffer = buffer[1024:]
                arr = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
                result = vad_iter(torch.from_numpy(arr))
                if result:
                    ts = time.monotonic()
                    if "start" in result:
                        print(f"  PASS SPEECH_START at t={ts:.2f}")
                        events.append("start")
                    elif "end" in result:
                        print(f"  PASS SPEECH_END   at t={ts:.2f}")
                        events.append("end")
                    if "start" in events and "end" in events:
                        break
        except qlib.Empty:
            pass
        if "start" in events and "end" in events:
            break

    stream.stop()
    stream.close()

    if "start" not in events:
        print("  FAIL FAIL: No SPEECH_START — VAD never detected voice.")
        print("  → Possible causes: mic too quiet, wrong device, threshold too high")
        return False, b""
    if "end" not in events:
        print("  WARN  No SPEECH_END yet — try waiting longer after speaking")

    print("  PASS PASS: VAD events firing correctly")
    return True, b""


async def test_stt():
    """Stage 3: record 5s of speech and transcribe with faster-whisper."""
    import queue as qlib
    import sounddevice as sd

    print("\n" + "="*60)
    print("STAGE 3: STT — speak now (5 seconds recording)")
    print("="*60)

    q: qlib.Queue = qlib.Queue()

    def cb(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        q.put_nowait(pcm)

    mic_dev = int(os.environ.get("MIC_DEVICE", "5"))
    stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=480, device=mic_dev, callback=cb)
    stream.start()
    print("  Recording for 5 seconds...")

    deadline = time.monotonic() + 5.0
    chunks = []
    while time.monotonic() < deadline:
        try:
            chunk = q.get(timeout=0.1)
            chunks.append(chunk)
        except qlib.Empty:
            pass

    stream.stop()
    stream.close()

    audio = b"".join(chunks)
    rms = float(np.sqrt(np.mean(
        np.frombuffer(audio, dtype=np.int16).astype(np.float32) ** 2
    ))) / 32768.0
    print(f"  Recorded {len(audio)} bytes, RMS={rms:.4f}")

    if rms < 0.005:
        print("  FAIL FAIL: Audio too quiet — check mic")
        return False

    print("  Transcribing with faster-whisper...")
    model_size = os.environ.get("WHISPER_MODEL", "tiny.en")
    device = os.environ.get("WHISPER_DEVICE", "cpu")

    from faster_whisper import WhisperModel
    t0 = time.monotonic()
    wmodel = WhisperModel(model_size, device=device, compute_type="int8")
    arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = wmodel.transcribe(arr, language="en", beam_size=3,
                                       vad_filter=False,
                                       condition_on_previous_text=False,
                                       no_speech_threshold=0.6)
    parts = []
    for seg in segments:
        print(f"  Segment: {seg.text!r}  no_speech_prob={seg.no_speech_prob:.2f}")
        if seg.no_speech_prob <= 0.5:
            parts.append(seg.text.strip())

    elapsed = (time.monotonic() - t0) * 1000
    transcript = " ".join(parts).strip()
    print(f"\n  Transcript: {transcript!r}")
    print(f"  Time:       {elapsed:.0f} ms")

    if not transcript:
        print("  FAIL FAIL: Empty transcript — Whisper heard nothing or rejected as hallucination")
        return False

    print("  PASS PASS: STT working")
    return True


async def test_full_vad_stt():
    """Stage 4: full VAD-gated STT — exactly what the pipeline does."""
    import queue as qlib
    import sounddevice as sd
    import torch
    from silero_vad import load_silero_vad, VADIterator
    from faster_whisper import WhisperModel

    print("\n" + "="*60)
    print("STAGE 4: Full VAD→STT (pipeline simulation)")
    print("Speak a sentence. Pipeline will capture it and transcribe.")
    print("="*60)

    vad_model = load_silero_vad()
    vad_iter = VADIterator(vad_model, threshold=0.5, sampling_rate=16000,
                           min_silence_duration_ms=400, speech_pad_ms=100)

    model_size = os.environ.get("WHISPER_MODEL", "tiny.en")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    stt_model = WhisperModel(model_size, device=device, compute_type="int8")

    q: qlib.Queue = qlib.Queue()

    def cb(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        q.put_nowait(pcm)

    stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                            blocksize=512, callback=cb)
    stream.start()

    speech_buffer = []
    speech_active = False
    vad_buffer = b""
    deadline = time.monotonic() + 15.0
    endpoint_fired = False

    print("  Waiting for speech... (up to 15 seconds)")

    while time.monotonic() < deadline and not endpoint_fired:
        try:
            chunk = q.get(timeout=0.05)
            if speech_active:
                speech_buffer.append(chunk)
            vad_buffer += chunk

            while len(vad_buffer) >= 1024:
                window = vad_buffer[:1024]
                vad_buffer = vad_buffer[1024:]
                arr = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
                result = vad_iter(torch.from_numpy(arr))
                if result:
                    if "start" in result:
                        print(f"  VAD: SPEECH_START")
                        speech_active = True
                        speech_buffer.clear()
                        # Re-add this window to buffer since speech started here
                        speech_buffer.append(window)
                    elif "end" in result:
                        print(f"  VAD: SPEECH_END — {len(speech_buffer)} chunks buffered")
                        endpoint_fired = True
                        break
        except qlib.Empty:
            pass

    stream.stop()
    stream.close()

    if not endpoint_fired:
        if not speech_active:
            print("  FAIL FAIL: VAD never detected speech start")
        else:
            print("  FAIL FAIL: VAD detected speech start but no endpoint (still speaking?)")
        return False

    audio = b"".join(speech_buffer)
    rms = float(np.sqrt(np.mean(
        np.frombuffer(audio, dtype=np.int16).astype(np.float32) ** 2
    ))) / 32768.0
    print(f"  Audio: {len(audio)} bytes, RMS={rms:.4f}")

    if rms < 0.02:
        print(f"  FAIL FAIL: RMS {rms:.4f} < 0.02 — pipeline would discard this as bleed")
        print("  → Speak louder or lower MIN_SPEECH_RMS in pipeline.py")
        return False

    print("  Transcribing...")
    t0 = time.monotonic()
    arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = stt_model.transcribe(arr, language="en", beam_size=3,
                                        vad_filter=False,
                                        condition_on_previous_text=False,
                                        no_speech_threshold=0.6)
    parts = []
    for seg in segments:
        print(f"  Segment: {seg.text!r}  no_speech_prob={seg.no_speech_prob:.2f}")
        if seg.no_speech_prob <= 0.5:
            parts.append(seg.text.strip())

    elapsed = (time.monotonic() - t0) * 1000
    transcript = " ".join(parts).strip()
    print(f"\n  Transcript : {transcript!r}")
    print(f"  STT time   : {elapsed:.0f} ms")

    if transcript:
        print("  PASS PASS: Full VAD→STT pipeline working")
        return True
    else:
        print("  FAIL FAIL: VAD fired but STT returned empty — Whisper rejected the audio")
        print("  → no_speech_prob too high, audio too short, or hallucination filter hit")
        return False


async def main():
    print("\nVoice Pipeline Diagnostic")
    print("Each stage must pass before the next is meaningful.\n")

    ok = await test_mic()
    if not ok:
        print("\n⛔ Mic failed — fix before continuing")
        return

    ok, _ = await test_vad()
    if not ok:
        print("\n⛔ VAD failed — fix before continuing")
        return

    ok = await test_stt()
    if not ok:
        print("\n⛔ STT failed — fix before continuing")
        return

    ok = await test_full_vad_stt()
    if ok:
        print("\nPASS All stages passing — pipeline should work")
    else:
        print("\n⛔ Full VAD→STT simulation failed — see above")


if __name__ == "__main__":
    asyncio.run(main())
