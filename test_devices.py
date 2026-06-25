"""Find which audio device actually captures mic input. Run and speak continuously."""
import queue
import time
import numpy as np
import sounddevice as sd

INPUT_DEVICES = [0, 1, 4, 5, 9, 11, 12]

print("Speak continuously while this runs...\n")

for dev_id in INPUT_DEVICES:
    try:
        info = sd.query_devices(dev_id)
        name = info["name"][:45]
        q = queue.Queue()

        def cb(indata, frames, t, status, _q=q):
            _q.put_nowait((indata[:, 0] * 32767).astype("int16").tobytes())

        s = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                           blocksize=480, device=dev_id, callback=cb)
        s.start()
        time.sleep(1.2)
        s.stop()
        s.close()

        chunks = []
        while not q.empty():
            chunks.append(q.get_nowait())

        if chunks:
            arr = np.frombuffer(b"".join(chunks), dtype="int16").astype("float32") / 32768.0
            rms = float(np.sqrt(np.mean(arr ** 2)))
            peak = float(np.max(np.abs(arr)))
        else:
            rms = peak = 0.0

        flag = "  <-- USE THIS" if rms > 0.01 else ""
        print(f"  Device {dev_id:2d} | RMS={rms:.4f} peak={peak:.4f} | {name}{flag}")

    except Exception as e:
        print(f"  Device {dev_id:2d} | ERROR: {e}")

print("\nSet MIC_DEVICE=<id> in .env to use a specific device.")
