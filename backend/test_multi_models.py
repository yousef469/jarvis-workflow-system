"""Test multiple wake word models simultaneously."""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pyaudio
import time
from openwakeword.model import Model

# Try to use built-in models which are usually more robust
models = ["hey_jarvis", "alexa", "hey_mycroft"]
print(f"Loading models: {models}")

m = Model(wakeword_models=models, inference_framework="onnx")
print(f"Models loaded! Keys: {list(m.prediction_buffer.keys())}")

pa = pyaudio.PyAudio()
# Stick to default mic for this test
MIC_INDEX = 0
stream = pa.open(format=pyaudio.paInt16, channels=2, rate=16000, input=True,
                 input_device_index=MIC_INDEX, frames_per_buffer=1280)

print(f"\n🎤 Say any of {models} now!")
print("Press Ctrl+C to stop.\n")

last_print = 0
try:
    while True:
        raw = stream.read(1280, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.int16)[::2] # Mono
        
        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        preds = m.predict(audio)
        
        now = time.time()
        if now - last_print > 0.5:
            scores = " | ".join([f"{k}: {preds.get(k, 0):.4f}" for k in models])
            bar = "█" * int(min(rms / 50, 40))
            print(f"  RMS: {rms:7.1f} | {scores} | {bar}")
            last_print = now
            
        for k in models:
            if preds.get(k, 0) >= 0.15:
                 print(f"\n✨ {k.upper()} DETECTED! Score: {preds.get(k, 0):.4f}")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
