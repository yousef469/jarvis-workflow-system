"""Comprehensive wake word diagnostic tool with AGC and multi-model support."""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pyaudio
import time
import soxr
from openwakeword.model import Model

models = ["hey_jarvis", "alexa", "hey_mycroft"]
print(f"Loading models: {models}")
m = Model(wakeword_models=models, inference_framework="onnx")

pa = pyaudio.PyAudio()
NATIVE_RATE = 44100
TARGET_RATE = 16000
CHANNELS = 2
CHUNK_NATIVE = 3528 

stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=NATIVE_RATE, input=True,
                 input_device_index=0, frames_per_buffer=CHUNK_NATIVE)

print(f"\n🎤 Say any: {models}")
print("Applying 2x Digital Gain. Press Ctrl+C to stop.\n")

last_print = 0
max_scores = {k: 0.0 for k in models}

try:
    while True:
        raw = stream.read(CHUNK_NATIVE, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.int16)[::2]
        
        # Apply Digital Gain (Boost 2x)
        audio_boosted = (audio.astype(np.float32) * 2.0).clip(-32768, 32767).astype(np.int16)
        
        # Resample
        audio_16k = soxr.resample(audio_boosted, NATIVE_RATE, TARGET_RATE).astype(np.int16)
        
        rms = np.sqrt(np.mean(audio_16k.astype(np.float32)**2))
        preds = m.predict(audio_16k)
        
        for k in models:
            score = preds.get(k, 0)
            max_scores[k] = max(max_scores[k], score)
            if score >= 0.15:
                print(f"✨ {k.upper()} triggered! Score: {score:.4f}")

        now = time.time()
        if now - last_print > 1.0:
            scores_str = " | ".join([f"{k}: {max_scores[k]:.4f}" for k in models])
            print(f"  RMS: {rms:7.1f} | MAX SINCE LAST: {scores_str}")
            # Reset max scores for next interval
            max_scores = {k: 0.0 for k in models}
            last_print = now

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
