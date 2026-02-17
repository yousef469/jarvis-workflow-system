"""Quick standalone test for wake word detection (NO logger redirection)."""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))

from config import WAKE_WORD_SENSITIVITY, MIC_INDEX, WAKE_WORD_MODEL_PATH
import numpy as np

# 1. Check model file
model_file = str(WAKE_WORD_MODEL_PATH / "hey_jarvis.onnx")
print(f"Model path: {model_file}")
print(f"Model exists: {os.path.exists(model_file)}")
print(f"MIC_INDEX: {MIC_INDEX}")
print(f"Sensitivity: {WAKE_WORD_SENSITIVITY}")

# 2. List available mics
import pyaudio
pa = pyaudio.PyAudio()
print(f"\n--- Available Microphones ---")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']}, rate: {int(info['defaultSampleRate'])})")

# 3. Load wake word model
from openwakeword.model import Model
print(f"\nLoading model...")
oww = Model(wakeword_models=[model_file], inference_framework="onnx")
print(f"Model loaded! Keys: {list(oww.prediction_buffer.keys())}")

# 4. Listen for wake word
print(f"\n🎤 Say 'Hey Jarvis' now! (Mic: {MIC_INDEX}, Sensitivity: {WAKE_WORD_SENSITIVITY})")
print("Press Ctrl+C to stop.\n")

stream = pa.open(
    format=pyaudio.paInt16,
    channels=2,
    rate=16000,
    input=True,
    input_device_index=MIC_INDEX,
    frames_per_buffer=1280
)

import time
last_print = 0

try:
    while True:
        raw = stream.read(1280, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.int16)
        
        # RMS for volume meter
        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        
        # Take left channel only (2ch -> 1ch)
        mono = audio[::2]
        
        pred = oww.predict(mono)
        score = pred.get("hey_jarvis", 0)
        
        now = time.time()
        if now - last_print > 0.5:
            bar = "█" * int(min(rms / 50, 40))
            print(f"  RMS: {rms:7.1f} | Score: {score:.4f} | {bar}")
            last_print = now
        
        if score >= WAKE_WORD_SENSITIVITY:
            print(f"\n✨ WAKE WORD DETECTED! Score: {score:.4f}")
            break

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
