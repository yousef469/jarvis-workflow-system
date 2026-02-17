import pyaudio
import numpy as np
from openwakeword.model import Model
import os

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1280
RECORD_SECONDS = 3
MODEL_PATH = "/Users/yousef/Desktop/jarvis/jarvis/backend/models/openwakeword/hey_jarvis.onnx"

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print(f"Loading model: {MODEL_PATH}")
oww = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")

print("Recording 3 seconds... SAY 'HEY JARVIS' CLEARLY.")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Processing recording...")
full_audio = np.frombuffer(b"".join(frames), dtype=np.int16)

# Process in chunks of 1280
max_score = 0
for i in range(0, len(full_audio), CHUNK * 2): # * 2 because interleved
    chunk = full_audio[i:i + CHUNK * 2]
    if len(chunk) < CHUNK * 2: continue
    
    # Mono conversion
    mono = chunk[::2]
    
    # RMS check
    rms = np.sqrt(np.mean(mono.astype(np.float32)**2))
    
    # Prediction
    pred = oww.predict(mono)
    score = pred.get("hey_jarvis", 0)
    
    if score > max_score:
        max_score = score
    
    # print(f"Chunk RMS: {rms:7.1f} | Score: {score:.4f}")

print(f"\nFinal Diagnostic Result:")
print(f"Max Wake Word Score reached: {max_score:.4f}")
if max_score > 0.3:
    print("✅ SUCCESS: The model can hear you on this mic!")
else:
    print("❌ FAILURE: The model still doesn't recognize your voice.")

stream.stop_stream()
stream.close()
audio.terminate()
