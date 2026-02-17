import pyaudio
import numpy as np
from openwakeword.model import Model

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1280
RECORD_SECONDS = 3

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Loading built-in 'hey_jarvis' model...")
oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("Recording 3 seconds... SAY 'HEY JARVIS' CLEARLY.")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Processing recording...")
full_audio = np.frombuffer(b"".join(frames), dtype=np.int16)

max_score = 0
for i in range(0, len(full_audio), CHUNK * 2):
    chunk = full_audio[i:i + CHUNK * 2]
    if len(chunk) < CHUNK * 2: continue
    mono = chunk[::2]
    pred = oww.predict(mono)
    score = pred.get("hey_jarvis", 0)
    if score > max_score: max_score = score

print(f"\nBuilt-in Diagnostic Result:")
print(f"Max Wake Word Score reached: {max_score:.4f}")
if max_score > 0.3:
    print("✅ SUCCESS: Built-in 'hey_jarvis' works!")
else:
    print("❌ FAILURE: Built-in model also failing. (Possible mic/system issue)")

stream.stop_stream()
stream.close()
audio.terminate()
