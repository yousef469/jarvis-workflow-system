"""Wake word test with high-quality soxr resampling."""
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

# Use built-in model name
model_name = "hey_jarvis"
print(f"Loading model: {model_name}")
m = Model(wakeword_models=[model_name], inference_framework="onnx")

pa = pyaudio.PyAudio()
# Native Mac rate
NATIVE_RATE = 44100
TARGET_RATE = 16000
CHANNELS = 2  # Mac built-in mic
CHUNK_NATIVE = 3528 # ~80ms at 44.1kHz (44100 * 0.08)

stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=NATIVE_RATE, input=True,
                 input_device_index=0, frames_per_buffer=CHUNK_NATIVE)

print(f"\n🎤 Say 'Hey Jarvis' now! (Native Rate: {NATIVE_RATE}Hz -> {TARGET_RATE}Hz)")
print("Using soxr for resampling. Press Ctrl+C to stop.\n")

last_print = 0
try:
    while True:
        raw = stream.read(CHUNK_NATIVE, exception_on_overflow=False)
        audio_native = np.frombuffer(raw, dtype=np.int16)
        
        # Take left channel (Stereo -> Mono)
        if CHANNELS == 2:
            audio_native = audio_native[::2]
            
        # Resample to 16000Hz using soxr
        audio_16k = soxr.resample(audio_native, NATIVE_RATE, TARGET_RATE)
        audio_16k = audio_16k.astype(np.int16)
        
        # Ensure we have exactly the right number of samples for OWW (which expects multiples of 1280)
        # Note: 80ms at 16kHz is 1280 samples.
        # 80ms at 44.1kHz is 3528 samples.
        
        rms = np.sqrt(np.mean(audio_16k.astype(np.float32)**2))
        preds = m.predict(audio_16k)
        score = preds.get(model_name, 0)
        
        now = time.time()
        if now - last_print > 0.4:
            bar = "█" * int(min(rms / 50, 40))
            print(f"  RMS: {rms:7.1f} | Score: {score:.4f} | {bar}")
            last_print = now
            
        if score >= 0.20:
             print(f"\n✨ WAKE WORD DETECTED! Score: {score:.4f}")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
