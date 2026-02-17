"""Real-time Whisper transcription test to verify audio quality."""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

import numpy as np
import pyaudio
import time
from whisper_cpp_engine import stt_engine

pa = pyaudio.PyAudio()
CHANNELS = 2
RATE = 16000 # Let's try 16kHz directly first to see if it works without soxr
CHUNK = 16000 # 1 second chunk

stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True,
                 input_device_index=0, frames_per_buffer=CHUNK)

print("\n🎤 Say something! I will transcribe 1-second chunks.")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.int16)[::2]
        
        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        
        # Convert to float32 for Whisper
        audio_f32 = audio.astype(np.float32) / 32768.0
        
        if rms > 150: # Only transcribe if there is sound
            print(f"  RMS: {rms:7.1f} | Transcribing...", end="", flush=True)
            text = stt_engine.transcribe(audio_f32)
            print(f" -> \"{text}\"")
        else:
            print(f"  RMS: {rms:7.1f} | [Silence]")
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
