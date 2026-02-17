"""Live wake word test with massive 50x Digital Gain."""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pyaudio
import time
from openwakeword.model import Model

# Built-in model name
model_name = "hey_jarvis"
m = Model(wakeword_models=[model_name], inference_framework="onnx")

pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=2, rate=16000, input=True,
                 input_device_index=0, frames_per_buffer=1280)

print(f"\n🎤 TESTING WITH 50x GAIN!")
print("Please say 'Hey Jarvis' loudly now.")
print("Press Ctrl+C to stop.\n")

last_print = 0
try:
    while True:
        raw = stream.read(1280, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.int16)[::2]
        
        # MASSIVE 50x GAIN
        audio_boosted = (audio.astype(np.float32) * 50.0).clip(-32768, 32767).astype(np.int16)
        
        rms_orig = np.sqrt(np.mean(audio.astype(np.float32)**2))
        rms_boosted = np.sqrt(np.mean(audio_boosted.astype(np.float32)**2))
        
        preds = m.predict(audio_boosted)
        score = preds.get(model_name, 0)
        
        now = time.time()
        if now - last_print > 0.5:
            bar = "█" * int(min(rms_boosted / 200, 40))
            print(f"  OrigRMS:{rms_orig:5.1f} | BoostRMS:{rms_boosted:7.1f} | Score: {score:.4f} | {bar}")
            last_print = now
            
        if score >= 0.15:
             print(f"\n✨ WAKE WORD DETECTED WITH GAIN! Score: {score:.4f}")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
