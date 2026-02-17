import pyaudio
import numpy as np
import time

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1280
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()

print(f"Recording {RECORD_SECONDS} seconds for diagnostics...")
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []
max_amp = 0

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    curr_max = np.max(np.abs(audio_data))
    if curr_max > max_amp:
        max_amp = curr_max
    frames.append(audio_data)

print(f"Finished. Max Amplitude: {max_amp}")
if max_amp > 30000:
    print("⚠️ WARNING: Audio is CLIPPING or very close to it.")
else:
    print("✅ Audio levels seem okay.")

stream.stop_stream()
stream.close()
audio.terminate()
