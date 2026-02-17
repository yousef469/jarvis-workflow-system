import pyaudio
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1280
RECORD_SECONDS = 3

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording 3 seconds... Speak into the mic.")
max_left = 0
max_right = 0

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    left = audio_data[::2]
    right = audio_data[1::2]
    
    max_left = max(max_left, np.max(np.abs(left)))
    max_right = max(max_right, np.max(np.abs(right)))

print(f"Max Left Channel: {max_left}")
print(f"Max Right Channel: {max_right}")

stream.stop_stream()
stream.close()
audio.terminate()
