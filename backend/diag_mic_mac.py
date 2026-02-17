import pyaudio
p = pyaudio.PyAudio()
print("Available Audio Devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels') > 0:
        print(f"Index {i}: {dev.get('name')} (Channels: {dev.get('maxInputChannels')}, Rate: {dev.get('defaultSampleRate')})")
p.terminate()
