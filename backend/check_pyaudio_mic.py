"""Check PyAudio microphones"""
import pyaudio

p = pyaudio.PyAudio()

print("\n--- PyAudio Microphones ---\n")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:  # Input device
        print(f"ID {i}: {info['name']}")
        print(f"       Channels: {info['maxInputChannels']}, Rate: {int(info['defaultSampleRate'])}")
        if i == p.get_default_input_device_info()['index']:
            print("       *** DEFAULT ***")
        print()

p.terminate()
