import pyaudio
import numpy as np
import os
import sys

def test_mic_capture():
    p = pyaudio.PyAudio()
    
    # Try different configurations
    configs = [
        {"rate": 44100, "channels": 2},
        {"rate": 44100, "channels": 1},
        {"rate": 16000, "channels": 2},
        {"rate": 16000, "channels": 1},
    ]
    
    print("\n--- MAC MIC CAPTURE TEST ---")
    
    for config in configs:
        rate = config["rate"]
        channels = config["channels"]
        print(f"\nTesting: {rate} Hz, {channels} Channels...")
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=1024
            )
            print("✅ SUCCESS: Stream opened!")
            
            # Try to read some data
            data = stream.read(1024, exception_on_overflow=False)
            print(f"✅ SUCCESS: Read {len(data)} bytes")
            
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"❌ FAILED: {e}")

    p.terminate()

if __name__ == "__main__":
    test_mic_capture()
