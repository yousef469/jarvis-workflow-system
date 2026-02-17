import pyaudio
import numpy as np
import wave
import sys
import os

def record_and_analyze():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2 # Best for Mac Built-in
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "backend/mic_test.wav"

    p = pyaudio.PyAudio()
    
    print(f"\n--- RECORDING {RECORD_SECONDS} SECONDS ---")
    print("Please Speak 'Hey Jarvis' multiple times now!")
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        max_rms = 0

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Analyze volume
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            if rms > max_rms:
                max_rms = rms

        print("\n--- ANALYSIS ---")
        print(f"Max RMS observed: {max_rms:.2f}")
        
        if max_rms < 10:
            print("⚠️ WARNING: The mic seems completely silent. (Permissions?)")
        elif max_rms < 100:
            print("⚠️ WARNING: Very low volume detected.")
        else:
            print("✅ Mic is receiving sound!")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"Saved recording to: {WAVE_OUTPUT_FILENAME}")
        
    except Exception as e:
        print(f"❌ Error during recording: {e}")

if __name__ == "__main__":
    record_and_analyze()
