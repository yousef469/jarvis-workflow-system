import wave
import numpy as np
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from config import WAKE_WORD_MODEL_PATH
from openwakeword.model import Model

def test_on_file():
    model_path = str(WAKE_WORD_MODEL_PATH / "hey_jarvis.onnx")
    wav_path = "backend/mic_test.wav"
    
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return

    # Load model
    model = Model(wakeword_models=[model_path], inference_framework="onnx")
    
    # Load WAV
    wf = wave.open(wav_path, 'rb')
    channels = wf.getnchannels()
    rate = wf.getframerate()
    width = wf.getsampwidth()
    
    print(f"Analyzing {wav_path}: {channels} channels, {rate} Hz, {width} bytes/sample")
    
    # Read chunk by chunk (1280 samples)
    CHUNK = 1280
    max_score = 0
    
    while True:
        data = wf.readframes(CHUNK)
        if not data:
            break
            
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # If stereo, take only the left channel
        if channels == 2:
            audio_data = audio_data[::2]
        
        # We might need to handle the case where the last chunk is smaller
        if len(audio_data) < CHUNK:
            padding = np.zeros(CHUNK - len(audio_data), dtype=np.int16)
            audio_data = np.concatenate((audio_data, padding))
            
        prediction = model.predict(audio_data)
        score = prediction.get('hey_jarvis', 0)
        
        if score > max_score:
            max_score = score
            
        if score > 0.05:
            print(f"Detection at {wf.tell()/rate:.2f}s: {score:.4f}")
            
    print(f"\nMax score detected in file: {max_score:.4f}")
    wf.close()

if __name__ == "__main__":
    test_on_file()
