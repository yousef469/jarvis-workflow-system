import os
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
from openvoice_engine_v2 import OpenVoiceEngine

def test_voice():
    print("\n--- JARVIS VOICE TEST (BRITISH FOUNDATION) ---")
    
    # 1. Setup paths
    project_root = "/Users/yousef/Desktop/live ai asistant"
    voice_sample = os.path.join(project_root, "Paul Bettany Breaks Down His Most Iconic Characters _ GQ-enhanced-v2.wav")
    checkpoint_path = os.path.join(project_root, "jarvis-system", "OpenVoice", "checkpoints_v2", "converter")
    
    print(f"Sample: {voice_sample}")
    print(f"Checkpoint: {checkpoint_path}")

    # 2. Initialize Engine
    engine = OpenVoiceEngine(
        checkpoint_path=checkpoint_path,
        voice_sample=voice_sample,
        speed=1.05  # Measured cadence
    )
    
    print("Loading models (Pre-warming)...")
    engine.preload_models()
    
    # 3. Text to Speak (Longer sentence to test chunking & phoneme stability)
    text = "I am functioning at optimized capacity, sir. I have updated my linguistic models and acoustic filters. Would you like me to assist you with any advanced protocols or system calibrations today?"
    print(f"Synthesizing: \"{text}\"")
    
    # 4. Synthesize (Manual process since we aren't using RealtimeTTS TextToAudioStream here)
    # We'll use the synthesize logic but capture the audio
    
    # OpenVoiceEngine.synthesize puts bytes into self.queue
    # We'll trigger it and then pull from the queue
    import threading
    
    def run_syn():
        engine.synthesize(text)
        engine.queue.put(None) # Sentinel to stop pulling
        
    thread = threading.Thread(target=run_syn)
    thread.start()
    
    audio_chunks = []
    print("Capturing output...")
    while True:
        item = engine.queue.get()
        if item is None:
            break
        # Convert bytes back to numpy float32
        chunk = np.frombuffer(item, dtype=np.float32)
        audio_chunks.append(chunk)
    
    if not audio_chunks:
        print("Error: No audio generated.")
        return

    full_audio = np.concatenate(audio_chunks)
    
    # 5. Playback
    print("Playing back audio...")
    sd.play(full_audio, 24000) # OpenVoice V2 is 24kHz
    sd.wait()
    
    # 6. Save for reference
    output_path = "jarvis_test_voice.wav"
    sf.write(output_path, full_audio, 24000)
    print(f"Test saved to: {os.path.abspath(output_path)}")
    print("--- TEST COMPLETE ---")

if __name__ == "__main__":
    try:
        test_voice()
    except Exception as e:
        print(f"Test failed error: {e}")
