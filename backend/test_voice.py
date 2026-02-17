import sys
import os
import time

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from openvoice_engine_v2 import OpenVoiceEngine
    from RealtimeTTS import TextToAudioStream
    print("[TEST] Imports successful.")
except ImportError as e:
    print(f"[TEST] Import Failed: {e}")
    sys.exit(1)

def test_openvoice():
    print("[TEST] Initializing OpenVoice Engine...")
    try:
        engine = OpenVoiceEngine(
            checkpoint_path="jarvis-system/OpenVoice/checkpoints_v2/converter",
            voice_sample="Paul Bettany Breaks Down His Most Iconic Characters _ GQ-enhanced-v2.wav",
            device="cuda" # Fallback to CPU handled internally
        )
        stream = TextToAudioStream(engine)
        print("[TEST] Engine initialized.")
        
        text = "Hello sir. My voice protocols have been upgraded to OpenVoice version 2. I am ready to serve."
        print(f"[TEST] Synthesizing: '{text}'")
        
        # We can't easily capture audio output here, but if this runs without error, it works.
        stream.feed(text)
        stream.play_async()
        
        while stream.is_playing():
            time.sleep(0.1)
            
        print("[TEST] Synthesis complete.")
        
    except Exception as e:
        print(f"[TEST] FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openvoice()
