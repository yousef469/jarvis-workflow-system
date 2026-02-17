import sys
import os
import time

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from piper_engine import get_piper_engine

def test_voice():
    print("\n" + "="*50)
    print("🗣️ JARVIS VOICE TEST (Ryan High)")
    print("="*50 + "\n")
    
    voice = get_piper_engine()
    
    text = "Hey sir, how can I help you? I am Jarvis, your personal AI assistant."
    print(f"Speaking: '{text}'")
    
    voice.speak(text, blocking=True)
    print("\n✅ Verification complete.")

if __name__ == "__main__":
    test_voice()
