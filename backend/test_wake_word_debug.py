import sys
import os
import time

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from wake_word import listen_for_wake_word
from config import WAKE_WORD_SENSITIVITY, MIC_INDEX

def test_wake_word():
    print("\n" + "="*50)
    print("🎙️ JARVIS WAKE WORD DEBUG TEST")
    print(f"Mic Index: {MIC_INDEX}")
    print(f"Sensitivity: {WAKE_WORD_SENSITIVITY}")
    print("="*50 + "\n")
    
    print("Speak 'Hey Jarvis' specifically to test.")
    print("I will show detection scores above 0.01 every 2 seconds.\n")
    
    try:
        detected = listen_for_wake_word()
        if detected:
            print("\n✅ SUCCESS: Wake word detected!")
        else:
            print("\n❌ FAILED: Wake word not detected or interrupted.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_wake_word()
