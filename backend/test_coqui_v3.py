import time
from jarvis_voice_v3 import get_voice_v3

def test_v3_integration():
    print("--- Testing JarvisVoiceV3 Coqui Integration ---")
    
    # 1. Initialize
    print("1. Initializing Engine...")
    voice = get_voice_v3()
    
    # Wait for loading check
    print("   Waiting for background model load...")
    while voice.is_loading:
        time.sleep(1)
    
    if voice.tts is None:
        print("❌ Coqui Init Failed")
        return

    print("✅ Engine Initialized")

    # 2. Test Cache Miss behavior
    test_text = "This is a unique test phrase delta 99."
    print(f"\n2. Testing Cache MISS: '{test_text}'")
    
    start = time.time()
    result = voice.speak(test_text)
    elapsed = time.time() - start
    
    if result is False:
        print(f"✅ Correct Behavior: speak() returned False immediately ({elapsed:.4f}s)")
        print("   Background generation should be running...")
    else:
        print(f"❌ Incorrect: speak() returned True on first run (Should affect cache only)")

    # 3. Wait for background gen
    print("\n3. Waiting for background generation (approx 45s)...")
    time.sleep(45) 
    
    if voice._is_cached(test_text):
        print("✅ Cache Verified: .npy file exists in memory")
    else:
        print("❌ Cache Failed: .npy key not found")
        return

    # 4. Test Cache HIT
    print(f"\n4. Testing Cache HIT: '{test_text}'")
    start = time.time()
    result = voice.speak(test_text, blocking=True)
    if result is True:
        print("✅ Correct Behavior: speak() returned True and played audio.")
    else:
        print("❌ Incorrect: speak() returned False on cached item.")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_v3_integration()
