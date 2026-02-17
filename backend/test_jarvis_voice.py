"""
Test JARVIS Voice Engine
Run this to verify Coqui TTS is working properly
"""

import time

print("=" * 60)
print("JARVIS Voice Engine Test")
print("=" * 60)

# Test 1: Import modules
print("\n[1] Testing imports...")
try:
    from jarvis_voice import JarvisVoice
    from jarvis_knowledge import JarvisKnowledgeBase, get_jarvis_response
    print("    ✓ Imports successful")
except ImportError as e:
    print(f"    ✗ Import failed: {e}")
    print("\n    Install required packages:")
    print("    pip install TTS sounddevice scipy")
    exit(1)

# Test 2: Knowledge base
print("\n[2] Testing knowledge base...")
test_queries = [
    "Who are you?",
    "Tell me about the Mark 50 suit",
    "What is the arc reactor?"
]

for query in test_queries:
    response = get_jarvis_response(query)
    if response:
        print(f"    Q: {query}")
        print(f"    A: {response[:80]}...")
    else:
        print(f"    ✗ No response for: {query}")

# Test 3: Voice synthesis
print("\n[3] Testing voice synthesis...")
try:
    jarvis = JarvisVoice()
    
    print("\n    Playing greeting...")
    jarvis.greet()
    time.sleep(0.5)
    
    print("\n    Playing acknowledgment...")
    jarvis.acknowledge()
    time.sleep(0.5)
    
    print("\n    Playing custom text...")
    jarvis.speak("All systems operational. The arc reactor is functioning at optimal capacity.")
    
    print("\n    ✓ Voice synthesis working!")
    
except Exception as e:
    print(f"    ✗ Voice synthesis failed: {e}")
    print("\n    This might be because Coqui TTS model needs to download.")
    print("    First run may take a few minutes to download the model (~300MB)")

# Test 4: Iron Man knowledge with voice
print("\n[4] Testing Iron Man knowledge with voice...")
try:
    response = get_jarvis_response("Who are you?")
    if response:
        print(f"    Response: {response[:100]}...")
        jarvis.speak(response)
        print("    ✓ Knowledge + Voice working!")
except Exception as e:
    print(f"    ✗ Failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
