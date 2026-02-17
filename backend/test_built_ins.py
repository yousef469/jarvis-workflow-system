from openwakeword.model import Model
import os

try:
    # Try loading several common variants
    variants = ["hey_jarvis", "jarvis", "hey_mycroft"]
    for v in variants:
        try:
            m = Model(wakeword_models=[v])
            print(f"✅ SUCCESSFULLY loaded built-in model: {v}")
            # print(f"Keys: {list(m.prediction_buffer.keys())}")
        except:
            print(f"❌ Failed to load built-in model: {v}")
except Exception as e:
    print(f"Critical error: {e}")
