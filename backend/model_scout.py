import os
import sys
from openwakeword.model import Model

def model_scout():
    print("\n--- OPENWAKEWORD MODEL SCOUT ---")
    
    # Check built-in models
    try:
        # Some versions use get_model_names, others don't have it easily accessible
        # In modern openwakeword, you often just pass a list of names or paths
        print("Scouting for models in the library...")
    except:
        pass

    # Check the user's custom models folder
    custom_path = "/Users/yousef/Desktop/jarvis/jarvis/backend/models/openwakeword"
    if os.path.exists(custom_path):
        print(f"\nFiles in {custom_path}:")
        for f in os.listdir(custom_path):
            if f.endswith(".onnx"):
                print(f" - {f}")
    
    # Test loading a few standard names to see what's bundled
    standard_names = ["alexa", "hey_google", "hey_siri", "hey_mycroft", "casita"]
    print("\nTesting which standard models are available...")
    for name in standard_names:
        try:
            m = Model(wakeword_models=[name])
            print(f" ✅ {name} is available")
        except:
            print(f" ❌ {name} not found")

if __name__ == "__main__":
    model_scout()
