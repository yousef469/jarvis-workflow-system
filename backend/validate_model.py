import numpy as np
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from config import WAKE_WORD_MODEL_PATH

try:
    from openwakeword.model import Model
    print("✅ openwakeword library loaded.")
except ImportError:
    print("❌ openwakeword library NOT found.")
    sys.exit(1)

def validate_model():
    model_path = str(WAKE_WORD_MODEL_PATH / "hey_jarvis.onnx")
    print(f"Checking model at: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Model file does NOT exist.")
        return
    
    try:
        # Try to load the model
        model = Model(wakeword_models=[model_path], inference_framework="onnx")
        print("✅ Model loaded successfully into openwakeword.")
        
        # Feed some random data to see if it predicts without crashing
        data = np.random.randint(-32768, 32767, 1280, dtype=np.int16)
        prediction = model.predict(data)
        print(f"✅ Prediction test pass. Score (random): {prediction}")
        
    except Exception as e:
        print(f"❌ Model load/test FAILED: {e}")

if __name__ == "__main__":
    validate_model()
