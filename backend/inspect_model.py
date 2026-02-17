from openwakeword.model import Model
import os
import sys

# Assume model path
model_path = "/Users/yousef/Desktop/jarvis/jarvis/backend/models/openwakeword/hey_jarvis.onnx"

if not os.path.exists(model_path):
    print("Model not found!")
    sys.exit(1)

print(f"Loading {model_path}...")
oww = Model(wakeword_models=[model_path], inference_framework="onnx")

print(f"Prediction buffer keys: {list(oww.prediction_buffer.keys())}")
print(f"Models keys: {list(oww.models.keys())}")
print(f"ONNX model paths: {oww.wakeword_models}")
