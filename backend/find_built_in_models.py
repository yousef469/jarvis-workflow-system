from openwakeword.model import Model
import openwakeword
import os

print(f"Openwakeword version: {openwakeword.__version__}")
# Attempt to initialize a Model with no arguments to see default behavior
try:
    # This usually loads all default models if they exist in the package
    model = Model()
    print(f"Default models loaded: {list(model.models.keys())}")
except Exception as e:
    print(f"Error loading default models: {e}")

# Check package directory for pre-trained models
package_dir = os.path.dirname(openwakeword.__file__)
models_dir = os.path.join(package_dir, "resources", "models")
if os.path.exists(models_dir):
    print(f"Models in package directory ({models_dir}):")
    for f in os.listdir(models_dir):
        if f.endswith(".onnx"):
            print(f" - {f}")
else:
    print(f"No models found in {models_dir}")
