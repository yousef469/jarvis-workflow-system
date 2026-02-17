import requests
import time
import json
from config import MODEL_BRAIN, MODEL_VISION, MODEL_EXPERT

OLLAMA_URL = "http://localhost:11434"

class ModelManager:
    """
    Manages dynamic loading and unloading of Ollama models to optimize VRAM.
    Enforces a strict cycle: Unload unused -> Load required.
    """
    def __init__(self):
        self.current_mode = "none"  # "brain", "vision", "expert", "none"
        self.vision_llm_loaded = False

    def _unload_model(self, model_name: str):
        """Force unload a model by setting keep_alive to 0."""
        print(f"[ModelManager] 🔻 Unloading {model_name}...")
        try:
            requests.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model_name,
                "keep_alive": 0
            }, timeout=5)
        except Exception as e:
            print(f"[ModelManager] Warning: Failed to unload {model_name}: {e}")

    def _preload_model(self, model_name: str):
        """Preload a model into memory."""
        print(f"[ModelManager] 🔼 Loading {model_name}...")
        try:
            # Send an empty request with long keep_alive to load it
            requests.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model_name,
                "keep_alive": "10m"
            }, timeout=30) # Preload can take a bit longer
        except Exception as e:
            print(f"[ModelManager] Error loading {model_name}: {e}")

    def switch_to_brain(self):
        """Switch to Chat/Coding Brain (Unload Vision)."""
        if self.current_mode == "brain": return
        
        print("\n[ModelManager] 🧠 Switching to BRAIN (Coding/Chat)...")
        self._unload_model(MODEL_VISION) # Free up VRAM
        self._preload_model(MODEL_BRAIN)
        self.current_mode = "brain"

    def switch_to_vision(self):
        """Switch to Vision Model (Unload Brain)."""
        if self.current_mode == "vision": return
        
        print("\n[ModelManager] 👁️ Switching to VISION (Note Taker)...")
        self._unload_model(MODEL_BRAIN) # Free up VRAM
        self._preload_model(MODEL_VISION)
        self.current_mode = "vision"

    def switch_to_expert(self):
        """Expertise is handled by the Brain."""
        self.switch_to_brain()

    def unload_all(self):
        """Unload ALL models to free VRAM for other tasks."""
        print("\n[ModelManager] 🛑 Unloading ALL models...")
        self._unload_model(MODEL_BRAIN)
        self._unload_model(MODEL_VISION)
        self.current_mode = "none"
        print("[ModelManager] ✅ VRAM Cleared\n")

model_manager = ModelManager()
