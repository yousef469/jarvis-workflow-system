"""Jarvis Configuration"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VOSK_MODEL_PATH = MODELS_DIR / "vosk-model-en-us-0.22"
WAKE_WORD_MODEL_PATH = MODELS_DIR / "openwakeword"

# User Settings
USER_NAME = os.getenv("USER_NAME", "Sir")
WAKE_WORD = os.getenv("WAKE_WORD", "jarvis")
WAKE_WORD_SENSITIVITY = float(os.getenv("WAKE_WORD_SENSITIVITY", "0.25"))
MODEL_BRAIN = "qwen2.5-coder:3b"
WHISPER_MODEL = "small.en"
DIGITAL_GAIN = 5.0
JARVIS_GREETING = "Yes sir."
JARVIS_GOODBYE = f"Going quiet, {USER_NAME}. Say my name when you need me."
JARVIS_CONFIRM = "Right away, sir."
JARVIS_ERROR = "I'm sorry, I couldn't do that."

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_PORT = int(os.getenv("API_PORT", "8765"))
WS_PORT = int(os.getenv("WS_PORT", "8766"))

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 2
CHUNK_SIZE = 4096  # Larger chunk prevents "Input Overflow"

# MICROPHONE ID (Found via check_mic.py)
# Options: 1 = External Mic, 2 = Internal Mic, 0 = Default
MIC_INDEX = 0  # Changed from 2 to 0 for Mac compatibility

# Timeouts
LISTENING_TIMEOUT = 60  # seconds of silence before going idle
COMMAND_TIMEOUT = 10    # seconds to wait for command after wake word

MODEL_ROUTER = "qwen2.5-coder:3b"
MODEL_EXPERT = "qwen2.5-coder:3b"
MODEL_VISION = "qwen3-vl:8b"    # V3 Vision Professor (Deep visual reasoning for lectures)
JARVIS_RESPONSES = {
    "greeting": [
        "Hey sir, how can I help you?",
        "At your service, sir.",
        "How can I help?",
        "I'm listening, sir.",
    ],
    "confirm": [
        "Right away, sir.",
        "On it.",
        "Consider it done.",
        "Executing now, sir.",
    ],
    "error": [
        "I'm sorry, I couldn't do that.",
        "That didn't work as expected, sir.",
        "I encountered an issue with that request.",
    ],
    "goodbye": [
        f"Going quiet, {USER_NAME}. Say my name when you need me.",
        "Standing by, sir.",
        "I'll be here when you need me.",
    ]
}
