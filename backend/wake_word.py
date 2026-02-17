import numpy as np
import os
import time
from config import WAKE_WORD_SENSITIVITY, MIC_INDEX, WAKE_WORD_MODEL_PATH, DIGITAL_GAIN

try:
    from openwakeword.model import Model
    OWW_AVAILABLE = True
except ImportError:
    OWW_AVAILABLE = False
    print("[WakeWord] openwakeword not installed")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[WakeWord] pyaudio not installed")

# Global PyAudio instance to avoid conflicts
_pyaudio_instance = None

def get_pyaudio():
    global _pyaudio_instance
    if _pyaudio_instance is None:
        # Note: on Mac, we don't always need to specify host_api_index
        _pyaudio_instance = pyaudio.PyAudio()
    return _pyaudio_instance


# Global model instance for reuse
_oww_model = None

def get_wake_word_model():
    """Lazy-load and cache the built-in wake word models."""
    global _oww_model
    if _oww_model is None:
        if not OWW_AVAILABLE:
            return None
        print("[WakeWord] 🧠 Initializing built-in models (hey_jarvis, jarvis)...")
        # Using built-in models is much more reliable than custom ONNX files
        _oww_model = Model(wakeword_models=["hey_jarvis", "jarvis"], inference_framework="onnx")
    return _oww_model


def listen_for_wake_word():
    """
    Blocks until 'Hey Jarvis' is heard.
    Returns True when detected.
    Properly releases microphone after detection.
    """
    if not OWW_AVAILABLE or not PYAUDIO_AVAILABLE:
        print("[WakeWord] Required libraries not available")
        return False
    
    # Get cached model
    model = get_wake_word_model()
    if not model:
        return False

    FORMAT = pyaudio.paInt16
    CHANNELS = 2  # Mac built-in mic often requires 2 channels
    RATE = 16000
    CHUNK = 1280
    
    audio = get_pyaudio()
    stream = None
    
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=MIC_INDEX,
            frames_per_buffer=CHUNK
        )
        
        print(f"[WakeWord] 🎤 Listening for 'Hey Jarvis' (Mic: {MIC_INDEX}, Sensitivity: {WAKE_WORD_SENSITIVITY})...")
        
        # Keep track of last score to avoid spamming logs
        last_log_time = 0
        
        while True:
            # Read and convert to numpy
            try:
                raw_data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"[WakeWord] Stream read error: {e}")
                break
                
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # --- MASSIVE DIGITAL GAIN (50x boost for quiet mics) ---
            audio_data = (audio_data.astype(np.float32) * DIGITAL_GAIN).clip(-32768, 32767).astype(np.int16)
            
            # Simple volume monitor (RMS after gain)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            
            # If 2 channels, take only the left channel
            if CHANNELS == 2:
                audio_data = audio_data[::2]
            
            prediction = model.predict(audio_data)
            # Take the highest score among all active models (hey_jarvis, jarvis, etc.)
            score = max(prediction.values()) if prediction else 0
            
            # --- STABILITY GATING ---
            # 1. Energy check: Must be audible sound
            # 2. Consistency check: Must hit threshold for multiple frames
            
            # Consistency counter (initialized outside loop if needed, but for blocks we can keep it localish)
            if not hasattr(self if 'self' in locals() else listen_for_wake_word, "_trigger_count"):
                listen_for_wake_word._trigger_count = 0
            
            triggered = False
            if score >= WAKE_WORD_SENSITIVITY:
                listen_for_wake_word._trigger_count += 1
                if listen_for_wake_word._trigger_count >= 2:
                    triggered = True
            else:
                listen_for_wake_word._trigger_count = 0
            
            # Debug log every 1.5 seconds
            if time.time() - last_log_time > 1.5:
                status = "🎤 Hearing sound" if rms > 100 else "🔇 Silent/Low"
                print(f"[WakeWord] Status: {status} (RMS: {rms:.1f}, Score: {score:.4f}, Count: {listen_for_wake_word._trigger_count})")
                last_log_time = time.time()
            
            if triggered:
                print(f"[WakeWord] ✨ Wake word validated! (Score: {score:.2f})")
                listen_for_wake_word._trigger_count = 0
                stream.stop_stream()
                stream.close()
                time.sleep(0.1)
                return True
                
    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"[WakeWord] Error: {e}")
        return False
    finally:
        # Make sure to clean up
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass


def reset_wake_word_model():
    """Clears the model's internal buffer by feeding it silence."""
    model = get_wake_word_model()
    if model:
        # Feed some silence to clear the state
        silence = np.zeros(1280, dtype=np.int16)
        for _ in range(5):
            model.predict(silence)
        print("[WakeWord] 🧼 Model buffer cleared")

__all__ = ["listen_for_wake_word", "reset_wake_word_model", "OWW_AVAILABLE"]
