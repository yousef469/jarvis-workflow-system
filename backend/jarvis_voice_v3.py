import os
import time
import hashlib
import threading
import queue
import numpy as np
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Dict

# Coqui TTS import
import torch
from torch.nn.utils import weight_norm as _weight_norm
def weight_norm(module, name='weight', dim=0):
    return _weight_norm(module, name, dim)
torch.nn.utils.weight_norm = weight_norm

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

class JarvisVoiceV3:
    """
    JARVIS Voice v3 - Coqui XTTS Restoration (DISABLED)
    """
    
    def __init__(
        self,
        voice_sample: str = os.path.join(os.path.dirname(__file__), "jarvis_test_voice.wav"),
        cache_dir: str = os.path.join(os.path.dirname(__file__), "jarvis_voice_cache")
    ):
        print("[VoiceV3] 🔕 LEGACY VOICE ENGINE DISABLED.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.voice_sample = voice_sample
        self.sample_rate = 24000
        self._lock = threading.Lock()
        self._is_speaking = False
        self._generation_queue = queue.Queue()
        self._audio_cache: Dict[str, np.ndarray] = {}
        
        # Coqui State
        self.tts = None
        self.is_loading = False
        
        # Load existing cache into memory for instant lookup
        self._load_cached_audio()
        
        # Start background worker
        self._start_background_generator()
        
        # Lazy load model
        if COQUI_AVAILABLE:
            threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        """Pre-load the heavy XTTS model (DISABLED)."""
        print("[VoiceV3] 🔕 Coqui XTTS model loading DISABLED by user request.")
        self.is_loading = False
        return

    def _get_cache_key(self, text: str) -> str:
        """MD5 hash of lowercase text (first 16 chars)"""
        clean = text.lower().strip()
        return hashlib.md5(clean.encode()).hexdigest()[:16]

    def _load_cached_audio(self):
        """Load all cached .npy files"""
        count = 0
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                audio = np.load(cache_file)
                self._audio_cache[cache_file.stem] = audio
                count += 1
            except: pass
        if count > 0:
            print(f"[VoiceV3] 📂 Loaded {count} phrases from long-term memory.")

    def _is_cached(self, text: str) -> bool:
        return self._get_cache_key(text) in self._audio_cache

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        key = self._get_cache_key(text)
        return self._audio_cache.get(key)

    def _generate_coqui(self, text: str) -> Optional[np.ndarray]:
        """Generate audio with tuned Coqui parameters"""
        if self.tts is None or self.is_loading:
            return None
        
        try:
            # We use a temp file because Coqui's tts() returns a list of floats 
            # but tts_to_file is more stable for long sentences
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_path = tf.name
            
            # Application of the "Loop-Fix" parameters
            self.tts.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker_wav=self.voice_sample,
                language="en",
                # Model-level overrides (Phoneme loop fixes)
                temperature=0.35,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.85,
                speed=1.08, # Speed override
                # Note: Some versions of TTS API might not pass these directly in tts_to_file
                # If they fail, we fallback to standard but these are the targets.
            )
            
            import soundfile as sf
            data, samplerate = sf.read(temp_path)
            
            try: os.remove(temp_path)
            except: pass
            
            return data.astype(np.float32)
        except Exception as e:
            print(f"[VoiceV3] Generation error: {e}")
            return None

    def _start_background_generator(self):
        """Background thread to generate voice for cache misses"""
        def worker():
            while True:
                try:
                    text = self._generation_queue.get()
                    if text is None: break
                    if self._is_cached(text): continue
                    
                    print(f"[VoiceV3] ⏳ Background Worker picked up: '{text[:20]}...'")
                    
                    # Wait for model if still loading
                    while self.is_loading and self.tts is None:
                        time.sleep(1)
                    
                    if self.tts:
                        print(f"[VoiceV3] 🎙️  Generating audio via Coqui...")
                        start = time.time()
                        audio = self._generate_coqui(text)
                        if audio is not None:
                            key = self._get_cache_key(text)
                            self._audio_cache[key] = audio
                            np.save(self.cache_dir / f"{key}.npy", audio)
                            elapsed = time.time() - start
                            print(f"[VoiceV3] ✨ Background Cache Complete: '{text[:25]}...' ({elapsed:.2f}s)")
                        else:
                            print("[VoiceV3] ❌ Generation failed (audio is None)")
                    else:
                        print("[VoiceV3] ❌ TTS Model is NOT loaded.")
                except Exception as e:
                    print(f"[VoiceV3] Background worker error: {e}")
                finally:
                    self._generation_queue.task_done()
        
        threading.Thread(target=worker, daemon=True).start()

    def speak(self, text: str, blocking: bool = False):
        """
        Main entry point.
        Cache HIT -> Instant Play.
        Cache MISS -> Text only (in server), Queue background generation.
        """
        if not text: return
        
        # Check cache
        audio_data = self._get_cached(text)
        
        if audio_data is None:
            # MISS: Queue for next time, return immediately (so server shows text)
            if not self._is_cached(text):
                self._generation_queue.put(text)
            return False

        # HIT: Playback
        self._is_speaking = True
        try:
            # Play via afplay for Mac system integration
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_path = tf.name
            
            # Convert float32 [-1, 1] to int16
            audio_int = (audio_data * 32767).astype(np.int16)
            
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int.tobytes())
            
            cmd = ["afplay", temp_path]
            if blocking:
                subprocess.run(cmd)
            else:
                subprocess.Popen(cmd)
            
            # Cleanup thread
            def cleanup():
                time.sleep(len(audio_data)/self.sample_rate + 1)
                try: os.remove(temp_path)
                except: pass
                self._is_speaking = False
            
            threading.Thread(target=cleanup, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"[VoiceV3] Playback error: {e}")
            self._is_speaking = False
            return False

    def is_speaking(self) -> bool:
        return self._is_speaking

    def stop(self):
        try:
            subprocess.run(["killall", "afplay"], stderr=subprocess.DEVNULL)
        except: pass
        self._is_speaking = False

# Global Instance
_jarvis_v3 = None

def get_voice_v3():
    global _jarvis_v3
    if _jarvis_v3 is None:
        _jarvis_v3 = JarvisVoiceV3()
    return _jarvis_v3
