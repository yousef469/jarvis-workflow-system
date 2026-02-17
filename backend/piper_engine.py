import os
import time
import hashlib
import threading
import queue
import wave
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict

# Piper imports
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("[PiperEngine] ❌ Piper TTS not installed. Run `pip install piper-tts`.")

class JarvisVoicePiper:
    """
    JARVIS Voice v4 - Piper TTS (Ryan High)
    Features:
    - Extremely fast (Real-time Factor < 1.0)
    - Local, offline synthesis
    - Persistent caching for instant replies
    """
    
    def __init__(
        self,
        model_path: str = os.path.join(os.path.dirname(__file__), "models", "piper", "en_US-ryan-high.onnx"),
        cache_dir: str = os.path.join(os.path.dirname(__file__), "piper_cache")
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = model_path
        self.config_path = f"{model_path}.json"
        
        # Point to the piper executable in the virtualenv
        self.piper_exe = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jarvis_env_mac", "bin", "piper")
        
        self._lock = threading.Lock()
        self._is_speaking = False
        self._generation_queue = queue.Queue()
        self._file_cache: Dict[str, str] = {} # Map hash -> filepath
        
        # Load cache index
        self._load_cache_index()
        
        # Verify Piper is available
        self._verify_piper()
        
        # Start background worker
        self._start_background_generator()
        
        # Pre-warm common caches (DISABLED FOR RAM)
        # self.pre_warm_cache()

    def _verify_piper(self):
        """Check if piper executable exists."""
        if os.path.exists(self.piper_exe):
            print(f"[PiperEngine] ✅ Piper CLI found at: {self.piper_exe}")
            print(f"[PiperEngine] ⚡️ Model: {os.path.basename(self.model_path)}")
        else:
            print(f"[PiperEngine] ❌ Piper CLI NOT found at: {self.piper_exe}")

    def _get_cache_key(self, text: str) -> str:
        """MD5 hash of lowercase text"""
        clean = text.lower().strip()
        return hashlib.md5(clean.encode()).hexdigest()

    def _load_cache_index(self):
        """Index existing wav files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.wav"):
            # Check for invalid 44-byte files and delete them
            if cache_file.stat().st_size <= 44:
                cache_file.unlink()
                continue
            self._file_cache[cache_file.stem] = str(cache_file)
            count += 1
        if count > 0:
            print(f"[PiperEngine] 📂 Loaded {count} valid cached phrases.")

    def _generate_audio(self, text: str) -> Optional[str]:
        """Generate audio via Piper CLI."""
        try:
            key = self._get_cache_key(text)
            output_path = self.cache_dir / f"{key}.wav"
            
            if output_path.exists():
                return str(output_path)
            
            print(f"[PiperEngine] 🛠 Generating: \"{text[:30]}...\"")
            
            # Simple CLI call (input via stdin)
            cmd = [
                self.piper_exe,
                "--model", self.model_path,
                "--output_file", str(output_path)
            ]
            
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0 and output_path.exists() and output_path.stat().st_size > 50:
                self._file_cache[key] = str(output_path)
                return str(output_path)
            else:
                print(f"[PiperEngine] ❌ Generation failed: {stderr}")
                return None
                
        except Exception as e:
            print(f"[PiperEngine] CLI error: {e}")
            return None

    def pre_warm_cache(self):
        """Pre-generate common phrases to ensure zero-latency responses."""
        common_phrases = [
            "Yes, sir.",
            "Certainly, sir.",
            "Right away, sir.",
            "I'm on it, sir.",
            "Opening Chrome, sir.",
            "Done, sir.",
            "I'm sorry sir, something went wrong.",
            "Systems operational.",
            "What can I do for you?"
        ]
        print(f"[PiperEngine] ⚡️ Pre-warming {len(common_phrases)} common phrases...")
        for phrase in common_phrases:
            self._generation_queue.put(phrase)

    def _start_background_generator(self):
        """Background worker to handle queued speech."""
        def worker():
            while True:
                try:
                    text = self._generation_queue.get()
                    if text is None: break
                    
                    key = self._get_cache_key(text)
                    if key in self._file_cache: continue
                    
                    # Generate
                    self._generate_audio(text)
                    
                except Exception as e:
                    print(f"[PiperEngine] Worker error: {e}")
                finally:
                    self._generation_queue.task_done()
        
        threading.Thread(target=worker, daemon=True).start()

    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using Piper (DISABLED).
        """
        print(f"[PiperEngine] 🔕 Piper TTS disabled by user request. Skipping: \"{text[:30]}...\"")
        return False

    def is_speaking(self) -> bool:
        return self._is_speaking

    def stop(self):
        try:
            subprocess.run(["killall", "afplay"], stderr=subprocess.DEVNULL)
        except: pass
        self._is_speaking = False

# Global Instance
_piper_engine = None

def get_piper_engine():
    global _piper_engine
    if _piper_engine is None:
        _piper_engine = JarvisVoicePiper()
    return _piper_engine
