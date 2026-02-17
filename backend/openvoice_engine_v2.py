import os
import torch
import numpy as np
import logging
import hashlib
import shutil
from typing import Union, List
from RealtimeTTS.engines import BaseEngine
from scipy.signal import butter, lfilter

# Add OpenVoice to path
import sys
from pathlib import Path
OPENVOICE_DIR = Path(__file__).parent.parent / "OpenVoice"
sys.path.append(str(OPENVOICE_DIR))

try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS
    OPENVOICE_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: OpenVoice Import Error: {e}")
    OPENVOICE_AVAILABLE = False
except Exception as e:
    print(f"DEBUG: OpenVoice Generic Error: {e}")
    OPENVOICE_AVAILABLE = False


# GLOBAL MODEL CACHE - Persists across instance recreation
_GLOBAL_CONVERTER = None
_GLOBAL_MELO_TTS = None
_GLOBAL_TARGET_SE = None

class OpenVoiceEngine(BaseEngine):
    def __init__(
        self,
        checkpoint_path: str = "jarvis-system/OpenVoice/checkpoints_v2/converter",
        device: str = "cpu",
        speed: float = 1.05,
        voice_sample: str = "Paul Bettany Breaks Down His Most Iconic Characters _ GQ-enhanced-v2.wav"
    ):
        """
        Initializes the OpenVoice V2 Engine for RealtimeTTS.
        """
        super().__init__()
        self.engine_name = "openvoice"
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.speed = speed
        self.voice_sample = voice_sample
        
        # New High-Fidelity Parameters
        self.pitch_shift = -1  # Slightly deeper for authority
        self.energy = 0.9      # Calm, measured energy
        
        # Caching Setup
        self.cache_dir = "jarvis_voice_cache_openvoice"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Pull from global cache
        self.converter = _GLOBAL_CONVERTER
        self.melo_tts = _GLOBAL_MELO_TTS
        self.target_se = _GLOBAL_TARGET_SE
        
        # Lazy load on first synthesize
        self.checkpoint_path = checkpoint_path
        
        if not OPENVOICE_AVAILABLE:
            logging.error("OpenVoice or MeloTTS not installed.")

    def post_init(self):
        self.engine_name = "openvoice"

    def preload_models(self):
        """Force load models immediately (blocks until done)."""
        self._load_models()

    def _load_models(self):
        global _GLOBAL_CONVERTER, _GLOBAL_MELO_TTS, _GLOBAL_TARGET_SE
        
        if _GLOBAL_CONVERTER is not None:
            self.converter = _GLOBAL_CONVERTER
            self.melo_tts = _GLOBAL_MELO_TTS
            self.target_se = _GLOBAL_TARGET_SE
            return

        print(f"[OpenVoice] Loading models on {self.device}...")
        try:
            config_path = os.path.join(self.checkpoint_path, 'config.json')
            ckpt_path = os.path.join(self.checkpoint_path, 'checkpoint.pth')
            
            _GLOBAL_CONVERTER = ToneColorConverter(config_path, device=self.device)
            try:
                _GLOBAL_CONVERTER.load_ckpt(ckpt_path)
            except RuntimeError as e:
                print(f"[OpenVoice] Warning: Checkpoint mismatch ({e}). Attempting to proceed...")
            
            # Load MeloTTS (English by default)
            _GLOBAL_MELO_TTS = MeloTTS(language='EN', device=self.device)
            
            # Extract target SE from voice sample
            if os.path.exists(self.voice_sample):
                _GLOBAL_TARGET_SE, _ = se_extractor.get_se(self.voice_sample, _GLOBAL_CONVERTER, vad=True)
            else:
                print(f"[OpenVoice] Warning: Voice sample {self.voice_sample} not found.")
                
            self.converter = _GLOBAL_CONVERTER
            self.melo_tts = _GLOBAL_MELO_TTS
            self.target_se = _GLOBAL_TARGET_SE
                
        except Exception as e:
            print(f"[OpenVoice] Load Error: {e}")

    def _clean_text(self, text: str) -> str:
        """Text preprocessing to fix phoneme stability."""
        text = text.strip()
        text = text.replace("...", ".")
        text = text.replace("?", "?.")
        text = text.replace("!", "!.")
        text = text.replace(",", ", ")
        return text

    def _chunk_text(self, text: str, max_words: int = 18) -> List[str]:
        """Sentence chunking to prevent phonetic loops."""
        words = text.split()
        chunks, current = [], []

        for w in words:
            current.append(w)
            if len(current) >= max_words or w.endswith(('.', '!', '?')):
                chunks.append(" ".join(current))
                current = []

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _apply_post_processing(self, data: np.ndarray, samplerate: int) -> np.ndarray:
        """High-pass filter, pitch shift, and normalization."""
        from scipy.interpolate import interp1d
        
        def highpass_filter(data, cutoff=80, fs=samplerate, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return lfilter(b, a, data)
        
        def shift_pitch(data, semitones):
            """Simple pitch shift via resampling."""
            factor = 2**(semitones/12.0)
            n = len(data)
            x = np.arange(n)
            y = data
            f = interp1d(x, y, kind='linear', fill_value="extrapolate")
            new_x = np.linspace(0, n-1, int(n / factor))
            return f(new_x)

        try:
            # 1. Pitch Shift (Deeper)
            if hasattr(self, 'pitch_shift') and self.pitch_shift != 0:
                data = shift_pitch(data, self.pitch_shift)
            
            # 2. High-pass filter @ 80Hz
            data = highpass_filter(data)
            
            # 3. Energy/Volume (Normalization)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                target_gain = getattr(self, 'energy', 0.9)
                data = data / max_val * target_gain
                
            return data.astype(np.float32)
        except Exception as e:
            print(f"[OpenVoice] Post-processing error: {e}")
            return data

    def get_stream_info(self):
        # OpenVoice/MeloTTS typical output is 44100Hz or 24000Hz. 
        # MeloTTS is usually 16kHz or 24kHz depending on config.
        # We'll use 24000Hz as default for OpenVoice V2.
        return 1, 1, 24000 

    def synthesize(self, text: str):
        if not OPENVOICE_AVAILABLE:
            return False
            
        self._load_models()
        self.stop_synthesis_event.clear()
        
        cleaned_text = self._clean_text(text)
        print(f"[OpenVoice] Synthesizing '{cleaned_text[:40]}...'")
        
        # Check Cache
        cache_key = f"{cleaned_text}_{self.speed}_{self.voice_sample}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_hash}.wav")
        
        import soundfile as sf
        
        if os.path.exists(cache_path):
            print(f"[OpenVoice] Cache hit: {cache_hash}")
            data, samplerate = sf.read(cache_path)
            # Feed chunks to queue
            chunk_size = 1024
            for i in range(0, len(data), chunk_size):
                if self.stop_synthesis_event.is_set(): break
                chunk = np.array(data[i:i+chunk_size]).astype(np.float32)
                self.queue.put(chunk.tobytes())
            return True

        chunks = self._chunk_text(cleaned_text)
        all_audio = []
        
        try:
            output_dir = "temp_voice"
            os.makedirs(output_dir, exist_ok=True)
            
            # Load British Base
            source_se_path = OPENVOICE_DIR / "checkpoints_v2/base_speakers/ses/en-br.pth"
            source_se = torch.load(source_se_path if source_se_path.exists() else list((OPENVOICE_DIR / "checkpoints_v2/base_speakers/ses").glob("*.pth"))[0], map_location=self.device)
            
            # Robust speaker ID access for MeloTTS HParams
            try:
                speaker_id = self.melo_tts.hps.data.spk2id['EN-BR']
            except:
                speaker_id = 1
            
            for idx, chunk in enumerate(chunks):
                if self.stop_synthesis_event.is_set(): break
                
                temp_wav = os.path.join(output_dir, f"base_{idx}.wav")
                final_wav = os.path.join(output_dir, f"final_{idx}.wav")
                
                # 1. Base TTS
                self.melo_tts.tts_to_file(chunk, speaker_id, temp_wav, speed=self.speed)
                
                # 2. Color Conversion
                self.converter.convert(
                    audio_src_path=temp_wav,
                    src_se=source_se,
                    tgt_se=self.target_se,
                    output_path=final_wav
                )
                
                # 3. Process Audio
                data, samplerate = sf.read(final_wav)
                processed_data = self._apply_post_processing(data, samplerate)
                all_audio.append(processed_data)
                
                # 4. Feed to queue immediately for low latency
                chunk_size = 1024
                for i in range(0, len(processed_data), chunk_size):
                    if self.stop_synthesis_event.is_set(): break
                    c = processed_data[i:i+chunk_size]
                    self.queue.put(c.tobytes())
            
            # Save to cache if complete
            if not self.stop_synthesis_event.is_set() and all_audio:
                full_audio = np.concatenate(all_audio)
                sf.write(cache_path, full_audio, 24000)
                
            return True
            
        except Exception as e:
            print(f"[OpenVoice] Synthesis Error: {e}")
            return False

    def get_voices(self):
        return []

    def set_voice(self, voice):
        pass

    def set_voice_parameters(self, **voice_parameters):
        if 'speed' in voice_parameters:
            self.speed = voice_parameters['speed']

    def shutdown(self):
        pass
