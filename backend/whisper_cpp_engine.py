"""
Faster-Whisper STT Engine - DUAL MODE (Commands + Lectures)
============================================================
MODE 1: "small.en" - For voice commands (fast, low RAM)
MODE 2: "turbo"    - For lecture transcription (high accuracy)
"""

import numpy as np
import time
import sys
import gc
from typing import Optional

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("[STT] faster-whisper not installed")


# GLOBAL MODEL CACHES - Persist across instance recreation
_GLOBAL_COMMAND_MODEL = None   # small.en (always loaded for commands)
_GLOBAL_LECTURE_MODEL = None   # turbo (loaded on-demand for lectures)

class WhisperCppEngine:
    """Dual-mode STT engine using faster-whisper"""
    
    def __init__(
        self,
        command_model: Optional[str] = None, # Loaded from config below
        lecture_model: str = "turbo",
        compute_type: str = "float32",  # Switched for stability (int8 segfaults on some Mac CPUs)
        energy_threshold: int = 250,
        silence_duration: float = 0.8,
        sample_rate: int = 16000
    ):
        from config import WHISPER_MODEL
        self.command_model_name = command_model or "small.en"
        self.lecture_model_name = lecture_model or WHISPER_MODEL
        self.lecture_model_name = lecture_model
        self.compute_type = compute_type
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        
        # Use global cache if available
        self.command_model = _GLOBAL_COMMAND_MODEL
        self.lecture_model = _GLOBAL_LECTURE_MODEL
        self.whisper_path = None
        
        self.hallucinations = [
            "oven", "oven chrome", "open current", "go to", "in england",
            "thank you.", "subtitles by", "you", "uh", "um", "in the", "re",
            "listening", "thought", "you.", "it", "so", "of england", "urban",
            "open current", "open browser", "open source", "open up", "open it"
        ]
    
    def _load_command_model(self):
        """Load the fast small.en model for voice commands."""
        global _GLOBAL_COMMAND_MODEL
        if _GLOBAL_COMMAND_MODEL is not None:
            self.command_model = _GLOBAL_COMMAND_MODEL
            return True
        
        if not FASTER_WHISPER_AVAILABLE:
            print("[STT] faster-whisper not available.")
            return False

        print(f"[STT] Loading COMMAND model: {self.command_model_name} ({self.compute_type} CPU)...")
        try:
            from faster_whisper import WhisperModel
            _GLOBAL_COMMAND_MODEL = WhisperModel(
                self.command_model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=1,  # Single-thread to prevent segfault on Mac ARM
                num_workers=1
            )
            self.command_model = _GLOBAL_COMMAND_MODEL
            print(f"[STT] ✅ Command model ({self.command_model_name}) loaded.")
            return True
        except Exception as e:
            print(f"[STT] ❌ Command model load failed: {e}")
            return False
    
    def _load_lecture_model(self):
        """Load the high-accuracy turbo model for lecture transcription."""
        global _GLOBAL_LECTURE_MODEL
        if _GLOBAL_LECTURE_MODEL is not None:
            self.lecture_model = _GLOBAL_LECTURE_MODEL
            return True
        
        if not FASTER_WHISPER_AVAILABLE:
            print("[STT] faster-whisper not available.")
            return False

        print(f"[STT] Loading LECTURE model: {self.lecture_model_name} ({self.compute_type} CPU)...")
        print(f"[STT] ⏳ This may take a moment (turbo is larger but much more accurate)...")
        try:
            from faster_whisper import WhisperModel
            _GLOBAL_LECTURE_MODEL = WhisperModel(
                self.lecture_model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=1,  # Single-thread to prevent segfault on Mac ARM
                num_workers=1
            )
            self.lecture_model = _GLOBAL_LECTURE_MODEL
            print(f"[STT] ✅ Lecture model ({self.lecture_model_name}) loaded.")
            return True
        except Exception as e:
            print(f"[STT] ❌ Lecture model load failed: {e}")
            return False
    
    # Legacy compatibility
    def _load_model(self):
        return self._load_command_model()
    
    @property
    def model(self):
        """Legacy property for backward compatibility."""
        return self.command_model
    
    @model.setter
    def model(self, value):
        self.command_model = value
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based speech detection"""
        energy = np.abs(audio_chunk).mean()
        return energy > self.energy_threshold
    
    def transcribe(self, audio_data: np.ndarray, mode: str = "command") -> Optional[str]:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Raw audio as int16 numpy array
            mode: "command" for fast small.en, "lecture" for high-accuracy turbo
        """
        if mode == "lecture":
            if not self._load_lecture_model():
                print("[STT] Falling back to command model for lecture...")
                if not self._load_command_model():
                    return None
                active_model = self.command_model
            else:
                active_model = self.lecture_model
        else:
            if not self._load_command_model():
                return None
            active_model = self.command_model
            
        try:
            # Convert int16 to float32 ONCE
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            sig_max = np.max(np.abs(audio_float))
            print(f"[STT] Transcribing: signal_max={sig_max:.4f}, mode={mode}")
            
            if mode == "lecture":
                # High-accuracy settings for lecture content
                segments, info = active_model.transcribe(
                    audio_float,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=600,    # Lectures have longer pauses
                        min_speech_duration_ms=250,     # Capture full sentences
                        speech_pad_ms=800               # Wide buffer for lecture pacing
                    ),
                    condition_on_previous_text=True      # Better context for lectures
                )
            else:
                # Fast settings for voice commands
                # NOTE: vad_filter DISABLED - it loads Silero VAD (ONNX) which
                # conflicts with OpenWakeWord's ONNX and causes segfault on Mac ARM.
                # Energy-based detection in listen_and_transcribe handles VAD instead.
                segments, info = active_model.transcribe(
                    audio_float,
                    beam_size=5, # Higher quality
                    language="en",
                    vad_filter=False,
                    condition_on_previous_text=False,
                    temperature=0.0 # Deterministic
                )
            
            result_text = " ".join([seg.text for seg in segments]).strip()
            print(f"[STT] Result: '{result_text[:50]}...' (Language: {info.language}, Prob: {info.language_probability:.2f})")
            
            # Force cleanup of native ctranslate2 objects to prevent segfault
            del segments
            gc.collect()
            
            return result_text
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return None
    
    def listen_and_transcribe(
        self,
        audio_stream,
        chunk_size: int = 1024,
        max_duration: float = 8.0,
        channels: int = 1,
        on_transcription=None
    ) -> Optional[str]:
        """
        Listen and transcribe with visual feedback (dots).
        Uses the COMMAND model (small.en) for fast voice commands.
        """
        audio_buffer = []
        speech_detected = False
        start_time = time.time()
        last_dot_time = time.time()
        silence_start = None
        from jarvis_logger import emit_sys_log
        emit_sys_log("Listening...", "INFO")
        
        try:
            while True:
                now = time.time()
                
                # Visual dots while waiting
                if now - last_dot_time > 0.4:
                    from jarvis_logger import emit_sys_log
                    emit_sys_log(".", "DEBUG")
                    last_dot_time = now
                
                # Timeout if no speech starts within 8s
                if not speech_detected and (now - start_time) > max_duration:
                    emit_sys_log("no sound detected.", "DEBUG")
                    return None
                
                # Safety cutoff for very long recordings
                if speech_detected and (now - start_time) > 20.0:
                    break
                
                # Visual feedback: dots while listening (only if not speech detected)
                if not speech_detected:
                    sys.stdout.write(".")
                    sys.stdout.flush()

                # Read audio chunk
                try:
                    chunk = audio_stream.read(chunk_size, exception_on_overflow=False)
                    audio_np = np.frombuffer(chunk, dtype=np.int16)
                    
                    # Use digital gain from config (Default 15x)
                    from config import DIGITAL_GAIN
                    audio_np = (audio_np.astype(np.float32) * DIGITAL_GAIN).clip(-32768, 32767).astype(np.int16)
                    
                    # [FIX] Handle 2-channel Mac audio
                    if channels == 2:
                        audio_np = audio_np[::2]
                except:
                    break
                
                is_talking = self.is_speech(audio_np)
                
                if is_talking:
                    if not speech_detected:
                        speech_detected = True
                        sys.stdout.write("\n[STT] 👂 Heard you\n")
                        sys.stdout.flush()
                    
                    silence_start = None  # Reset silence timer while talking
                    audio_buffer.append(audio_np)
                
                elif speech_detected:
                    audio_buffer.append(audio_np)
                    if silence_start is None:
                        silence_start = now
                    
                    # Half a second of silence to finish (as requested)
                    if now - silence_start > self.silence_duration:
                        break
        
        except KeyboardInterrupt:
            return None
        
        if not audio_buffer:
            return None
        
        # Final process - uses COMMAND model (small.en)
        full_audio = np.concatenate(audio_buffer)
        text = self.transcribe(full_audio, mode="command")
        
        if text:
            text = text.strip()
            
            # --- NOISE FILTER (Anti-Hallucination) ---
            text_clean = text.lower().strip(" .!?,")
            if text_clean in self.hallucinations or len(text_clean) <= 2:
                emit_sys_log(f"noise filtered ({text_clean})", "DEBUG")
                return None
            
            emit_sys_log(f">> {text}", "INFO")
            if on_transcription:
                on_transcription(text)
            return text
        else:
            emit_sys_log("noise filtered.", "DEBUG")
            return None

    def transcribe_lecture_chunk(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Dedicated method for lecture transcription.
        Uses the TURBO model for maximum accuracy.
        Called by the note engine during active lecture sessions.
        """
        return self.transcribe(audio_data, mode="lecture")


# Singleton
stt_engine = WhisperCppEngine()
