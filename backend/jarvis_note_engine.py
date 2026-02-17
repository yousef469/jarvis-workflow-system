import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import ollama
from config import MODEL_BRAIN, MODEL_VISION
from whisper_cpp_engine import stt_engine
from vision_engine_v3 import vision_v3
from model_manager import model_manager
import base64
import numpy as np
import io
from PIL import Image
from pydub import AudioSegment
import subprocess

NOTES_DIR = Path("./saved_notes")
NOTES_DIR.mkdir(exist_ok=True)

STRUCTURED_OUTPUT_PROMPT = """Return output strictly in this JSON format:
{
  "topic": "",
  "summary": "",
  "bullet_points": [],
  "key_terms": [{"term": "", "definition": ""}],
  "flashcards": [{"q": "", "a": ""}],
  "questions": [],
  "mindmap": {}
}"""

def parse_json_from_response(text: str) -> Dict:
    try:
        # Simple extraction of anything between { and }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return {"raw": text}
    except Exception:
        return {"raw": text}

def parse_notes_tags(text: str) -> List[Dict]:
    lines = text.split('\n')
    notes = []
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('[SMALL]'):
            notes.append({"type": "small", "text": line.replace('[SMALL]', '').strip()})
        elif line.startswith('[BIG]'):
            notes.append({"type": "big", "text": line.replace('[BIG]', '').strip()})
        elif len(line) > 5:
            notes.append({"type": "small", "text": line})
            
    return notes

async def generate_structured_notes(text: str) -> Dict:
    """Uses Local LLM to generate structured notes from text."""
    prompt = f"You are an expert note generator. Convert this transcript into organized notes.\n\nTranscript:\n{text}\n\n{STRUCTURED_OUTPUT_PROMPT}"
    
    response = await asyncio.to_thread(
        ollama.chat,
        model=MODEL_BRAIN,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3}
    )
    
    return parse_json_from_response(response['message']['content'])

async def analyze_image_notes(image_b64: str, prompt_text: str) -> Dict:
    """Uses Local Vision LLM to analyze screenshots or images."""
    prompt = f"{prompt_text}\n\n{STRUCTURED_OUTPUT_PROMPT}"
    
    response = await asyncio.to_thread(
        ollama.generate,
        model=MODEL_VISION,
        prompt=prompt,
        images=[image_b64],
        stream=False
    )
    
    return parse_json_from_response(response['response'])

async def process_live_audio_chunk(audio_b64: str) -> List[Dict]:
    """Transcribes audio chunks using Whisper."""
    if _audio_lock.locked():
        return []
        
    async with _audio_lock:
        log_path = os.path.join(os.path.dirname(__file__), "audio_debug.log")
        with open(log_path, "a") as f:
            try:
                f.write(f"[{time.ctime()}] Received audio chunk, size: {len(audio_b64)}\n")
                f.flush()
                
                audio_bytes = base64.b64decode(audio_b64)
                
                # 1. Decode audio using pydub
                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                except:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                    
                f.write(f"[{time.ctime()}] Audio decoded: {len(audio)}ms, {audio.frame_rate}Hz\n")
                f.flush()
                
                # 2. Resample to 16kHz Mono as required by Whisper
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                samples = np.array(audio.get_array_of_samples())
                
                sig_min = np.min(samples)
                sig_max = np.max(samples)
                sig_mean = np.mean(np.abs(samples))
                
                f.write(f"[{time.ctime()}] Samples ready: {len(samples)} samples, min={sig_min}, max={sig_max}, abs_mean={sig_mean:.2f}\n")
                f.flush()
                
                # 3. Transcribe using FAST model for live updates
                f.write(f"[{time.ctime()}] --> Starting transcription (FAST mode)...\n")
                f.flush()
                # transcript = await asyncio.to_thread(stt_engine.transcribe_lecture_chunk, samples)
                transcript = await asyncio.to_thread(stt_engine.transcribe, samples, mode="command")
                f.write(f"[{time.ctime()}] <-- Transcription finished, length: {len(transcript) if transcript else 0}\n")
                f.flush()
                
                if not transcript or len(transcript.strip()) < 1:
                    f.write(f"[{time.ctime()}] No transcript or too short: '{transcript}'\n")
                    f.flush()
                    return []
                    
                f.write(f"[{time.ctime()}] Audio transcript: {transcript}\n")
                f.flush()
                
                # 4. Return raw transcript for live display
                return [{"type": "transcript", "text": transcript}]
                
            except Exception as e:
                f.write(f"[{time.ctime()}] Audio processing error: {e}\n")
                import traceback
                f.write(traceback.format_exc())
                f.flush()
                return []

_frame_lock = asyncio.Lock()
_audio_lock = asyncio.Lock()

async def process_live_frame(image_b64: str = None, context: str = "") -> List[Dict]:
    """
    Analyzes the current screen using Hybrid Vision V3 (Scout + Professor).
    - Scout (OCR): Always runs.
    - Professor (VLM): Runs only on change.
    """
    if _frame_lock.locked():
        print("[NoteEngine] Skipping frame: Analysis already in progress.")
        return []
        
    async with _frame_lock:
        log_path = os.path.join(os.path.dirname(__file__), "vision_debug.log")
        with open(log_path, "a") as f:
            try:
                f.write(f"[{time.ctime()}] Received frame for analysis\n")
                f.flush()
                
                # 1. Run Hybrid Analysis (Switch to Vision Model)
                f.write(f"[{time.ctime()}] --> Switching to vision model...\n")
                f.flush()
                model_manager.switch_to_vision()
                f.write(f"[{time.ctime()}] <-- Switched. Running vision_v3.analyze...\n")
                f.flush()
                
                analysis = await asyncio.to_thread(vision_v3.analyze)
                f.write(f"[{time.ctime()}] <-- Vision analysis done.\n")
                f.flush()
                
                ocr_text = analysis.get("ocr_text", "")
                vlm_analysis = analysis.get("vlm_analysis", "")
                triggered_vlm = analysis.get("triggered_vlm", False)
                
                f.write(f"[{time.ctime()}] Vision result: OCR={len(ocr_text)} chars, VLM={len(vlm_analysis)} chars, Triggered={triggered_vlm}\n")
                f.flush()
                
                # If nothing detected at all
                if not ocr_text.strip() and not vlm_analysis:
                    return []
        
                # 2. Return raw OCR text for collection (skip live reasoning)
                if not ocr_text.strip():
                    return []
                    
                return [{"type": "ocr", "text": ocr_text}]
            except Exception as e:
                f.write(f"[{time.ctime()}] Vision processing error: {e}\n")
                import traceback
                f.write(traceback.format_exc())
                f.flush()
                return []

async def finalize_session(transcript: str, visual_notes: str) -> Dict:
    """Merges all session data into a final structured note."""
    prompt = f"""You are the JARVIS Master Summarizer.
Your goal is to convert a long lecture transcript and visual screen data into PROFESSIONAL-GRADE research notes.

CORE INSTRUCTIONS:
1. Synthesize the Audio Transcript and Visual Data (OCR/VLM logs).
2. Look for "Gold" information: definitions, code snippets, complex logic, and key dates/names.
3. Organize into a logical flow. Don't just list facts—explain the *relationships* between concepts.

TRANSCRIPT:
{transcript}

VISUAL DATA:
{visual_notes}

{STRUCTURED_OUTPUT_PROMPT}"""

    response = await asyncio.to_thread(
        ollama.chat,
        model=MODEL_BRAIN,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3}
    )
    
    return parse_json_from_response(response['message']['content'])

def save_note(title: str, notes: Dict, transcript: str = "", visual_notes: str = "") -> str:
    timestamp = int(time.time())
    safe_title = "".join([c if c.isalnum() else "_" for c in title])
    filename = f"{timestamp}-{safe_title}.json"
    filepath = NOTES_DIR / filename
    
    data = {
        "title": title,
        "notes": notes,
        "transcript": transcript,
        "visual_notes": visual_notes,
        "createdAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
    return filename

def list_notes() -> List[Dict]:
    notes = []
    for f in NOTES_DIR.glob("*.json"):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                notes.append({
                    "filename": f.name,
                    "title": data.get("title", "Untitled"),
                    "createdAt": data.get("createdAt", "")
                })
        except Exception:
            pass
    return sorted(notes, key=lambda x: x['createdAt'], reverse=True)

def get_note(filename: str) -> Optional[Dict]:
    filepath = NOTES_DIR / filename
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

# Note: Full audio transcription happened via whisper_cpp_engine
