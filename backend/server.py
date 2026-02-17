import os
import sys
import types
from typing import Optional, List
import threading
import contextlib
import signal
import time
import json
import base64
import uuid

# --- 0. CRITICAL ENVIRONMENT FIXES (Before any heavy imports) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"        # Single-thread to prevent ONNX/ctranslate2 segfault on Mac ARM
os.environ["MKL_NUM_THREADS"] = "1"        # Same for MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # Same for OpenBLAS
os.environ["CT2_FORCE_CPU_ISA"] = "GENERIC"  # Stable ctranslate2 on ARM
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ----------------------------------------------------------------

# --- 1. EMERGENCY LOGGER REDIRECTION (SETUP ONLY) ---
from jarvis_logger import emit_sys_log, set_logger_refs, inject_terminal_hook
# (Injection moved below to avoid being overwritten by TextIOWrapper)
# -----------------------------------------------------

# MeCab Compatibility: Ensured unidic-lite is used (fixed dictionary path).

# Legacy LangChain Compatibility Patch
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    m_docstore = types.ModuleType('langchain.docstore')
    sys.modules['langchain.docstore'] = m_docstore
    m_doc = types.ModuleType('langchain.docstore.document')
    sys.modules['langchain.docstore.document'] = m_doc
    m_doc.Document = Document
    
    m_splitter = types.ModuleType('langchain.text_splitter')
    sys.modules['langchain.text_splitter'] = m_splitter
    m_splitter.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
except ImportError:
    pass

# MeCab Compatibility: Ensured unidic-lite is used (fixed dictionary path).

import os
import io
import functools
import time



# [MINIMAL MODE] Disable for full optimized system

os.environ["JARVIS_MINIMAL_MODE"] = "False"



# =============================================================================

# FORCE UNBUFFERED OUTPUT - See everything in real-time!

# =============================================================================



# Make print() flush immediately

print = functools.partial(print, flush=True)



if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# --- 2. ACTIVATE TERMINAL HOOK (AFTER WRAPPERS) ---
inject_terminal_hook()
# ---------------------------------------------------



emit_sys_log("J.A.R.V.I.S. UNIFIED BRAIN - STARTING UP", "INFO")

# print("[Startup] Initializing system subroutines...")

try:
    from jarvis_dispatcher_v245 import dispatcher_v245
    emit_sys_log("[Startup] Dispatcher (v24.5) loaded.", "INFO")
except Exception as e:
    emit_sys_log(f"[Critical] Failed to load V24.5 Dispatcher: {e}", "ERROR")

import json

# Global refs for state emission
_sio = None
_loop = None


# === 3. STATE EMISSION HELPER ===
def emit_state(status: str, details: str = None):
    """
    Broadcasts JARVIS's cognitive state to the UI.
    Statuses: idle, listening, thinking, speaking, processing
    """
    try:
        global socket_app
        # We need to access the internal sio instance if possible, or use the loop
        if _sio and _loop:
             _loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_sio.emit('status_update', {
                    "status": status,
                    "details": details,
                    "timestamp": time.time()
                }))
            )
             # Also log it for debug
             # emit_sys_log(f"State -> {status}", "DEBUG")
    except:
        pass

import numpy as np

import pyaudio

import re
import webbrowser
import pyautogui
import image_generator # Added for image generation support
from pdf_generator import create_medical_pdf # Added for PDF support
from config import (
    MODEL_BRAIN, MODEL_ROUTER, MODEL_EXPERT, MODEL_VISION, 
    MIC_INDEX, SAMPLE_RATE, CHUNK_SIZE as CHUNK, CHANNELS,
    WAKE_WORD_SENSITIVITY
)

import ollama
from model_manager import model_manager
import socketio
import psutil
# Already injected at top
try:
    import GPUtil
except ImportError:
    GPUtil = None

# SOCKET.IO SETUP
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

@sio.on('trigger_voice')
async def handle_trigger_voice(sid, data):
    emit_sys_log(f"[Socket] Received trigger_voice from {sid}", "INFO")
    print(f"[Socket] Triggering one-shot voice interaction from {sid}")
    # Run synchronous interaction in a thread to keep the async loop smooth
    try:
        t = threading.Thread(target=one_shot_voice_interaction, daemon=True)
        t.start()
        emit_sys_log(f"[Socket] Interaction thread started for {sid}", "DEBUG")
    except Exception as e:
        emit_sys_log(f"[Socket] ERROR starting interaction thread: {e}", "ERROR")
        print(f"[Socket] ERROR starting interaction thread: {e}")

@sio.event
async def connect(sid, environ):
    print(f"[Socket] Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[Socket] Client disconnected: {sid}")

# (Wrapper will be defined after 'app' below)

# === JARVIS v22 MASTER UPGRADE IMPORTS ===
import torch
try:
    torch.set_num_threads(4) # Increased for better performance on Mac
except:
    pass

# STT: whisper.cpp (no SileroVAD needed)
from whisper_cpp_engine import stt_engine, WhisperCppEngine

# V24.5 Unified Cognitive Path Active
CREWAI_ROUTING_AVAILABLE = False

# TTS: Jarvis Voice V4 (Piper TTS)
try:
    from piper_engine import get_piper_engine
    jarvis_voice = get_piper_engine()
    JARVIS_VOICE_AVAILABLE = True
except ImportError as e:
    JARVIS_VOICE_AVAILABLE = False
    print(f"[Warning] Piper Engine not found: {e}")

# Suppress annoying warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Memory & Tools (V24.5)
import chromadb
from pydantic import BaseModel
try:
    from ahk import AHK
except ImportError:
    pass
# =========================================

# === JARVIS v22 MASTER ENGINES (Independent Initialization) ===
startup_status = {}

def track_status(name, success, error=None):
    startup_status[name] = "✅ LOADED" if success else f"❌ FAILED ({error})" if error else "❌ FAILED"

memory_v2 = automation_v2 = vision_v2 = None
process_image_request = OpenVoiceEngine = jarvis_voice = None

# Legacy Router Removed

try:
    from memory_engine_v2 import memory_v2
    memory_engine = memory_v2 # ALIAS FOR LEGACY CODE
    track_status("Memory Engine (ChromaDB)", True)
except Exception as e:
    track_status("Memory Engine (ChromaDB)", False, str(e))

try:
    # Silence PaddleOCR noise during initialization
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from vision_engine_v3 import vision_v3 as vision_v2
    track_status("Vision Engine (OmniParser)", True)
except Exception as e:
    track_status("Vision Engine (OmniParser)", False, str(e))

# Document Engine Removed (Handled by Memory Worker)

# Legacy Image Request Handler Removed

try:
    from jarvis_voice_v3 import get_voice_v3
    jarvis_voice = get_voice_v3()
    track_status("Voice (Coqui V3)", True)
except Exception as e:
    track_status("Voice (Coqui V3)", False, str(e))

# STT Pre-warm (DUAL MODE)
# DISABLED FOR 8GB RAM OPTIMIZATION (Load on demand instead)
# def pre_warm_whisper():
#     try:
#         emit_sys_log("[Startup] Pre-warming Whisper engine (base.en)...", "INFO")
#         stt_engine._load_command_model()
#         # Deferred: stt_engine._load_lecture_model()
#         track_status("STT (Whisper Base)", True)
#     except Exception as e:
#         track_status("STT (Whisper Base)", False, str(e))
# 
# threading.Thread(target=pre_warm_whisper, daemon=True).start()

def print_startup_report():
    emit_sys_log("SYSTEM INITIALIZATION REPORT", "INFO")
    for component, status in startup_status.items():
        emit_sys_log(f"{component:<20} | {status}", "INFO")

print_startup_report()

# GLOBAL STATE (Synced with config.py)

MODEL_BRAIN = MODEL_BRAIN
MODEL_VISION = MODEL_VISION

# PRE-WARM: Load Specialists (Vision/Tools)
if CREWAI_ROUTING_AVAILABLE and crew_orchestrator:
    try:
        if vision_v2:
            emit_sys_log("[Startup] Pre-warming Vision Engine...", "DEBUG")
            vision_v2.pre_warm()
    except Exception as e:
        print(f"[Startup] Specialised pre-warm warning: {e}")

emit_sys_log("[Startup] Master Engines Ready. Systems Operational.", "INFO")


MODEL_FAST = MODEL_BRAIN
MODEL_DEEP = MODEL_BRAIN
MODEL_REASONING = MODEL_BRAIN
MODEL_AI_KNOWLEDGE = MODEL_BRAIN
MODEL_CODE = MODEL_BRAIN
MODEL_CODE_FAST = MODEL_BRAIN
MODEL_CODE_PYTHON = MODEL_BRAIN
MODEL_CODE_MULTI = MODEL_BRAIN



DRAFTING_MODEL_NAME = None

LAST_GENERATED_MODEL = None  # Stores the last successful 3D generation result

LAST_DESIGN_PARAMS = {}      # Stores the last drafted engineering parameters (e.g. detail_level)

IS_BUSY = False              # Global flag to pause listening during heavy tasks



def enforce_symmetry(design_json):

    """

    Precision Engineering Guard:

    1. Grid Snapping: Forces 5mm alignment to prevent 'shaky' look.

    2. Symmetry Correction: Fixes L/R naming collisions.

    3. Vertical Alignment: Ensures rotors are UP and legs are DOWN.

    """

    if "parts" not in design_json:

        return design_json

        

    parts_map = {}

    for i, part in enumerate(design_json["parts"]):

        # SKIP SANITY CHECKS FOR DIRECT BLUEPRINTS (God-Mode Trust)

        if design_json.get("is_direct_blueprint"):

            # Still snapped to 1mm for clean geometry, but no movement

            part["pos"] = [round(p * 1000) / 1000 for p in part.get("pos", [0,0,0])]

            continue



        name = part.get("name", "").lower()

        pos = part.get("pos", [0,0,0])

        

        # 1. GRID SNAPPING (1mm precision for flush assembly)

        part["pos"] = [round(p * 1000) / 1000 for p in pos]

        

        # 2. STRUCTURAL SANITY BIAS (Assumes Z-up for Aerospace)

        # Rotors/Masts should generally be in the upper Z-hemisphere

        if any(word in name for word in ["rotor", "blade", "mast", "solar", "antenna"]):

            if part["pos"][2] < 0: # If AI put it at bottom of Z

                part["pos"][2] = abs(part["pos"][2]) # Flip to top

                

        # Legs/Feet/Struts should generally be in the lower Z-hemisphere

        if any(word in name for word in ["leg", "foot", "strut", "gear"]):

            if part["pos"][2] > 0: # If AI put it at top of Z

                part["pos"][2] = -abs(part["pos"][2]) # Flip to bottom



        # Map potential L/R pairs

        base_name = None

        side = None

        if "_l" in name or "_left" in name:

            base_name = name.replace("_l", "").replace("_left", "")

            side = "L"

        elif "_r" in name or "_right" in name:

            base_name = name.replace("_r", "").replace("_right", "")

            side = "R"

            

        if base_name:

            if base_name not in parts_map: parts_map[base_name] = {}

            parts_map[base_name][side] = i



    # 3. Symmetry Check

    for base, sides in parts_map.items():

        if "L" in sides and "R" in sides:

            idx_l, idx_r = sides["L"], sides["R"]

            part_l, part_r = design_json["parts"][idx_l], design_json["parts"][idx_r]

            pos_l, pos_r = part_l["pos"], part_r["pos"]

            

            # If coordinates are identical (collision), force mirroring

            if sum([(a-b)**2 for a, b in zip(pos_l, pos_r)]) < 0.001:

                print(f"[3DGen] [FIX] Symmetry Guard: Mirroring {base} X-axis.")

                part_r["pos"][0] = -pos_l[0] if pos_l[0] != 0 else 0.2

                

    return design_json



def flatten_hierarchy(assembly_data):

    """Recursively flattens nested 'sub_parts' into a flat list for the generator."""

    flat_parts = []

    

    # 0. Detect God-Mode early

    is_direct = False

    project_name = "Robotic_Assembly"

    

    if isinstance(assembly_data, dict):

        # Check both top-level and potential nested 'is_direct_blueprint'

        is_direct = assembly_data.get("is_direct_blueprint", False)

        project_name = assembly_data.get("project", "Robotic_Assembly")

    elif isinstance(assembly_data, list) and len(assembly_data) > 0:

        # Check if the list contains parts that might have the flag (unlikely but safe)

        project_name = assembly_data[0].get("name", "Robotic_Assembly")



    def walk(node, parent_pos=[0,0,0], parent_name="world"):

        # 1. NORMALIZE KEYS (Handle AI hallucinations)

        p_type = node.get("type", node.get("shape", "box"))

        if isinstance(p_type, list): p_type = p_type[0]

        p_type = str(p_type).lower()

        if "_" in p_type and "hex_tile" not in p_type:

            p_type = p_type.split('_')[0]

        if p_type == "rod": p_type = "cylinder"

        

        dims = node.get("dims", node.get("dimensions", node.get("size", [0.1, 0.1, 0.1])))

        if isinstance(dims, dict):

            # ... (Map common labels: length, width, height, radius, diameter, x, y, z)

            l = dims.get("length", dims.get("l", dims.get("x", 0.1)))

            w = dims.get("width", dims.get("w", dims.get("y", 0.1)))

            h = dims.get("height", dims.get("h", dims.get("z", 0.1)))

            r = dims.get("radius", dims.get("diameter", 0.1) / 2)

            if "radius" in dims or "diameter" in dims:

                dims = [r, l]

            else:

                dims = [l, w, h]

        

        # Position normalization (Relative -> Absolute) - SUPPORT INSTANCING

        positions_list = node.get("positions", [])

        if not positions_list:

            pos = node.get("pos", node.get("coordinates", node.get("position", [0, 0, 0])))

            positions_list = [pos]



        # Handle each instance

        for i, local_pos in enumerate(positions_list):

            if isinstance(local_pos, dict):

                local_pos = [local_pos.get("x", 0), local_pos.get("y", 0), local_pos.get("z", 0)]

            

            # 3. DIRECT BLUEPRINT STABILITY: If God-Mode, we honor the hierarchy strictly.

            if is_direct:

                # We trust the provided hierarchy. NO FLATTENING to "world".

                # This allows nested rotations (tilted legs, hubs) to work correctly.

                rel_pos = [float(local_pos[0]), float(local_pos[1]), float(local_pos[2])]

                actual_parent = parent_name

                

                # Still calculate intended world pos for recursion tracking

                intended_abs_pos = [

                    parent_pos[0] + rel_pos[0],

                    parent_pos[1] + rel_pos[1],

                    parent_pos[2] + rel_pos[2]

                ]

            else:

                # LLM AI Drafting Path (Probabilistic / Unreliable Hierarchy)

                # We use the Smart Fusion guesser to recover absolute positions from mangled drafts.

                # Here we DO flatten to 'world' to prevent the recursive line bug if hierarchy is garbage.

                intended_abs_pos = [

                    parent_pos[0] + float(local_pos[0]),

                    parent_pos[1] + float(local_pos[1]),

                    parent_pos[2] + float(local_pos[2])

                ]

                

                is_absolute = False

                for j in range(3):

                    if local_pos[j] != 0 and abs(float(local_pos[j]) - float(parent_pos[j])) < 0.05:

                        is_absolute = True

                        break

                

                dist_to_parent = np.linalg.norm(np.array(local_pos) - np.array(parent_pos))

                dist_to_origin = np.linalg.norm(np.array(local_pos))

                

                if is_absolute or (dist_to_origin > 0.2 and dist_to_parent < dist_to_origin * 0.5):

                    intended_abs_pos = [float(local_pos[0]), float(local_pos[1]), float(local_pos[2])]

                    rel_pos = intended_abs_pos

                else:

                    rel_pos = intended_abs_pos

                

                actual_parent = "world"



            p_name = node.get("name", f"part_{len(flat_parts)}")

            if len(positions_list) > 1:

                p_name = f"{p_name}_{i+1}"



            part = {

                "name": p_name,

                "type": p_type,

                "dims": dims,

                "pos": rel_pos, 

                "rot": node.get("rot", [0, 0, 0]),

                "material": node.get("material", "default"),

                "subsystem": node.get("subsystem", "structures"),

                "parent": actual_parent

            }



            # --- ARRAY INSTANCING PASSTHROUGH ---

            for ak in ["radial_array", "linear_array", "grid_array"]:

                if ak in node:

                    part[ak] = node[ak]



            flat_parts.append(part)

            

            # Recurse through children

            children = node.get("sub_parts", [])

            for child in children:

                walk(child, parent_pos=intended_abs_pos, parent_name=part["name"])

            

    # Handle both top-level list and nested 'assembly' object

    if isinstance(assembly_data, list):

        for p in assembly_data: walk(p)

    elif isinstance(assembly_data, dict):

        if "parts" in assembly_data:

            for p in assembly_data["parts"]: walk(p)

        elif "assembly" in assembly_data:

            walk(assembly_data["assembly"])

        elif "project" in assembly_data: # Implicit root

             if "type" in assembly_data or "shape" in assembly_data:

                 walk(assembly_data)

             elif "sub_parts" in assembly_data:

                 for p in assembly_data["sub_parts"]: walk(p)



    return {

        "project": project_name, 

        "parts": flat_parts,

        "is_direct_blueprint": is_direct

    }



def detect_drafting_model():

    """Hardcoded for 7B Single-Brain Architecture."""

    global DRAFTING_MODEL_NAME

    DRAFTING_MODEL_NAME = MODEL_BRAIN

    print("\n" + "!"*60)

    print(f"  [ROCKET] UNIFIED-BRAIN MODE ACTIVATED")

    print(f"  - Single Model: {MODEL_BRAIN} (All Tasks)")

    print(f"  - Status:       UNIFIED BRAIN ACTIVATED (7B)")

    print("!"*60 + "\n")



# Run detection immediately

detect_drafting_model()



def set_busy(state: bool):

    """Pauses/Resumes the main listening ear during heavy processing."""

    global IS_BUSY

    IS_BUSY = state

    if state:

        print("[System] [LOCKED] Heavy operation detected. Pausing ear system...")

    else:

        print("[System] [UNLOCKED] Operation complete. Resuming ear system...")

import random

import subprocess

from pathlib import Path

from datetime import datetime

import shutil

import uuid



print("[Startup] Loading core models (this may take a few moments)...")



# Optional Features Tracking

WAKE_WORD_ENABLED = False

WHISPER_ENABLED = False

SCREENSHOT_ENABLED = False

AUTOMATION_ENABLED = False

JARVIS_VOICE_ENABLED = False

PHYSICS_ENABLED = False



# === JARVIS Unified Interaction Engine ===
def start_unified_interaction():
    """
    The main interaction loop (FULLY SYNCHRONOUS):
    1. Listen for 'Hey Jarvis' (OpenWakeWord)
    2. Greet user (Piper Ryan High)
    3. Transcription (Whisper)
    4. Full Cognitive Pipeline via dispatcher (sync)
    5. Vocal feedback (Piper Ryan High)
    """
    from wake_word import listen_for_wake_word, get_pyaudio
    from whisper_cpp_engine import stt_engine
    from jarvis_dispatcher_v245 import dispatch_cognitive_sync
    from piper_engine import get_piper_engine
    
    voice = get_piper_engine()
    
    print("[UnifiedEngine] ✅ Voice pipeline ready (Wake Word → STT → Brain → Piper Ryan)")
    
    while True:
        try:
            # 1. Wake Word detection (blocks until "Hey Jarvis" heard)
            if not listen_for_wake_word():
                time.sleep(0.5)
                continue
            
            emit_state("listening", "Listening for command...")
            voice.stop()  # Stop any active speech
            
            # 2. Greet user (Piper Ryan High)
            # Use JARVIS_GREETING from config for consistency
            from config import JARVIS_GREETING
            voice.speak(JARVIS_GREETING, blocking=True)
            
            # 3. Open Mic and Transcribe
            mic = get_pyaudio()
            time.sleep(0.2)  # Let hardware settle
            
            stream = None
            text = None
            try:
                stream = mic.open(
                    format=pyaudio.paInt16,
                    channels=2,
                    rate=16000,
                    input=True,
                    input_device_index=MIC_INDEX,
                    frames_per_buffer=1024
                )
                text = stt_engine.listen_and_transcribe(stream, channels=2, max_duration=20.0)
            except Exception as e:
                print(f"[UnifiedEngine] Mic/STT error: {e}")
            finally:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except:
                        pass
            
            if not text:
                emit_state("idle", "No command heard")
                continue
            
            print(f"[UnifiedEngine] Heard: {text}")
            emit_state("thinking", f"Processing: {text}")
            
            # 4. Full Cognitive Pipeline (SYNC - brain + worker + review + memory)
            try:
                result = dispatch_cognitive_sync(text)
                response = result.get("response", "I'm sorry sir, I couldn't process that.")
                worker = result.get("worker", "none")
                print(f"[UnifiedEngine] Pipeline complete: worker={worker}")
            except Exception as e:
                print(f"[UnifiedEngine] Pipeline error: {e}")
                response = "I'm sorry sir, I encountered an issue processing that."
            
            # 5. Speak response with Piper Ryan High voice
            print(f"[UnifiedEngine] Response: {response[:80]}...")
            emit_state("speaking", response)
            voice.speak(response)
            
            # Post-speech safety sleep + buffer reset
            time.sleep(0.5)
            reset_wake_word_model()
            
            emit_state("idle", "Ready")
                
        except KeyboardInterrupt:
            print("[UnifiedEngine] Shutting down...")
            break
        except Exception as e:
            print(f"[UnifiedEngine] Loop error: {e}")
            time.sleep(1)
            reset_wake_word_model() # Also reset on error to clear buffer

def one_shot_voice_interaction():
    """Performs a single STT -> Brain -> TTS interaction sequence."""
    from wake_word import get_pyaudio
    from whisper_cpp_engine import stt_engine
    from jarvis_dispatcher_v245 import dispatch_cognitive_sync
    # TTS REMOVED as per user request
    
    try:
        emit_state("listening", "Listening for command...")
        
        # 1. Greet user (Visual only/Status)
        # voice.speak(JARVIS_GREETING, blocking=True) # REMOVED
        
        # 2. Open Mic and Transcribe
        mic = get_pyaudio()
        time.sleep(0.2) # Hardware settle
        
        stream = None
        text = None
        try:
            stream = mic.open(
                format=pyaudio.paInt16,
                channels=CHANNELS, # Use CHANNELS from config (usually 1 or 2)
                rate=16000,
                input=True,
                input_device_index=MIC_INDEX,
                frames_per_buffer=1024
            )
            text = stt_engine.listen_and_transcribe(stream, channels=CHANNELS, max_duration=20.0)
        except Exception as e:
            print(f"[TriggerVoice] Mic/STT error: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
        
        if not text:
            emit_state("idle", "No command heard")
            return
            
        print(f"[TriggerVoice] Heard: {text}")
        # EMIT transcription to UI immediately
        if _sio and _loop:
            _loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_sio.emit('transcription', {"text": text}))
            )
            
        emit_state("thinking", f"Processing: {text}")
        
        # 3. Brain Pipeline
        try:
            result = dispatch_cognitive_sync(text)
            response = result.get("response", "I'm sorry sir, I couldn't process that.")
            worker_res = result.get("worker_result")
        except Exception as e:
            print(f"[TriggerVoice] Pipeline error: {e}")
            response = "I'm sorry sir, I encountered an issue."
            worker_res = None
            
        # 4. Speak Response (REMOVED)
        # emit_state("speaking", response)
        # voice.speak(response)
        
        # 5. Emit final response back to UI via Socket.io
        if _sio and _loop:
            _loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_sio.emit('assistant_response', {
                    "text": response,
                    "worker_result": worker_res
                }))
            )
        
        time.sleep(0.5)
        emit_state("idle", "Ready")
        
    except Exception as e:
        print(f"[TriggerVoice] Fatal error: {e}")
        emit_state("idle", "Ready")

# signal_handler and other utilities...
import traceback
import sys
import signal
import os
import contextlib
import socketio


def signal_handler(sig, frame):
    print(f"\n[Backend] 🛑 Received signal {sig}. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
except ImportError:
    print("[Error] FastAPI not found. Please run 'pip install -r requirements.txt'")
    sys.exit(1)
from fastapi.responses import StreamingResponse
import jarvis_note_engine

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles

import uvicorn

import asyncio

import base64



# Import JARVIS voice and knowledge
try:
    from jarvis_knowledge import JarvisKnowledgeBase, get_jarvis_response
    JARVIS_VOICE_ENABLED = JARVIS_VOICE_AVAILABLE
except ImportError:
    JARVIS_VOICE_ENABLED = False




# [MINIMAL MODE] Physics and 3D are kept

try:

    from physics_engine import FlightSimulator6DOF

    PHYSICS_ENABLED = True

except ImportError:

    print("[Warning] Physics engine not available")

    PHYSICS_ENABLED = False



# [MINIMAL MODE] Automation disabled

AUTOMATION_ENABLED = False

# try:

#     from automation_engine import init_scheduler, parse_automation, create_automation, execute_automation, shutdown_scheduler

#     from llm_client import VisionClient

#     AUTOMATION_ENABLED = True

# except ImportError:

#     AUTOMATION_ENABLED = False



# FastAPI App Initialization

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    if AUTOMATION_ENABLED:
        try: init_scheduler()
        except: pass
    
    # Start Telemetry Worker
    telemetry_task = asyncio.create_task(telemetry_worker())
    print("[Startup] Telemetry Broadcast Started.")

    if not os.path.exists("generated_models"):
        os.makedirs("generated_models")

    # Set global refs for state emission
    global _sio, _loop
    _sio = sio
    _loop = asyncio.get_event_loop()
    
    # [DISABLED] Continuous Wake Word Listener (Moved to manual trigger in UI)
    # interaction_thread = threading.Thread(target=start_unified_interaction, daemon=True)
    # interaction_thread.start()
    # print("[Startup] JARVIS Unified Interaction Started.")
    print("[Startup] JARVIS Push-to-Talk (Manual Trigger) Enabled.")

    yield
    
    # Shutdown logic
    telemetry_task.cancel()
    if AUTOMATION_ENABLED:
        try: shutdown_scheduler()
        except: pass
    print("[Shutdown] JARVIS Backend Services going offline.")



app = FastAPI(
    title="JARVIS Unified Backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SOCKET.IO BRIDGE (Unified Port)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# TELEMETRY & INTROSPECTION TOOLS
def emit_truth(intent, reasoning, status="success", details=None):
    """Global hook to emit introspection data to Section 1 (Truth Panel)"""
    import asyncio
    import uuid
    from datetime import datetime
    try:
        data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "intent": intent,
            "reasoning": reasoning,
            "status": status,
            "details": details or {}
        }
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(sio.emit('truth', data), loop)
    except Exception as e:
        print(f"[Socket] Error emitting truth: {e}")

# Logger is now in jarvis_logger.py

@app.get("/api/knowledge/list")
async def get_all_knowledge():
    """Fetches all memories/facts/lessons from ChromaDB"""
    from jarvis_brain_v245 import brain_v245
    
    memories = []
    
    def safe_get(collection, context_name, type_map):
        if not collection: return []
        try:
            items = collection.get()
            results = []
            for i in range(len(items['ids'])):
                # Parse metadata if needed
                meta = items['metadatas'][i] if items['metadatas'] and i < len(items['metadatas']) else {}
                ts = meta.get('timestamp', 0) if meta else 0
                
                results.append({
                    "id": items['ids'][i],
                    "context": context_name,
                    "lesson": items['documents'][i],
                    "type": type_map,
                    "confidence": 1.0, # Stored memories are trusted
                    "lastUsed": "Recent" if (time.time() - float(ts) < 86400) else "Archived",
                    "locked": context_name in ["fact", "preference"] # Lock core knowledge
                })
            return results
        except:
            return []

    try:
        # 1. Facts (Success/Green)
        memories.extend(safe_get(brain_v245.facts, "fact", "success"))
        
        # 2. Lessons (Workflow/Blue)
        memories.extend(safe_get(brain_v245.lessons, "lesson", "workflow"))
        
        # 3. Preferences (Workflow/Blue)
        memories.extend(safe_get(brain_v245.preferences, "preference", "workflow"))
        
        # 4. Mistakes (Failure/Red)
        memories.extend(safe_get(brain_v245.mistakes, "mistake", "failure"))
        
        # 5. Assets (Success/Green)
        memories.extend(safe_get(brain_v245.assets, "asset", "success"))
        
        # 6. Action History (Workflow/Blue) - Limit to latest 50
        actions = safe_get(brain_v245.actions, "action", "workflow")
        memories.extend(actions[-50:]) # Only show recent actions to prevent bloat or reverse it

        return {"success": True, "memories": memories}
        
    except Exception as e:
        print(f"[API] Knowledge fetch error: {e}")
        return {"success": False, "error": str(e), "memories": []}
        
        # 2. Lessons
        if brain_v245.lessons:
            lessons = brain_v245.lessons.get()
            for i in range(len(lessons['ids'])):
                memories.append({
                    "id": lessons['ids'][i],
                    "context": "lesson",
                    "lesson": lessons['documents'][i],
                    "type": "workflow",
                    "confidence": 0.88,
                    "lastUsed": "Recent",
                    "locked": False
                })
        
        # 3. Mistakes
        if brain_v245.mistakes:
            mistakes = brain_v245.mistakes.get()
            for i in range(len(mistakes['ids'])):
                memories.append({
                    "id": mistakes['ids'][i],
                    "context": "correction",
                    "lesson": mistakes['documents'][i],
                    "type": "failure",
                    "confidence": 0.99,
                    "lastUsed": "Critical",
                    "locked": False
                })
                
        return {"success": True, "memories": memories}
    except Exception as e:
        print(f"[API] Knowledge fetch error: {e}")
        return {"success": False, "error": str(e)}

async def telemetry_worker():
    """Background task to broadcast real PC stats to Section 2 (Computer Hub)"""
    import asyncio
    import time
    while True:
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            gpu_load = 0
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus: gpu_load = gpus[0].load * 100
                except: pass
            
            # Fetch top 4 processes by CPU
            processes = []
            try:
                for proc in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']), 
                                 key=lambda x: x.info['cpu_percent'] or 0, reverse=True)[:4]:
                    processes.append({
                        "id": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu": f"{proc.info['cpu_percent']:.1f}%",
                        "mem": f"{proc.info['memory_info'].rss / (1024*1024):.0f}MB",
                        "status": "accent" if proc.info['cpu_percent'] > 5 else "normal"
                    })
            except: pass

            await sio.emit('telemetry', {
                "cpu": cpu,
                "ram": ram,
                "disk": disk,
                "gpu": gpu_load,
                "processes": processes,
                "timestamp": time.time()
            })
        except Exception as e:
            pass
        await asyncio.sleep(1) # Faster updates for "Live" feel

@sio.event
async def connect(sid, environ):
    print(f"[Socket] Client connected: {sid}")

# Mount static directory for 3D models

if not os.path.exists("generated_models"):

    os.makedirs("generated_models")

app.mount("/models", StaticFiles(directory="generated_models"), name="generated_models")
app.mount("/images", StaticFiles(directory="generated_images"), name="images")





@app.websocket("/ws/simulation")

async def websocket_simulation_endpoint(websocket: WebSocket):

    """

    Real-time Physics Streaming for Unity

    """

    await websocket.accept()

    print("[Simulation] Unity Client Connected")

    

    if not PHYSICS_ENABLED:

        await websocket.send_json({"error": "Physics engine module not found"})

        await websocket.close()

        return



    # Initialize Simulation

    sim = FlightSimulator6DOF({"total_mass_kg": 5000})

    

    try:

        while True:

            # 1. Update Physics

            data = sim.update()

            

            # 2. Send Data to Unity

            await websocket.send_json({

                "type": "telemetry",

                "payload": data

            })

            

            # Target ~60Hz

            await asyncio.sleep(0.016) 

            

    except WebSocketDisconnect:

        print("[Simulation] Unity Client Disconnected")

    except Exception as e:

        print(f"[Simulation] Error: {e}")

        try:

            await websocket.close()

        except:

            pass





# Load LLM Client (Lazy loaded for vision)

llm_client = VisionClient() if AUTOMATION_ENABLED else None



# [MINIMAL MODE] Memory Disabled

MEMORY_ENABLED = False

memory_collection = None

embedder = None



# Initialize memory v22
if MEMORY_ENABLED:
    try:
        # Use the V22 Persistent Memory
        from memory_engine_v2 import memory_v2
        memory_collection = memory_v2.facts
        print(f"[Memory] [v22] ChromaDB Persistent Storage Ready ({memory_collection.count()} memories)")
    except Exception as e:
        print(f"[Memory] v22 Initialization failed: {e}")
        MEMORY_ENABLED = False



# [MINIMAL MODE] Image Gen, Vision, Hand, Device disabled

IMAGE_GEN_ENABLED = False

ENHANCED_VISION_ENABLED = False

HAND_VISION_ENABLED = False

DEVICE_TRANSFER_ENABLED = False

try:

    from image_generator import generate_image, quick_image, unload_model as unload_image_model

    IMAGE_GEN_ENABLED = True
    print("[Server] Image Generator Module ENABLED")

except ImportError:

    IMAGE_GEN_ENABLED = False
    print("[Server] Image Generator Module DISABLED (Missing dependencies)")

# 

# try:

#     from vision_engine import enhanced_vision, format_for_brain, cleanup as cleanup_vision

#     ENHANCED_VISION_ENABLED = True

# except ImportError as e:

#     print(f"[Warning] Enhanced vision not available: {e}")

#     ENHANCED_VISION_ENABLED = False

# 

# try:

#     from hand_vision import HandVision, Gesture

#     HAND_VISION_ENABLED = True

# except ImportError as e:

#     print(f"[Warning] Hand vision not available: {e}")

#     HAND_VISION_ENABLED = False

# 

# try:

#     from device_transfer import DeviceTransfer

#     DEVICE_TRANSFER_ENABLED = True

# except ImportError as e:

#     print(f"[Warning] Device transfer not available: {e}")

#     DEVICE_TRANSFER_ENABLED = False



# Try to import Structural 3D Generator

try:

    from three_d_generator import Structural3DGenerator

    STRUCTURAL_3D_ENABLED = True

    three_d_gen = Structural3DGenerator()

except ImportError as e:

    print(f"[Warning] Structural 3D generator not available: {e}")

    STRUCTURAL_3D_ENABLED = False



print(f"[Startup] Structural 3D Modeler: {'ENABLED' if STRUCTURAL_3D_ENABLED else 'DISABLED'}")



# Try to import Auto Discovery

try:

    from auto_discovery import DeviceDiscovery, SimpleImageReceiver

    AUTO_DISCOVERY_ENABLED = True

except ImportError as e:

    print(f"[Warning] Auto discovery not available: {e}")

    AUTO_DISCOVERY_ENABLED = False



# Global instances

hand_vision_instance = None

device_transfer_instance = None

device_discovery_instance = None



# Configuration

SAMPLE_RATE = 16000

CHUNK = 1024

WHISPER_MODEL = "small"  # Upgraded from "base" for better accuracy



# =============================================================================

# SPECIALIST AI MODELS - CLEAN MODULAR ARCHITECTURE

# =============================================================================

# 

# 🟢 CORE BRAIN (ALWAYS LOADED) - Phi-3 Mini 3.8B

#    - Handles: Knowledge, Reasoning, Routing, Decision making, Natural language

#    - Response time: 1-3 seconds

#    - RAM: ~2.2GB (stays in memory permanently)

#

# 🔵 PYTHON TOOLS (LIGHT, FAST, SMART)

#    - Vision/OCR: Pillow, OpenCV, EasyOCR (replaces heavy vision models)

#    - System: Open apps, control files, browser, OS commands

#    - Python executes → Phi-3 decides

#

# 🟠 COMPLEX QUESTIONS - Phi-3 + Web Search

#    - Phi-3 tries first, if unsure → searches web for info

#    - No heavy models needed! Saves ~4GB RAM

#

# =============================================================================




# =============================================================================
# MODEL MANAGER INTEGRATION
# =============================================================================

def load_brain():
    """Wrapper for ModelManager - ensures brain is loaded"""
    model_manager.switch_to_brain()
    return True
    
def force_unload_model():
    """Wrapper for ModelManager - ensures heavy models unloaded"""
    model_manager.switch_to_brain()

# Helper for compatibility
def load_model(model_name):
    if model_name == model_manager.MODEL_BRAIN: # Use manager constants if possible or global
        model_manager.switch_to_brain()
    elif model_name == model_manager.MODEL_VISION:
        model_manager.switch_to_vision()
    # If using globals from this file
    elif model_name == "qwen3:4b":
         model_manager.switch_to_brain()
    elif model_name == "qwen3-vl:4b":
         model_manager.switch_to_vision()
    else:
         # Fallback for other models
         model_manager._preload_model(model_name)
    return True

# Ensure Startup Load
model_manager.switch_to_brain()

# Cleanup old globals if mistakenly referenced
_current_loaded_model = None


# VAD & AUDIO SETTINGS

# =============================================================================

SILENCE_THRESHOLD = 30 # Drastically lowered to catch faint speech
SILENCE_DURATION = 1.0 # Wait 1 full second of silence before stopping (Fixes "cutting off")
MAX_RECORD_TIME = 10 # Allow up to 10 seconds of speech
CONVERSATION_TIMEOUT = 3600 # Set to 1 hour to "remove" the standby timer



# [MINIMAL MODE] Lightweight components only

print("[Startup] Initializing JARVIS (MINIMAL 3D MODE)...")



# === JARVIS v22 STARTUP SEQUENCE ===
print("[Startup] Bypassing legacy initialization. V22 Engines Active...")

# STT (Whisper.cpp) and TTS (OpenVoice) are initialized at the top level.
WHISPER_ENABLED = True # v22 uses whisper.cpp
WAKE_WORD_ENABLED = True
JARVIS_VOICE_ENABLED = True # v22 uses OpenVoice
try:
    from openwakeword.model import Model as WakeModel
    wake_model = WakeModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    print("[Startup] Wake word engine (OpenWakeWord) initialized.")
except Exception as e:
    print(f"[Startup] Wake word failed: {e}")
    wake_model = None
    WAKE_WORD_ENABLED = False

MEMORY_ENABLED = True
print("[Startup] Memory system ACTIVE.")


# Load brain model at startup - stays in RAM forever

# print("[Startup] Loading brain model (permanent)...")

load_brain()



# Quick speed test to verify performance

# print("[Startup] Running speed test...")
# test_start = time.time()
# try:
#     test_resp = ollama.chat(model=MODEL_BRAIN, messages=[
#         {"role": "system", "content": "Reply in 5 words max."},
#         {"role": "user", "content": "Hello"}
#     ], options={"temperature": 0.1, "num_predict": 10, "num_ctx": 256}, keep_alive=-1)
#     test_time = time.time() - test_start
#     print(f"[Startup] Speed test: {test_time:.2f}s (target: < 2s)")
# except:
#     pass



print("[Startup] Ready! Brain loaded, heavy models on-demand")



# =============================================================================

# HELPER FUNCTIONS

# =============================================================================

def generate_asset_title(concept):
    """Generates a short, descriptive title for a file (v19/v21)"""
    prompt = f"Summarize this concept into a short 4-5 word file title (no extension): {concept}"
    try:
        response = ollama.chat(model=MODEL_BRAIN, messages=[{'role': 'user', 'content': prompt}])
        title = response['message']['content'].strip().strip('"').replace(" ", "_").lower()
        return title[:50] # Safety limit
    except:
        return concept.replace(" ", "_")[:30]


# =============================================================================

# JARVIS PERSONALITY - Generates ALL responses as JARVIS character

# =============================================================================



JARVIS_SYSTEM_PROMPT = """You are JARVIS, Tony Stark's AI assistant. 

Tone:

- British sophistication, wit, and extreme efficiency.

- Polite, confident, slightly formal, and proactive.

- Technical when needed, but never speculative.



Rules:

- Reply in ONE concise, direct sentence (max 20 words).

- Address the user as "sir" occasionally.

- If asked for info, answer directly. If unclear, ask ONE clarifying question.

- No emojis, no filler phrases, no "As an AI...".

- If an action is implied, suggest the next step.

- Focus: Engineering, structural stability, and system status."""



def query_gemini(prompt, system_prompt=JARVIS_SYSTEM_PROMPT):
    """
    Unified brain query function (using shared local client)
    """
    from shared_clients import shared_brain
    return shared_brain.chat(prompt, system_prompt=system_prompt)

def play_activation_sound():
    """Plays the cinematic 'link' sound."""
    print("\a", end="") # Native system beep
    print("[Sound] 🔊 Linked.")

def play_deactivation_sound():
    """Plays the 'offline' sound."""
    print("[Sound] 🔇 Standby.")

def jarvis_reply(context: str, task_type: str = "general") -> str:

    """

    Generate a JARVIS-style response using Phi3 brain.

    For speed, uses pre-cached responses for common actions.

    """

    # FAST PATH: Use cached responses for common actions (< 0.1 sec)

    fast_responses = {

        "greeting": ["Yes sir?", "At your service, sir.", "How may I assist?", "Ready, sir."],

        "ack_action": ["Right away, sir.", "On it, sir.", "Certainly, sir.", "As you wish, sir."],

        "done_action": ["Done, sir.", "Complete, sir.", "Finished, sir.", "All done, sir."],

        "thinking": ["One moment, sir.", "Processing, sir.", "Working on it, sir."],

        "error": ["My apologies, sir.", "I'm afraid there was an issue, sir."],

    }

    

    if task_type in fast_responses:

        response = random.choice(fast_responses[task_type])

        return response

    

    # SLOW PATH: Generate with Phi3 for custom responses

    prompts = {

        "general": context,

    }

    

    prompt = prompts.get(task_type, context)

    

    try:

        resp = ollama.chat(model=MODEL_BRAIN, messages=[

            {"role": "system", "content": JARVIS_SYSTEM_PROMPT},

            {"role": "user", "content": prompt}

        ], options={

            "temperature": 0.7, 

            "num_predict": 30,  # Shorter for speed

            "num_ctx": 256,     # Minimal context

        }, keep_alive=-1)

        response = resp["message"]["content"].strip()

        print(f"[Phi3] → \"{response}\"")

        return response

    except:

        return "Yes sir?"

def speak(text, blocking=True):
    """Text-to-speech with Coqui V3 (Cache -> Play or Text -> Background Gen)"""
    print(f"[Speak] -> {text}", flush=True) 
    
    if jarvis_voice:
        # speak() returns True if audio played, False if queued/missing
        emit_state("speaking", text)
        try:
            played = jarvis_voice.speak(text, blocking=blocking)
        finally:
            emit_state("idle")

        if not played:
            print(f"[Speak] 🔇 Cache Miss. Displaying text only. Building voice in background...")
    else:
        print("[Speak] ❌ Voice Engine not available.")




def save_to_memory(user_text, response):

    """Save interaction to memory"""

    if not MEMORY_ENABLED or not memory_collection:

        return

    try:

        embedding = embedder.encode(user_text).tolist()

        memory_collection.add(

            documents=[f"User: {user_text}\nJarvis: {response}"],

            embeddings=[embedding],

            ids=[f"mem_{int(time.time()*1000)}"]

        )

    except Exception as e:

        print(f"[Memory] Save error: {e}")



def query_memory(text, top_k=2):

    """Get relevant context from memory"""

    if not MEMORY_ENABLED or not memory_collection or memory_collection.count() == 0:

        return ""

    try:

        embedding = embedder.encode(text).tolist()

        results = memory_collection.query(query_embeddings=[embedding], n_results=top_k)

        if results['documents'] and results['documents'][0]:

            return "\n".join(results['documents'][0])

    except:

        pass

    return ""



# Track last action for corrections

last_action_context = {"text": "", "response": "", "time": 0}



# =============================================================================

# PYTHON ROUTER - INSTANT keyword matching, NO AI for routing!

# =============================================================================

# 

# SPEED TARGETS:

# - Actions (open, search, click): < 1 sec (Python only)

# - Simple Q&A: < 3 sec (Phi-3)

# - Complex/search: < 5 sec (Phi-3 + web)

# - Image generation: < 30 sec (SD)

# - Vision: < 10 sec (VL model)

#

# =============================================================================



def process_vision(text):

    """

    Enhanced Vision - Small VL + Python helpers for low-RAM, high-quality analysis.

    

    Pipeline:

    1. Phi-3 acknowledges (in character)

    2. Screenshot + OCR + UI detection (Python helpers)

    3. Small VL model for rough scene understanding

    4. Combine all sources into structured data

    5. Phi-3 interprets and responds (in character)

    """

    # 1. Phi-3 acknowledges

    ack_response = jarvis_reply("analyzing the screen", "thinking")

    speak(ack_response)

    

    # 2-4. Enhanced vision analysis

    if ENHANCED_VISION_ENABLED:

        print("[Vision] Using enhanced vision engine...")

        try:

            # Get structured analysis

            analysis = enhanced_vision(text)

            context = format_for_brain(analysis)

            

            print(f"[Vision] Analysis: {context[:200]}...")

            

            # 5. Phi-3 interprets the structured data

            interpretation_prompt = f"""Based on this screen analysis, answer the user's question.



User question: {text}



Screen analysis: {context}



Give a helpful, concise answer as JARVIS."""

            

            resp = ollama.chat(model=MODEL_BRAIN, messages=[

                {"role": "system", "content": "You are JARVIS. Interpret screen analysis data and answer concisely."},

                {"role": "user", "content": interpretation_prompt}

            ], options={"temperature": 0.3, "num_predict": 150}, keep_alive=-1)

            

            # Cleanup temp files

            cleanup_vision()

            

            return {"response": resp["message"]["content"].strip()}

            

        except Exception as e:

            print(f"[Vision] Enhanced vision error: {e}")

            cleanup_vision()

    

    # Fallback: Direct VL model (old method)

    print("[Vision] Fallback to direct VL...")

    try:

        import pyautogui

        screenshot = pyautogui.screenshot()

        screenshot.save("_temp_screenshot.png")

        

        if not load_model(MODEL_VISION):

            os.remove("_temp_screenshot.png")

            return {"response": jarvis_reply("vision module not available", "error")}

        

        resp = ollama.chat(model=MODEL_VISION, messages=[

            {"role": "user", "content": text, "images": ["_temp_screenshot.png"]}

        ], options={"temperature": 0.2, "num_predict": 200})

        

        os.remove("_temp_screenshot.png")

        force_unload_model()

        return {"response": resp["message"]["content"].strip()}

        

    except Exception as e:

        print(f"[Vision] Error: {e}")

        force_unload_model()

        if os.path.exists("_temp_screenshot.png"):

            os.remove("_temp_screenshot.png")

        return {"response": jarvis_reply("had trouble analyzing the screen", "error")}



def try_repair_json(s):

    """Ultra-robust attempt to salvage partial or broken JSON engineering drafts."""

    if not s: return ""

    

    # 1. TRACE START: Find the first actual JSON object start

    start = s.find('{')

    if start == -1: return ""

    s = s[start:]

    

    # 2. STRIP TRAILING NOISE: Find the LAST structural character before any markdown closing

    # We look for the last '}' that could potentially balance our start '{'

    last_brace = s.rfind('}')

    if last_brace != -1:

        # Check if there are other objects starting after it? No, find the most balanced end

        # Aggressive: trim everything after the last '}'

        s = s[:last_brace+1]



    # 3. HANDLE TRUNCATION: If we are still unbalanced

    if s.count('"') % 2 != 0:

        s += '"' # Close dangling quote

        

    s = s.strip()



    # 4. BALANCE: Close all open brackets/braces

    braces = s.count('{') - s.count('}')

    brackets = s.count('[') - s.count(']')

    

    if brackets > 0: s += ']' * brackets

    if braces > 0: s += '}' * braces

    

    return s



def heal_json(s):

    """Aggressive Snap-to-Schema alignment for AI drafts with diagnostic logging."""

    if not s: return s

    

    # Pre-clean: Remove MD blocks if they encase the JSON

    if "```json" in s:

        match = re.search(r'```json\s*(\{.*?\})\s*```', s, re.DOTALL)

        if match: s = match.group(1)

        else: s = s.split("```json")[1].split("```")[0]

    

    # 0. Initial Repair (Close partial JSONs)

    s = try_repair_json(s)

    if not s: return ""

    

    # 1. Remove comments

    s = re.sub(r'//.*', '', s)

    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

    

    # 2. Fix unquoted keys (Only if preceded by whitespace, { or , to avoid corrupting strings)

    s = re.sub(r'([{,]\s*|\n\s*)(\w+)\s*:', r'\1"\2":', s)



    # 3. SNAP-TO-SCHEMA: Fix mutated keys (nameiname -> name)

    s = re.sub(r'"id"\s*:', r'"name":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*name[a-zA-Z0-9_]*"\s*:', r'"name":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*type[a-zA-Z0-9_]*"\s*:', r'"type":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*dims[a-zA-Z0-9_]*"\s*:', r'"dims":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*pos[a-zA-Z0-9_]*"\s*:', r'"pos":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*rot[a-zA-Z0-9_]*"\s*:', r'"rot":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*material[a-zA-Z0-9_]*"\s*:', r'"material":', s, flags=re.IGNORECASE)

    s = re.sub(r'"[a-zA-Z0-9_]*subsystem[a-zA-Z0-9_]*"\s*:', r'"subsystem":', s, flags=re.IGNORECASE)

    

    # 4. Handle specific coordinate patterns & hallucinations

    # [0, -1, 0] brackets hallucinations

    s = re.sub(r'":\s*"\[([0-9.-]+,\s*[0-9.-]+,\s*[0-9.-]+)\]"\]', r'": [\1]', s)

    s = re.sub(r'":\s*"\[([0-9.-]+,\s*[0-9.-]+,\s*[0-9.-]+)\]"', r'": [\1]', s)

    s = re.sub(r'"in"\s*:\s*\[([0-9.-]+,\s*[0-9.-]+,\s*[0-9.-]+)\]\s*\]', r'"pos": [\1]', s)

    s = re.sub(r'"name[0-9]+"\s*:', r'"name":', s)

    

    # 5. Fix nested dims hallucination

    s = re.sub(r'\[\s*\[([0-9.-]+),\s*([0-9.-]+)\]\s*,\s*\[([0-9.-]+),\s*([0-9.-]+)\]\s*\]', r'[\1, \2, \3]', s)



    # 6. Final sanitation for trailing commas and balance

    s = re.sub(r',\s*([\]}])', r'\1', s)

    s = re.sub(r'\]\s*\]\s*\}', r'] }', s)

    

    return s.strip()



async def process_3d_generation(text, image_path=None, broadcast_progress=None, unity_path=None, params=None):

    """

    Hierarchical 3D Structural Generation.

    Uses Phi-3 (and LLaVA if image) to create a blueprint, then Structural3DGenerator builds it.

    """

    if params is None: params = {}

    

    set_busy(True) # LOCK EAR

    try:

        if broadcast_progress is None:

            async def broadcast_progress(s): print(f"[Progress] {s}")

        if not STRUCTURAL_3D_ENABLED:

            return {

                "success": False,

                "response": jarvis_reply("3D generation module is not active", "error"),

                "error": "Structural 3D module (trimesh) not installed or failed to load."

            }



        # 0. DIRECT BLUEPRINT DETECTION (The "God-Mode" Injector)

        is_direct_blueprint = False

        best_recipe = None

        

        print(f"[3DGen] [SEARCH] Checking for blueprints. text_len={len(text)}, params_type={type(params)}")

        

        # Priority 1: Check if params already has the design (Persisted from DesignGen step)

        if params and (isinstance(params, list) or (isinstance(params, dict) and "parts" in params)):

            print(f"[3DGen] [GEM] PERSISTED DESIGN DETECTED in params. Bypassing AI drafting.")

            if isinstance(params, dict):

                params["is_direct_blueprint"] = True

            elif isinstance(params, list):

                # If list, we can't easily set a flag on it, but Structural3DGenerator handles count > 100

                pass

            best_recipe = params

            is_direct_blueprint = True

            

            # Try to heal even if braces are missing (mangled text)

            clean_text = text.strip()

            # FIX: Replace underscores with spaces if this looks like a mangled blueprint transcript

            if "_" in clean_text and ('"id":' in clean_text.replace("_", " ") or '"parts":' in clean_text.replace("_", " ")):

                print("[3DGen] [FIX] Correcting underscore-mangled blueprint transcript.")

                clean_text = clean_text.replace("_", " ")



            if '"parts":' in clean_text or '"id":' in clean_text:

                if not (clean_text.startswith("{") or clean_text.startswith("[")):

                    print("[3DGen] [WARN] Mangled blueprint detected (missing braces). Attempting deep recovery.")

                    clean_text = "{" + clean_text + "}"

                

                potential_json = heal_json(clean_text)

                if potential_json:

                    try:

                        blueprint_data = json.loads(potential_json)

                        if isinstance(blueprint_data, list) or (isinstance(blueprint_data, dict) and "parts" in blueprint_data):

                            print(f"[3DGen] [GEM] DIRECT BLUEPRINT DETECTED ({len(potential_json)} chars). Bypassing AI drafting.")

                            speak("Direct engineering blueprint detected. Initializing orbital fabrication sequence immediately.", blocking=False)

                            if isinstance(blueprint_data, dict):

                                blueprint_data["is_direct_blueprint"] = True

                            best_recipe = blueprint_data

                            is_direct_blueprint = True

                    except Exception as je:

                        print(f"[3DGen] Deep recovery failed: {je}")

        

        if not is_direct_blueprint:

            print("[3DGen] ⚙️ Proceeding with AI Drafting (no blueprint found).")



        if not is_direct_blueprint:

            speak("Initiating aerospace engineering sequence. Designing structural subsystems and calculating mass distributions now.", blocking=False)

            

            # 1. OPTIONAL: Use LLaVA for visual context

            visual_context = ""

            if image_path and ENHANCED_VISION_ENABLED:

                try:

                    await broadcast_progress("Analyzing reference image with Vision AI...")

                    print(f"[3DGen] Analyzing reference image: {image_path}")

                    analysis = enhanced_vision("Describe the 3D structure, dimensions, and layout of the main object in this image for modeling.", image_path)

                    visual_context = format_for_brain(analysis)

                except Exception as e:

                    print(f"[3DGen] Visual analysis failed: {e}")



            # SELECT MODEL: Use globally detected drafting model

            # FORCE 7B - Disable 'Dual Brain' optimization to prevent 3b fallback

            DRAFTING_MODEL = MODEL_BRAIN 



            recipe_prompt = f"""

            [SYSTEM: NASA CHIEF ROBOTICS & AEROSPACE SYSTEMS ENGINEER]

            TASK: Generate a DETERMINISTIC structural assembly for: '{text}'

            

            [RULE #1: ALLOWED PRIMITIVES]

            - box: [x, y, z]

            - cylinder: [radius, height]

            - tube: [outer_r, inner_r, height]

            - panel: [x, y, thickness]

            - sphere: [radius]

            - wing: [span, chord, thickness]

            - nozzle: [radius, length]

            - cone: [radius, height]

            - bolt: [radius, length]

            - rib: [x, y, z]



            [RULE #2: NO TEXT OUTSIDE JSON]

            Reject conversational filler. Output ONLY the JSON.



            [STRICT ENGINEERING RULES - PHYSICAL INTEGRITY]

            1. PART COUNT: EXACTLY 300-500 UNIQUE PARTS.

            2. SCALE: All dims in METERS (0.01 - 2.0). 

            3. NESTED STRUCTURE [CRITICAL]: You MUST use the 'sub_parts' key to nest components.

            

            [TEMPLATE]

            {{

              "project": "Mars_Helicopter",

              "parts": [

                {{

                  "name": "Landing_Gear_Leg_1",

                  "type": "cylinder", "dims": [0.03, 0.8], "pos": [0.5, -0.5, -0.5],

                  "sub_parts": [

                     {{ "name": "Leg_1_Joint_Upper", "type": "sphere", "dims": [0.05], "pos": [0, 0.4, 0] }},

                     {{ "name": "Leg_1_Piston_Dampener", "type": "tube", "dims": [0.04, 0.03, 0.3], "pos": [0, -0.2, 0] }},

                     {{ "name": "Leg_1_Foot_Pad", "type": "cone", "dims": [0.15, 0.05], "pos": [0, -0.4, 0] }},

                     {{ "name": "Leg_1_Bolt_A", "type": "bolt", "dims": [0.01, 0.03], "pos": [0.03, 0.35, 0] }}

                  ]

                }}

              ]

            }}

            """



            # MULTI-PASS GENERATION STRATEGY (Divide & Conquer for 300+ parts)

            # Trigger if "extreme" in text OR if detail_level is MAX (5)

            is_extreme = ("detailed" in text.lower() or "extreme" in text.lower()) or (params.get("detail_level", 1) >= 5)

            

            if is_extreme:

                print("[3DGen] [ROCKET] EXTREME MODE: Engaging Multi-Pass Divide & Conquer Strategy...")

                await broadcast_progress("Engaging Multi-Pass Engineering (5 Stages)...")

                

                phases = [

                    {"name": "Frame_Hull", "focus": "1x 'tube' hull, 60x internal 'rib' (staggered), 80x 'bolt' (on ribs)."},

                    {"name": "Propulsion", "focus": "2x Motor Assemblies. Each: 1x 'cylinder' Core, 6x 'wing' Blades, 4x 'sphere' joints, 40x 'bolt'."},

                    {"name": "Landing_Gear", "focus": "4x Legs. Each: 5x 'sphere' Joints, 2x 'tube' Piston, 1x 'cone' Foot, 25x 'bolt'."},

                    {"name": "Avionics_Power", "focus": "6x Unique 'box' Boards. 16x 'cylinder' Battery Cells. 50x 'bolt' connectors."},

                    {"name": "Sensors_Fasteners", "focus": "140x 'bolt', 30x 'sphere' Sensors, 30x 'panel' brackets."}

                ]

                

                master_parts = []

                

                for i, phase in enumerate(phases):

                    phase_name = phase["name"]

                    phase_focus = phase["focus"]

                    

                    await broadcast_progress(f"Engineering Module {i+1}/5: {phase_name}...")

                    print(f"[3DGen] Generating Phase {i+1}: {phase_name}")

                    

                    phase_prompt = f"""

                    [SYSTEM: NASA SUBSYSTEM ENGINEER - {phase_name}]

                    TASK: Generate detailed 3D parts ONLY for '{phase_name}' for: '{text}'.

                    

                    [STRICT RULES]

                    1. UNIQUE PARTS: Focus on structural complexity.

                    2. PRIMITIVES ONLY: box, cylinder, tube, panel, sphere, wing, nozzle, cone, bolt, rib.

                    3. OUTPUT: JSON ONLY. Use key "parts" at top level.

                    4. COORDINATES: Absolute world coordinates.

                    5. [GUIDELINE]: Be as detailed as possible. provide all sub-components.

                    

                    [EXAMPLE STRUCTURE]:

                    {{ "parts": [ 

                       {{ "name": "Hull_Rib_1", "type": "rib", "dims": [0.05, 0.05, 0.5], "pos": [0,0,0] }},

                       {{ "name": "Hull_Bolt_1", "type": "bolt", "dims": [0.01, 0.02], "pos": [0.03,0,0] }}

                    ] }}

                    """

                    

                    # Call Ollama for this phase - FIRST TIME SUCCESS ARCHITECTURE

                    phase_parts = []

                    phase_content = ""

                    try:

                        # Temperature 0.2 for strict JSON adherence

                        stream = ollama.chat(

                            model=DRAFTING_MODEL,

                            messages=[{"role": "user", "content": phase_prompt}],

                            options={"temperature": 0.2, "num_ctx": 16384, "num_predict": 16384, "top_p": 0.9}, 

                            stream=True

                        )

                        for chunk in stream:

                            phase_content += chunk["message"]["content"]

                            if len(phase_content) % 1000 == 0:

                                sys.stdout.write("█")

                            else:

                                sys.stdout.write(".")

                            sys.stdout.flush()

                        

                        print(f"\n[3DGen] Phase {phase_name} captured: {len(phase_content)} chars.")

                        

                        # Extract and heal JSON

                        repaired_json = heal_json(phase_content)

                        if repaired_json:

                            data = json.loads(repaired_json)

                            flattened = Structural3DGenerator.flatten_hierarchy(data)

                            phase_parts = flattened.get("parts", [])

                            print(f"[3DGen] Success: {len(phase_parts)} parts found.")

                            

                    except Exception as e:

                        print(f"\n[3DGen] Phase {phase_name} failed: {e}")

                        # Log failure but continue to next phase

                        log_path = os.path.join("logs", f"3d_failure_{phase_name}.txt")

                        os.makedirs("logs", exist_ok=True)

                        with open(log_path, "w", encoding="utf-8") as f: f.write(phase_content)

                    

                    master_parts.extend(phase_parts)



                print(f"[3DGen] Multi-Pass Complete. Total Parts: {len(master_parts)}")

                best_recipe = {"project": "Extreme_Ingenuity_Assembly", "parts": master_parts}

                

            else:

                # SINGLE PASS (Legacy/Simple)

                try:

                    # 2. UNIFIED BRAIN CREATES HIERARCHICAL RECIPE

                    await broadcast_progress("Unified Engineering Brain: Designing Robot...")

                    print(f"[3DGen] {DRAFTING_MODEL} generating standard assembly recipe...")

                    

                    full_content = ""

                    

                    # Use streaming to provide "heartbeat" progress updates

                    async def stream_ai_design():

                        nonlocal full_content

                        max_retries = 3

                        for attempt in range(max_retries):

                            try:

                                full_content = "" 

                                stream = ollama.chat(

                                    model=DRAFTING_MODEL, 

                                    messages=[

                                        {"role": "system", "content": "Master Robotics Engineer. Output valid JSON assembly only."},

                                        {"role": "user", "content": recipe_prompt}

                                    ], 

                                    options={"temperature": 0.4, "num_predict": 16384}, 

                                    keep_alive=-1,

                                    stream=True

                                )

                                for chunk in stream:

                                    token = chunk["message"]["content"]

                                    sys.stdout.write(token)

                                    sys.stdout.flush()

                                    full_content += token

                                return 

                            except Exception as e:

                                if attempt < max_retries - 1: await asyncio.sleep(2)

                                else: raise e

                    

                    await asyncio.wait_for(stream_ai_design(), timeout=600.0)

                    

                    # Heal single pass

                    repaired_json = heal_json(full_content)

                    best_recipe = json.loads(repaired_json) if repaired_json else None

                    

                except Exception as e:

                     raise ValueError(f"Generation failed: {e}")



        # 2.5: UNIFIED ASSEMBLY (Flatten & Validate)

        if best_recipe:

            # 2.5: FLATTEN HIERARCHY (Support nested sub_parts)

            # This ensures any nested structures from either pass are unified

            best_recipe = flatten_hierarchy(best_recipe)

            

            # FIDELITY GUARD: Reject 'Lazy' AI drafts for complex requests

            current_count = len(best_recipe.get("parts", []))

            print(f"[3DGen] Fidelity Check: {current_count} parts found.")

            if current_count < 50:

                print(f"[3DGen] [WARN] LAZINESS DETECTED: AI only generated {current_count} parts. Mission requires 300+.")

                await broadcast_progress("Fidelity Alert: AI generated a simplified structure. Attempting to upscale detail...")

            

            # AUTO-FIX: Symmetry

            best_recipe = enforce_symmetry(best_recipe)

            

        recipe = best_recipe

        print(f"[3DGen] Blueprint RECOVERED: {recipe.get('project', 'Unnamed')} with {len(recipe['parts'])} parts.")

        

        # 3. GENERATE GLB (NASA Modules 1 & 2)

        await broadcast_progress("Generating Geometry (Mesh Processing)...")

        

        # Thread-safe progress callback

        loop = asyncio.get_running_loop()

        def sync_progress_callback(status):

            loop.call_soon_threadsafe(

                lambda: asyncio.create_task(broadcast_progress(status))

            )



        # Offload blocking mesh processing

        gen_result = await asyncio.to_thread(

            three_d_gen.generate_glb, 

            recipe, 

            progress_callback=sync_progress_callback, 

            unity_path=unity_path

        )

            

        # Parse return value (dict or string fallback)

        if isinstance(gen_result, dict):

            visual_path = gen_result["visual"]

            collision_path = gen_result["collision"]

            if "success" in gen_result and gen_result["success"]:

                # Cache success for persistence

                global LAST_GENERATED_MODEL

                LAST_GENERATED_MODEL = gen_result

                final_status = f"Assembly Complete. {gen_result.get('message', '')}"

                await broadcast_progress(final_status)

            physics_path = gen_result["physics"]

            sdf_path = gen_result.get("sdf", "")

            filename = os.path.basename(visual_path)

        else:

            visual_path = gen_result

            collision_path = gen_result # Fallback

            physics_path = ""

            sdf_path = ""

            filename = os.path.basename(visual_path)

        

        from config import HOST, API_PORT

        url_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST

        

        # Generate URLs for all assets

        model_url = f"http://{url_host}:{API_PORT}/models/{filename}"

        collision_url = f"http://{url_host}:{API_PORT}/models/{os.path.basename(collision_path)}"

        physics_url = f"http://{url_host}:{API_PORT}/models/{os.path.basename(physics_path)}"

        sdf_url = f"http://{url_host}:{API_PORT}/models/{os.path.basename(sdf_path)}" if sdf_path else ""

        

        print(f"[3DGen] Visual: {visual_path} -> {model_url}")

        print(f"[3DGen] Collision: {collision_path}")

        print(f"[3DGen] SDF: {sdf_path}")

        

        speak(f"Engineering assembly for {recipe.get('project', 'the vehicle')} is complete. Gazebo SDF, collision meshes, and visual assets are ready for deployment.")

        

        return {

            "success": True,

            "response": f"3D Model '{recipe.get('project')}' generated successfully.",

            "model_path": model_url,

            "visual_url": model_url,

            "collision_url": collision_url,

            "physics_url": physics_url,

            "sdf_url": sdf_url,

            "recipe": recipe

        }

        

    except Exception as e:

        error_msg = str(e)

        print(f"[3DGen] CRITICAL ERROR: {error_msg}")

        import traceback

        traceback.print_exc()

        

        reply = jarvis_reply("had trouble assembling the 3D structure", "error")

        return {

            "success": False, 

            "response": reply,

            "error": error_msg

        }

    finally:

        set_busy(False) # UNLOCK EAR



def process_jarvis_knowledge(text):

    """Handle JARVIS/Iron Man knowledge queries instantly (no AI)"""

    print("[JARVIS Knowledge] Processing...")

    

    if JARVIS_VOICE_ENABLED:

        response = get_jarvis_response(text)

        if response:

            return {"response": response}

    

    # Fallback to reasoning if no knowledge match

    return process_reasoning(text)



def process_correction(text):

    """Handle user corrections - learn from mistakes"""

    global last_action_context

    

    if MEMORY_ENABLED and last_action_context["text"]:

        correction_note = f"CORRECTION: When user said '{last_action_context['text']}', I responded with '{last_action_context['response']}' but user corrected me: '{text}'"

        save_to_memory(last_action_context["text"], correction_note)

        print(f"[Memory] Saved correction")

    

    # Let Phi3 generate the apology as JARVIS

    return {"response": jarvis_reply(f"user corrected me about: {text}", "error")}



# =============================================================================

# AUTOMATION - Direct Python actions (NO AI needed, INSTANT!)

# =============================================================================



def process_automation(text):

    """

    Fast automation - Python executes, Phi3 responds as JARVIS.

    Actions are instant, response is generated by brain.

    """

    text_lower = text.lower()

    actions = []

    action_descriptions = []

    

    new_tab = "new tab" in text_lower

    

    # Open apps

    if "open chrome" in text_lower:

        actions.append({"action": "open_chrome", "parameter": ""})

        action_descriptions.append("open Chrome")

    

    if "open youtube" in text_lower and "search" not in text_lower:

        actions.append({"action": "open_youtube", "parameter": ""})

        action_descriptions.append("open YouTube")

    if "open spotify" in text_lower:

        actions.append({"action": "open_spotify", "parameter": ""})

        action_descriptions.append("open Spotify")

    if "open notepad" in text_lower:

        actions.append({"action": "open_notepad", "parameter": ""})

        action_descriptions.append("open Notepad")

    

    # Searches - extract ONLY the search term

    if "youtube" in text_lower:

        # Find the actual search query - look for patterns like "for X" or "search X"

        query = None

        

        # Pattern 1: "for BMW" - get everything after last "for"

        if " for " in text_lower:

            parts = text_lower.split(" for ")

            query = parts[-1].strip()  # Get last part after "for"

        # Pattern 2: "youtube BMW" - get word after youtube

        elif "youtube " in text_lower:

            parts = text_lower.split("youtube ")

            if len(parts) > 1:

                query = parts[-1].strip()

        

        if query:

            # Clean up the query

            query = query.replace("?", "").replace(".", "").replace("!", "").strip()

            # Remove trailing words that aren't part of search

            for suffix in ["please", "thanks", "thank you", "sir"]:

                query = query.replace(suffix, "").strip()

            

            print(f"[Automation] YouTube search: '{query}'")

            

            if query and len(query) > 1:

                actions.append({"action": "search_youtube", "parameter": query})

                action_descriptions.append(f"search YouTube for {query}")

            

    elif "search" in text_lower and "youtube" not in text_lower:

        # Google search

        query = None

        

        if " for " in text_lower:

            parts = text_lower.split(" for ")

            query = parts[-1].strip()

        elif "search " in text_lower:

            parts = text_lower.split("search ")

            if len(parts) > 1:

                query = parts[-1].strip()

        

        if query:

            query = query.replace("?", "").replace(".", "").replace("!", "").strip()

            for suffix in ["please", "thanks", "thank you", "sir", "on google", "google"]:

                query = query.replace(suffix, "").strip()

            

            print(f"[Automation] Google search: '{query}'")

            

            if query and len(query) > 1:

                actions.append({"action": "search_google", "parameter": query})

                action_descriptions.append(f"search Google for {query}")

    

    if actions:

        # Let Phi3 generate a JARVIS-style response

        action_text = " and ".join(action_descriptions)

        response = jarvis_reply(action_text, "ack_action")

        return {"actions": actions, "response": response}

    

    # Time/Date - handle different locations

    if "time" in text_lower:

        # Check for specific location

        if "egypt" in text_lower or "cairo" in text_lower:

            return {"actions": [{"action": "get_time_egypt", "parameter": ""}], "response": ""}

        elif "new york" in text_lower or "nyc" in text_lower:

            return {"actions": [{"action": "get_time_ny", "parameter": ""}], "response": ""}

        elif "london" in text_lower or "uk" in text_lower:

            return {"actions": [{"action": "get_time_london", "parameter": ""}], "response": ""}

        else:

            return {"actions": [{"action": "get_time", "parameter": ""}], "response": ""}

    if "date" in text_lower:

        return {"actions": [{"action": "get_date", "parameter": ""}], "response": ""}

    

    # Click actions

    if any(kw in text_lower for kw in ["click", "select", "choose", "play"]):

        if any(kw in text_lower for kw in ["first", "1st", "one"]):

            return {"actions": [{"action": "click_first", "parameter": ""}], "response": jarvis_reply("select the first item", "ack_action")}

        if any(kw in text_lower for kw in ["second", "2nd", "two"]):

            return {"actions": [{"action": "click_second", "parameter": ""}], "response": jarvis_reply("select the second item", "ack_action")}

        if any(kw in text_lower for kw in ["third", "3rd", "three"]):

            return {"actions": [{"action": "click_third", "parameter": ""}], "response": jarvis_reply("select the third item", "ack_action")}

    

    # Volume

    if "volume" in text_lower:

        if "up" in text_lower or "increase" in text_lower:

            return {"actions": [{"action": "volume_up", "parameter": ""}], "response": jarvis_reply("increase volume", "done_action")}

        if "down" in text_lower or "decrease" in text_lower:

            return {"actions": [{"action": "volume_down", "parameter": ""}], "response": jarvis_reply("decrease volume", "done_action")}

    

    if "mute" in text_lower:

        return {"actions": [{"action": "mute", "parameter": ""}], "response": jarvis_reply("mute audio", "done_action")}

    

    # Scroll

    if "scroll" in text_lower:

        if "down" in text_lower:

            return {"actions": [{"action": "scroll_down", "parameter": ""}], "response": jarvis_reply("scroll down", "done_action")}

        if "up" in text_lower:

            return {"actions": [{"action": "scroll_up", "parameter": ""}], "response": jarvis_reply("scroll up", "done_action")}

    

    # Navigation

    if "go back" in text_lower or "back" in text_lower:

        return {"actions": [{"action": "go_back", "parameter": ""}], "response": jarvis_reply("go back", "done_action")}

    if "refresh" in text_lower or "reload" in text_lower:

        return {"actions": [{"action": "refresh", "parameter": ""}], "response": jarvis_reply("refresh page", "done_action")}

    

    return {"actions": [], "response": jarvis_reply("not sure how to do that task", "error")}



def execute_actions(data):

    """Execute automation actions"""

    from actions import browser, apps, system

    from datetime import datetime

    

    response = data.get("response", "")

    if response:

        speak(response)

    

    for i, act in enumerate(data.get("actions", [])):

        action, param = act.get("action", ""), act.get("parameter", "")

        print(f"[Action {i+1}] {action} {param}")

        

        if i > 0:

            time.sleep(1.0)

        

        try:

            if action == "open_chrome": 

                browser.open_url("https://google.com", new_tab=True)

            elif action == "open_youtube": 

                browser.open_url("https://youtube.com", new_tab=True)

            elif action == "open_spotify": 

                apps.open_app("spotify")

            elif action == "open_notepad": 

                apps.open_app("notepad")

            elif action == "search_google": 

                browser.search_google(param, new_tab=False)

            elif action == "search_youtube": 

                browser.search_youtube(param, new_tab=False)

            elif action == "search_google_new_tab": 

                browser.search_google(param, new_tab=True)

            elif action == "search_youtube_new_tab": 

                browser.search_youtube(param, new_tab=True)

            elif action == "screenshot": 

                system.screenshot()

            elif action == "get_time": 

                speak(f"It's {datetime.now().strftime('%I:%M %p')}.")

            elif action == "get_time_egypt":

                # Egypt is UTC+2

                from datetime import timezone, timedelta

                egypt_time = datetime.now(timezone(timedelta(hours=2)))

                speak(f"In Egypt, it's {egypt_time.strftime('%I:%M %p')}.")

            elif action == "get_time_ny":

                # New York is UTC-5 (or -4 DST)

                from datetime import timezone, timedelta

                ny_time = datetime.now(timezone(timedelta(hours=-5)))

                speak(f"In New York, it's {ny_time.strftime('%I:%M %p')}.")

            elif action == "get_time_london":

                # London is UTC+0 (or +1 DST)

                from datetime import timezone, timedelta

                london_time = datetime.now(timezone(timedelta(hours=0)))

                speak(f"In London, it's {london_time.strftime('%I:%M %p')}.")

            elif action == "get_date": 

                speak(f"Today is {datetime.now().strftime('%A, %B %d')}.")

            elif action == "click_first":

                import pyautogui; time.sleep(0.3); pyautogui.click(400, 400)

            elif action == "click_second":

                import pyautogui; pyautogui.click(400, 600)

            elif action == "click_third":

                import pyautogui; pyautogui.click(400, 800)

            elif action == "scroll_down":

                import pyautogui; pyautogui.scroll(-3)

            elif action == "scroll_up":

                import pyautogui; pyautogui.scroll(3)

            elif action == "go_back":

                import pyautogui; pyautogui.hotkey('alt', 'left')

            elif action == "refresh":

                import pyautogui; pyautogui.press('f5')

            elif action == "volume_up":

                import pyautogui; pyautogui.press('volumeup')

            elif action == "volume_down":

                import pyautogui; pyautogui.press('volumedown')

            elif action == "mute":

                import pyautogui; pyautogui.press('volumemute')

        except Exception as e:

            print(f"[Error] {e}")



# =============================================================================

# AUDIO FUNCTIONS

# =============================================================================



def record_until_silence():
    """Record with VAD - INSTANT response after speech ends. Self-healing on audio errors."""
    print("[Audio] 🎤", end=" ", flush=True)
    
    audio = None
    stream = None
    frames = []
    has_speech = False
    
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, 
                            input=True, frames_per_buffer=CHUNK, input_device_index=MIC_INDEX)
        
        silent_chunks = 0
        start_time = time.time()
        silence_chunks_needed = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
        zero_level_count = 0

        while time.time() - start_time < MAX_RECORD_TIME:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
            except OSError as e:
                print(f"\n[Audio] Recording Error (Hardware): {e}")
                break # Exit loop, return what we have
            except Exception as e:
                print(f"\n[Audio] Recording Error: {e}")
                break

            level = np.abs(audio_np).mean()
            
            # === ZERO LEVEL DETECTION === (Indicates mic not working)
            if level < 5:
                zero_level_count += 1
                if zero_level_count > 50: # ~1 second of silence
                    print("\n[Audio] ⚠️ No audio input detected - check mic connection!")
                    break
            else:
                zero_level_count = 0
                    
            # Debug print to see what the mic is actually hearing
            if random.random() < 0.1: print(f"[{int(level)}]", end="", flush=True)

            if not has_speech and time.time() - start_time > 4.0:
                print("(waiting...)")
                break

            frames.append(audio_np)

            if level > SILENCE_THRESHOLD:
                has_speech = True
                silent_chunks = 0
                print(".", end="", flush=True)
            elif has_speech:
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    print(" [OK]")
                    break
                    
    except Exception as init_error:
        print(f"[Audio] Failed to initialize recording: {init_error}")
        
    finally:
        # Always clean up audio resources
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except: pass
        if audio:
            try:
                audio.terminate()
            except: pass

    return np.concatenate(frames).astype(np.float32) / 32768.0 if has_speech and len(frames) > 0 else None



def transcribe(audio_data):

    """Transcription with Faster Whisper - optimized for CPU & RAM"""

    start = time.time()

    # OPTIMIZED for CPU: beam_size=1 is MUCH faster
    # VAD filter enabled to cut noise and prevent hallucinations
    segments, info = whisper_model.transcribe(

        audio_data, 

        beam_size=1,  # Fast on CPU

        language="en",

        initial_prompt="Hey Jarvis.",

        vad_filter=True,  # Re-enabled to filter out noise and prevent hallucinations

        vad_parameters=dict(min_silence_duration_ms=300),  # Fast cutoff

        no_speech_threshold=0.6,  # Skip audio chunks with no speech

    )

    

    text = "".join([segment.text for segment in segments]).strip()
    
    # === Hallucination/Noise Filter ===
    if text.replace("0", "").strip() == "" and len(text) > 0:
        print(f"[Whisper] 🔇 Noise/Hallucination detected ('{text}'), ignoring.")
        return ""
    
    print(f"[Whisper] '{text}' ({time.time()-start:.1f}s)")
    return text


# =============================================================================

# GIBBERISH DETECTION - Skip unintelligible speech

# =============================================================================



def is_gibberish(text):
    """
    Detect if transcribed text is gibberish/noise/background chatter.
    Returns True if we should skip processing.
    """
    if not text:
        return True
    
    text_lower = text.strip().lower()
    clean_text = text_lower.rstrip(".,!?").strip()
    
    # 1. Too short (VAD-Lite)
    if len(clean_text) < 3:
        return True
        
    # 2. Common Whisper Noise/Stutter words
    noise_words = [
        "uh", "um", "ah", "oh", "hmm", "well", "actually", 
        "transcribed by", "thank you for watching", "subtitles",
        "the", "a", "an", "is", "am", "are" # Ignore single articles/verbs if alone
    ]
    
    words = clean_text.split()
    if len(words) == 1 and words[0] in noise_words:
        return True
        
    # 3. Random Character Strings (Non-speech noise)
    if re.fullmatch(r'[^aeiouy]{4,}', clean_text): # 4+ consonants in a row
        return True
        
    return False

    

    # Common Whisper hallucinations/noise

    noise_patterns = [

        "thank you", "thanks for watching", "subscribe", "like and subscribe",

        "you", "bye", "okay", "um", "uh", "hmm", "ah", "oh",

        "...", ".", ",", "!", "?",

        "music", "applause", "laughter", "[music]", "[applause]",

        "the", "a", "an", "is", "it", "i", "and", "or", "but",

    ]

    

    # Exact match with noise

    if clean_text in noise_patterns:

        return True

    
    # Only punctuation or very short meaningless

    alpha_only = ''.join(c for c in clean_text if c.isalpha())

    if len(alpha_only) < 3:

        return True

    

    # Repeated characters (like "ahhhhh" or "ummmmm")

    if len(set(alpha_only)) < 3 and len(alpha_only) > 3:

        return True

    

    # Check if it has at least one real word (3+ chars)

    # Clean each word of punctuation before checking

    words = clean_text.split()

    real_words = [w.rstrip(".,!?") for w in words if len(w.rstrip(".,!?")) >= 3]

    if len(real_words) == 0:

        return True

    

    return False



# =============================================================================

# INSTANT PYTHON ACTIONS - No AI needed, < 1 second!

# =============================================================================



def action_time():

    """Get current time - INSTANT"""

    from datetime import datetime

    now = datetime.now()

    time_str = now.strftime("%I:%M %p")

    return f"It's {time_str}, sir."



def action_date():

    """Get current date - INSTANT"""

    from datetime import datetime

    now = datetime.now()

    date_str = now.strftime("%A, %B %d, %Y")

    return f"Today is {date_str}, sir."



def open_in_chrome(url):
    """Explicitly force Chrome with specific user profile (v21.1)"""
    print(f"[Browser] Forcing Chrome (Profile: youseflovemessi41) for: {url}")
    import subprocess
    import os # Added for os.path.exists
    try:
        # Common Windows paths for Google Chrome
        chrome_exe = None
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        ]
        
        for p in paths:
            if os.path.exists(p):
                chrome_exe = p
                break
        
        if chrome_exe:
            # Force the specific profile used by youseflovemessi41@gmail.com
            # Verified as 'Default'
            subprocess.Popen([chrome_exe, f"--profile-directory=Default", url])
        else:
            # Fallback
            import webbrowser
            webbrowser.open(url)
    except Exception as e:
        print(f"[Browser] Profile Force Error: {e}")
        import webbrowser
        webbrowser.open(url)


def action_open(text):
    """
    Fuzzy Multi-App Opening (v21.2)
    Supports: "open chrome", "open chrome and youtube", "launch spotify then excel"
    """
    text_lower = text.lower().replace("launch ", "open ").replace("run ", "open ").replace("start ", "open ")
    import subprocess
    import webbrowser
    import re
    
    # 1. Split for multi-app requests
    parts = re.split(r'\s+(?:and|then|\&)\s+', text_lower)
    opened = []
    
    for part in parts:
        app = part.replace("open ", "").strip().rstrip(".,!?")
        # Remove politeness
        for word in ["please", "sir", "thanks", "now"]:
            app = app.replace(word, "").strip()
            
        if not app: continue
        
        # Fuzzy mapping
        if any(k in app for k in ["chrome", "browser", "internet", "google"]):
            open_in_chrome("https://google.com")
            opened.append("Chrome")
        elif "youtube" in app:
            open_in_chrome("https://youtube.com")
            opened.append("YouTube")
        elif "spotify" in app:
            try:
                if sys.platform == 'darwin': subprocess.Popen(["open", "-a", "Spotify"])
                else: os.startfile("spotify:")
            except: open_in_chrome("https://open.spotify.com")
            opened.append("Spotify")
        elif any(k in app for k in ["code", "visual studio", "vscode"]):
            subprocess.Popen(["code"]) if sys.platform != 'darwin' else subprocess.Popen(["open", "-a", "Visual Studio Code"])
            opened.append("VS Code")
        elif any(k in app for k in ["terminal", "cmd", "iterm"]):
             subprocess.Popen(["open", "-a", "iTerm"]) if sys.platform == 'darwin' else subprocess.Popen(["cmd.exe"])
             opened.append("Terminal")
        elif "safari" in app:
            subprocess.Popen(["open", "-a", "Safari"])
            opened.append("Safari")
        elif "music" in app or "apple music" in app:
            subprocess.Popen(["open", "-a", "Music"])
            opened.append("Music")
        else:
            # Fallback: Try generic 'open' command
            try:
                if sys.platform == 'darwin': subprocess.Popen(["open", "-a", app.capitalize()])
                else: subprocess.Popen(["start", app], shell=True)
                opened.append(app)
            except:
                print(f"[Open] Could not find app: {app}")
    
    if opened:
        res_msg = f"Opening {', '.join(opened)}, sir."
        return res_msg
    return "I couldn't quite identify the application, sir."

def action_youtube_search(text):

    """Search YouTube - INSTANT, can auto-play first result"""

    import webbrowser
    import urllib.parse
    import time as t
    
    # Extract search query
    text_lower = text.lower()
    auto_play = "play" in text_lower and "first" in text_lower
    
    if " for " in text_lower:
        query = text_lower.split(" for ")[-1].strip()
    elif "youtube " in text_lower:
        query = text_lower.split("youtube ")[-1].strip()
    elif "search " in text_lower:
        query = text_lower.split("search ")[-1].strip()
    
    # Clean query
    for word in ["on youtube", "youtube", "please", "sir", "and play", "play first", "first video", "the first"]:
        query = query.replace(word, "").strip()
    
    if query:
        encoded = urllib.parse.quote(query)
        url = f"https://www.youtube.com/results?search_query={encoded}"
        open_in_chrome(url)  # Force Chrome (v21)
        
        # Auto-play first video if requested
        if auto_play:
            import pyautogui
            t.sleep(2.5)  # Wait for page to load
            screen_width = pyautogui.size()[0]
            pyautogui.click(screen_width // 2 - 200, 350)  # Click first video
            return f"Playing {query} on YouTube, sir."
        
        return f"Searching YouTube for {query}, sir."
    return "What would you like me to search for, sir?"



def action_google_search(text):

    """Search Google - INSTANT"""

    import webbrowser
    import urllib.parse
    
    text_lower = text.lower()
    query = ""
    
    if " for " in text_lower:
        query = text_lower.split(" for ")[-1].strip()
    elif "search " in text_lower:
        query = text_lower.split("search ")[-1].strip()
    elif "google " in text_lower:
        query = text_lower.split("google ")[-1].strip()
    
    for word in ["on google", "google", "please", "sir"]:
        query = query.replace(word, "").strip()
    
    if query:
        encoded = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded}"
        open_in_chrome(url)  # Force Chrome (v21)
        return f"Searching for {query}, sir."
    return "What would you like me to search for, sir?"


def action_smart_search(text):

    """

    SMART search handler - parses compound commands like:

    - "open chrome and search for cats" → Google search

    - "open chrome and search on youtube for cats" → YouTube search

    - "open chrome and search on adobe after effects for tutorials" → Site-specific search

    """

    import webbrowser
    import urllib.parse
    
    text_lower = text.lower()
    
    # Known sites with search URLs
    SITE_SEARCH_URLS = {
        "youtube": "https://www.youtube.com/results?search_query=",
        "google": "https://www.google.com/search?q=",
        "bing": "https://www.bing.com/search?q=",
        "amazon": "https://www.amazon.com/s?k=",
        "ebay": "https://www.ebay.com/sch/i.html?_nkw=",
        "reddit": "https://www.reddit.com/search/?q=",
        "twitter": "https://twitter.com/search?q=",
        "x": "https://twitter.com/search?q=",
        "github": "https://github.com/search?q=",
        "stackoverflow": "https://stackoverflow.com/search?q=",
        "stack overflow": "https://stackoverflow.com/search?q=",
        "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search=",
        "spotify": "https://open.spotify.com/search/",
        "netflix": "https://www.netflix.com/search?q=",
        "adobe": "https://www.adobe.com/search.html?q=",
        "adobe after effects": "https://www.google.com/search?q=site:adobe.com+after+effects+",
        "after effects": "https://www.google.com/search?q=after+effects+",
        "premiere": "https://www.google.com/search?q=adobe+premiere+",
        "photoshop": "https://www.google.com/search?q=photoshop+",
    }
    
    # Extract the site and query
    site = None
    query = ""
    
    # Pattern: "search on [site] for [query]"
    if " on " in text_lower and " for " in text_lower:
        # Get everything between "on" and "for" as the site
        on_idx = text_lower.find(" on ") + 4
        for_idx = text_lower.find(" for ", on_idx)
        if for_idx > on_idx:
            site = text_lower[on_idx:for_idx].strip()
            query = text_lower[for_idx + 5:].strip()
    
    # Pattern: "search for [query]" (no site = Google)
    elif " for " in text_lower:
        query = text_lower.split(" for ")[-1].strip()
        site = "google"
    
    # Pattern: "search [site] [query]"
    elif "search " in text_lower:
        parts = text_lower.split("search ")[-1].strip()
        # Check if first word is a known site
        for known_site in SITE_SEARCH_URLS:
            if parts.startswith(known_site):
                site = known_site
                query = parts[len(known_site):].strip()
                break
        if not site:
            site = "google"
            query = parts
    
    # Clean up query
    for word in ["please", "sir", "thanks", "thank you"]:
        query = query.replace(word, "").strip()
    
    if not query:
        return "What would you like me to search for, sir?"
    
    # Find the search URL
    search_url = None
    site_name = site or "google"
    
    # Check for exact match first
    if site in SITE_SEARCH_URLS:
        search_url = SITE_SEARCH_URLS[site]
    else:
        # Check for partial match
        for known_site, url in SITE_SEARCH_URLS.items():
            if known_site in site or site in known_site:
                search_url = url
                site_name = known_site
                break
    
    # Default to Google with site-specific search
    if not search_url:
        # Search Google for "[site] [query]"
        encoded = urllib.parse.quote(f"{site} {query}")
        url = f"https://www.google.com/search?q={encoded}"
        open_in_chrome(url)
        return f"Searching for {query} on {site}, sir."
    
    # Open the search URL
    encoded_query = urllib.parse.quote(query)
    full_url = f"{search_url}{encoded_query}"
    open_in_chrome(full_url)
    
    return f"Searching {site_name} for {query}, sir."


def action_click(text):

    """Click on screen elements - INSTANT (optimized for YouTube)"""

    import pyautogui
    import time as t
    
    text_lower = text.lower()
    t.sleep(0.8)  # Wait for page to load
    
    # YouTube video results are typically at these positions (1080p screen)
    # First video thumbnail is around y=350, second ~550, third ~750
    screen_width = pyautogui.size()[0]
    center_x = screen_width // 2  # Click center of screen
    
    if "first" in text_lower or "1st" in text_lower or "one" in text_lower:
        pyautogui.click(center_x - 200, 350)  # First YouTube result
        return "Clicking the first video, sir."
    elif "second" in text_lower or "2nd" in text_lower or "two" in text_lower:
        pyautogui.click(center_x - 200, 550)  # Second YouTube result
        return "Clicking the second video, sir."
    elif "third" in text_lower or "3rd" in text_lower or "three" in text_lower:
        pyautogui.click(center_x - 200, 750)  # Third YouTube result
        return "Clicking the third video, sir."
    elif "play" in text_lower:
        # Click center of screen (for video player)
        pyautogui.click(center_x, pyautogui.size()[1] // 2)
        return "Playing, sir."
    return "Click where, sir?"


def action_volume(text):

    """Volume control - INSTANT"""

    import pyautogui
    
    text_lower = text.lower()
    if "up" in text_lower or "louder" in text_lower:
        pyautogui.press('volumeup')
        pyautogui.press('volumeup')
        return "Volume up, sir."
    elif "down" in text_lower or "quieter" in text_lower:
        pyautogui.press('volumedown')
        pyautogui.press('volumedown')
        return "Volume down, sir."
    elif "mute" in text_lower:
        pyautogui.press('volumemute')
        return "Muted, sir."
    return "Done, sir."


def action_media(text):

    """Media controls - INSTANT"""

    import pyautogui
    
    text_lower = text.lower()
    if "pause" in text_lower or "stop" in text_lower:
        pyautogui.press('playpause')
        return "Paused, sir."
    elif "play" in text_lower or "resume" in text_lower:
        pyautogui.press('playpause')
        return "Playing, sir."
    elif "next" in text_lower or "skip" in text_lower:
        pyautogui.press('nexttrack')
        return "Next track, sir."
    elif "previous" in text_lower or "back" in text_lower:
        pyautogui.press('prevtrack')
        return "Previous track, sir."
    return "Done, sir."


def action_scroll(text):

    """Scroll control - INSTANT"""

    import pyautogui
    
    text_lower = text.lower()
    if "down" in text_lower:
        pyautogui.scroll(-5)
        return "Scrolling down, sir."
    elif "up" in text_lower:
        pyautogui.scroll(5)
        return "Scrolling up, sir."
    return "Done, sir."


def action_fullscreen(text):

    """Fullscreen toggle - INSTANT"""

    import pyautogui
    pyautogui.press('f')  # YouTube/most players use F for fullscreen
    return "Toggling fullscreen, sir."


def action_close(text):

    """Close current window/tab - INSTANT"""

    import pyautogui
    
    text_lower = text.lower()
    if "tab" in text_lower:
        pyautogui.hotkey('ctrl', 'w')
        return "Closing tab, sir."
    else:
        pyautogui.hotkey('alt', 'f4')
        return "Closing window, sir."


def action_minimize(text):

    """Minimize window - INSTANT"""

    import pyautogui
    pyautogui.hotkey('win', 'd')  # Show desktop / minimize all
    return "Minimized, sir."


def action_screenshot(text):

    """Take screenshot - INSTANT"""

    import pyautogui
    from datetime import datetime
    
    screenshot = pyautogui.screenshot()
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    screenshot.save(filename)
    return f"Screenshot saved as {filename}, sir."


def action_type(text):

    """Type text - INSTANT"""

    import pyautogui
    
    # Extract what to type
    text_lower = text.lower()
    to_type = ""
    
    if "type " in text_lower:
        to_type = text[text_lower.find("type ") + 5:].strip()
    elif "write " in text_lower:
        to_type = text[text_lower.find("write ") + 6:].strip()
    
    if to_type:
        pyautogui.typewrite(to_type, interval=0.02)
        return f"Typed it, sir."
    return "What should I type, sir?"


# =============================================================================

# HAND VISION & DEVICE TRANSFER ACTIONS

# =============================================================================



# Track last captured image for sending

last_captured_image = None



def on_hand_capture(image_path):

    """Called when hand gesture captures an image - auto-discovers and sends"""

    global last_captured_image, device_discovery_instance

    last_captured_image = image_path

    print(f"[HandVision] 📸 Captured: {image_path}")

    speak("Image captured, sir.")
    
    # Auto-discover and send to device
    if AUTO_DISCOVERY_ENABLED:
        if not device_discovery_instance:
            device_discovery_instance = DeviceDiscovery()
        
        speak("Searching for devices.")
        devices = device_discovery_instance.discover_once(timeout=2.0)
        
        if devices:
            device = devices[0]
            speak(f"Found {device['name']}. Sending image.")
            
            # Send image via TCP
            success = send_image_to_device(image_path, device['ip'], device.get('port', 5757))
            if success:
                speak("Image transferred successfully, sir.")
            else:
                speak("Transfer failed, sir.")
        else:
            speak("No devices found on network, sir.")
    else:
        speak("Device discovery not available, sir.")


def send_image_to_device(image_path: str, ip: str, port: int = 5757) -> bool:

    """Send image to a device via TCP"""

    import socket
    import struct
    
    if not os.path.exists(image_path):
        print(f"[Transfer] Image not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    filename = os.path.basename(image_path)
    
    header = {
        "filename": filename,
        "size": len(image_data),
    }
    header_json = json.dumps(header).encode()
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((ip, port))
        
        # Send header length + header
        sock.send(struct.pack('!I', len(header_json)))
        sock.send(header_json)
        
        # Send image data
        sock.sendall(image_data)
        
        # Wait for confirmation
        response = sock.recv(1024).decode()
        sock.close()
        
        if response == "OK":
            print(f"[Transfer] [OK] Image sent to {ip}")
            return True
        else:
            print(f"[Transfer] ✗ Failed: {response}")
            return False
            
    except Exception as e:
        print(f"[Transfer] Error: {e}")
        return False


def action_hand_detect(text):

    """Start hand detection mode with image overlay"""

    global hand_vision_instance, last_captured_image
    
    if not HAND_VISION_ENABLED:
        return "Hand vision module not available, sir. Install mediapipe."
    
    if hand_vision_instance and hand_vision_instance.running:
        return "Hand detection is already running, sir."
    
    # Find the last image to display
    overlay_image = None
    
    # Check generated images
    gen_dir = Path("./generated_images")
    if gen_dir.exists():
        images = sorted(gen_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
        if images:
            overlay_image = str(images[0])
    
    # Check captured images
    cap_dir = Path("./captured_images")
    if cap_dir.exists():
        cap_images = sorted(cap_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
        if cap_images:
            # Use whichever is newer
            if overlay_image:
                if cap_images[0].stat().st_mtime > Path(overlay_image).stat().st_mtime:
                    overlay_image = str(cap_images[0])
            else:
                overlay_image = str(cap_images[0])
    
    try:
        hand_vision_instance = HandVision(
            on_capture=on_hand_capture,
            overlay_image=overlay_image,  # Show this image on screen
        )
        # Start in background thread
        import threading
        thread = threading.Thread(target=hand_vision_instance.start, kwargs={"show_window": True})
        thread.daemon = True
        thread.start()
        
        if overlay_image:
            return "Hand detection activated with image overlay, sir. Catch the image to send it."
        return "Hand detection activated, sir. Make a catch gesture to capture."
    except Exception as e:
        print(f"[HandVision] Error: {e}")
        return f"Could not start hand detection: {e}"


def action_hand_stop(text):

    """Stop hand detection mode"""

    global hand_vision_instance
    
    if hand_vision_instance:
        hand_vision_instance.stop()
        hand_vision_instance = None
        return "Hand detection stopped, sir."
    return "Hand detection was not running, sir."


def action_send_image(text):

    """Send last captured/generated image to device"""

    global last_captured_image, device_transfer_instance
    
    if not DEVICE_TRANSFER_ENABLED:
        return "Device transfer module not available, sir."
    
    if not device_transfer_instance:
        device_transfer_instance = DeviceTransfer()
    
    # Find image to send
    image_path = None
    
    # Check for last captured image
    if last_captured_image and Path(last_captured_image).exists():
        image_path = last_captured_image
    else:
        # Check for last generated image
        gen_dir = Path("./generated_images")
        if gen_dir.exists():
            images = sorted(gen_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            if images:
                image_path = str(images[0])
    
    if not image_path:
        return "No image to send, sir. Capture or generate one first."
    
    # Find online device
    online = device_transfer_instance.find_online_devices()
    if not online:
        return "No devices online, sir."
    
    # Send
    success = device_transfer_instance.send_image(str(image_path))
    if success:
        return f"Image sent to {online[0].name}, sir."
    return "Transfer failed, sir."


def action_pair_device(text):

    """Start device pairing - generates QR code"""

    global device_transfer_instance
    
    if not DEVICE_TRANSFER_ENABLED:
        return "Device transfer module not available, sir."
    
    if not device_transfer_instance:
        device_transfer_instance = DeviceTransfer()
    
    # Generate pairing QR
    qr_path = device_transfer_instance.pair_device("JARVIS")
    if qr_path:
        # Open the QR image
        try:
            os.startfile(qr_path)
        except:
            pass
        return "Pairing QR code generated and displayed, sir. Scan it with the other device."
    return "Could not generate pairing code, sir."


def action_list_devices(text):

    """List trusted and online devices"""

    global device_transfer_instance
    
    if not DEVICE_TRANSFER_ENABLED:
        return "Device transfer module not available, sir."
    
    if not device_transfer_instance:
        device_transfer_instance = DeviceTransfer()
    
    # List all trusted
    trusted = device_transfer_instance.memory.list_all()
    print(f"\n[Devices] Trusted devices: {len(trusted)}")
    for dev in trusted:
        print(f"  - {dev.name}: {dev.ip}")
    
    # Find online
    online = device_transfer_instance.find_online_devices()
    if online:
        names = ", ".join([d.name for d in online])
        return f"Online devices: {names}"
    return "No devices online, sir."


    return "No devices online, sir."


# =============================================================================
# MISSING FUNCTIONS IMPLEMENTATION
# =============================================================================


# [DECOMMISSIONED] Legacy routing logic removed in favor of Unified Agentic Path.



def process_image_generation(text):
    """
    Handles image generation requests via image_generator.py
    With Qwen refinement (User Request: "Gemma listens and explains")
    """
    print(f"[Image] Processing: {text}")
    
    # 1. Extract prompt (Simple cleanup)
    prompt = text.lower()
    for trigger in ["generate image of", "create image of", "make an image of", "draw a", "generate image", "create image", "draw", "create an image of", "make image of", "show me a", "show me", "visualize"]:
        prompt = prompt.replace(trigger, "")
    prompt = prompt.replace("image", "").replace("photo", "").replace("picture", "").strip()
    
    if not prompt:
        return {"response": "I didn't hear what to generate, sir."}

    # 2. Direct Handoff (Image Generator handles refinement internally now)
    try:
        result = image_generator.process_image_request(prompt)
        
        if result.get("success"):
            path = result.get("path")
            # User Request: "I have created the image for [prompt]"
            final_prompt = result.get('prompt')
            # Use original short prompt for speech to be natural, or refined if needed.
            # actually user asked to say "I have created the image for [cat sleeping]"
            
            response_text = f"I have created the image for {prompt}"
            speak(response_text)
            
            try: 
                from memory_engine_v2 import memory_v2
                memory_v2.open_asset(path)
            except: pass
            
            return {"response": response_text}
        else:
            return {"response": f"I'm sorry sir, I couldn't generate that image: {result.get('error')}"}
            
    except Exception as e:
        print(f"[Image] Generator Error: {e}")
        return {"response": f"I encountered an error generating the image, sir."}


def generate_asset_title(prompt):
    """
    Asks the brain to generate a 2-3 word title for an asset based on the prompt.
    Used for filenames and memory descriptions. (v19)
    """
    try:
        title_prompt = (
            f"Task: Provide a concise 2-3 word title for a file based on this description: '{prompt}'.\n"
            f"Constraint: Output ONLY the 2-3 words. No punctuation, no quotes, no extra text."
        )
        response = ollama.chat(model=MODEL_BRAIN, messages=[{'role': 'user', 'content': title_prompt}])
        title = response['message']['content'].strip()
        
        # Clean up title for filename safety
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip().lower()
        return title if title else "unnamed_asset"
    except Exception as e:
        print(f"[TitleGen] Error: {e}")
        return "unnamed_asset"

def action_show_asset(text):
    """
    Search memory for an asset (image/pdf) and open it. (v17)
    """
    text_lower = text.lower()
    preferred_type = None
    if any(k in text_lower for k in ["image", "picture", "photo", "drawing"]):
        preferred_type = "image"
    elif any(k in text_lower for k in ["pdf", "report", "document", "medical file"]):
        preferred_type = "pdf"
        
    # Extract query
    query = text_lower.replace("show me", "").replace("open my", "").replace("the", "").replace("my", "").strip()
    print(f"[Memory] Searching for {preferred_type or 'asset'} using query: '{query}'")
    
    asset = memory_engine.find_asset(query, preferred_type=preferred_type)
    if asset:
        path = asset["path"]
        if os.path.exists(path):
            try:
                os.startfile(path)
                return f"Certainly, sir. Opening your {asset['type']} of {asset['description']}."
            except Exception as e:
                return f"I found the {asset['type']}, sir, but encountered an error opening the file."
        else:
            return f"I remember creating that {asset['type']}, sir, but the file is no longer at {path}."
    
    return "I'm sorry sir, I don't recall creating any images or reports matching that description."

def action_whatsapp(text):
    """
    Automates sending WhatsApp messages. (v20)
    1. Extracts recipient and message using Gemma.
    2. Looks up number in memory.
    3. Opens WhatsApp Web and auto-sends.
    """
    print(f"[WhatsApp] Processing request: {text}")
    speak("I'm preparing the WhatsApp message now, sir.")
    
    # 1. Extract Details using Gemma
    extract_prompt = (
        f"Task: Extract the recipient name and the EXACT message to send from this command: '{text}'.\n"
        f"Output format: RECIPIENT: [name] | MESSAGE: [message]\n"
        f"Example: 'text my mom i will be late' -> RECIPIENT: mom | MESSAGE: i will be late\n"
        f"Output ONLY the formatted string."
    )
    
    try:
        response = ollama.chat(model=MODEL_ROUTER, messages=[{'role': 'user', 'content': extract_prompt}])
        extraction = response['message']['content'].strip()
        print(f"[WhatsApp] Extraction: {extraction}")
        
        recipient = ""
        message = ""
        
        if "|" in extraction:
            parts = extraction.split("|")
            recipient = parts[0].replace("RECIPIENT:", "").strip().lower()
            message = parts[1].replace("MESSAGE:", "").strip()
        
        if not recipient or not message:
            return {"response": "I couldn't quite catch who to send the message to or what to say, sir."}
            
        # 2. Lookup Number
        number = memory_engine.get_contact_number(recipient)
        if not number:
            speak(f"I'm sorry sir, I don't have a phone number for {recipient} in my memory.")
            return {"response": f"Contact not found: {recipient}"}
            
        # 3. Open WhatsApp Web
        import urllib.parse
        encoded_msg = urllib.parse.quote(message)
        whatsapp_url = f"https://web.whatsapp.com/send?phone={number}&text={encoded_msg}"
        
        print(f"[WhatsApp] Opening: {whatsapp_url}")
        open_in_chrome(whatsapp_url) # Force Chrome (v21)
        
        # 4. Auto-Send (Optional/User requested)
        # Wait for page to load (WhatsApp Web can be slow)
        speak(f"Opening the chat with {recipient}. I'll send the message in a moment.")
        
        def auto_send():
            time.sleep(15) # Wait for WA Web to initialize
            pyautogui.press('enter')
            print("[WhatsApp] Auto-send: Enter pressed.")
            
        # Run auto-send in background or just wait
        import threading
        threading.Thread(target=auto_send, daemon=True).start()
        
        return {"response": f"I've opened the chat with {recipient} and I'm sending your message now, sir."}
        
    except Exception as e:
        print(f"[WhatsApp] Error: {e}")
        return {"response": "I encountered an error while trying to send the WhatsApp message, sir."}

def process_pdf_generation(text):
    """
    Generates a professional PDF report.
    1. Brain writes the content
    2. FPDF creates the file
    """
    print(f"[PDF] Processing request: {text}")
    speak("I'll get that report drafted for you immediately, sir.")
    
    # Extract topic - be more flexible with words
    topic = text.lower()
    # Broad cleanup of common triggers
    for trigger in ["make a pdf about", "make me a pdf about", "create a pdf about", 
                    "make a medical report about", "create a document about", 
                    "create a report about", "make a medical file about",
                    "make a report about", "create a report on", "generate a pdf for"]:
        topic = topic.replace(trigger, "")
    
    # Generic cleanup
    for word in ["pdf", "report", "document", "medical", "file", "make me a", "create a", "show me my"]:
        topic = topic.replace(word, "")
        
    topic = topic.strip().strip(".?!")
    if not topic:
        topic = "General Report"
        
    # 1. Ask Brain to write report
    prompt = (
        f"Task: Write a detailed, professional medical report for a doctor about: {topic}.\n"
        f"Context: The report is for Yousef. Use a professional medical tone.\n"
        f"Include: Symptoms, Potential Treatments, and Recommendations.\n"
        f"Length: Roughly 200-300 words.\n"
        f"Output ONLY the report text. No conversational filler."
    )
    
    try:
        response = ollama.chat(model=MODEL_BRAIN, messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content'].strip()
        
        # 2. Generate PDF
        pdf_dir = os.path.join(os.path.dirname(__file__), "pdf_reports")
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            
        # Generate Smart Title (v19)
        smart_title = generate_asset_title(topic)
        safe_filename = smart_title.replace(" ", "_").strip("_")
        
        filename = f"{safe_filename}_{int(time.time())}.pdf"
        output_path = os.path.join(pdf_dir, filename)
        
        print(f"[PDF] Attempting to create: {output_path}")
        result_path = create_medical_pdf(output_path, smart_title, content)
        
        if result_path and os.path.exists(result_path):
            # 3. Register in Memory (v19)
            memory_engine.add_asset("pdf", result_path, smart_title)
            
            # 4. Auto-open
            try:
                os.startfile(result_path)
            except Exception as e:
                print(f"[PDF] Could not auto-open: {e}")
                
            speak(f"PDF generation complete, sir. I've saved the report as {smart_title}.")
            return {"response": f"Report generated and saved as: {filename}"}
        else:
            return {"response": "I'm sorry sir, I drafted the text but failed to create the PDF file."}

    except Exception as e:
        print(f"[PDF] Error: {e}")
        return {"response": "I encountered an error while drafting the report, sir."}

def process_reasoning(text):
    """
    Core Chat Logic (Unified Gemma 4B + Memory)
    - Uses Memory Engine
    - Always uses the powerful 4B brain
    """
    print(f"[Reasoning] Thinking about: {text}")
    
    # 1. Get Memory Context (Relevant to current query)
    mem_context = memory_engine.get_context(text)
    
    # Ensure brain is pre-warmed
    model_manager.switch_to_brain()

    # 2. Build System Prompt
    system_prompt = (
        f"You are JARVIS, a professional AI assistant powered by the Qwen 2.5 Coder 7B offline model.\n"
        f"NEVER call yourself GPT-4 or OpenAI. User: Yousef.\n"
        f"Personality: Professional, concise, loyal, helpful.\n"
        f"{mem_context}\n"
        f"Keep answers short (under 2 sentences) unless asked for more details."
    )
    
    try:
        response = ollama.chat(
            model=MODEL_BRAIN,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            options={
                "temperature": 0.7,
                "repeat_penalty": 1.2,
                "repeat_last_n": 128,
                "num_predict": 512,
                "num_ctx": 4096,
                "top_k": 40,
                "top_p": 0.9
            }
        )
        
        import re
        reply = response['message']['content'].strip()
        reply = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', reply, flags=re.IGNORECASE)
        reply = re.sub(r'\b((?:\w+\s+){1,3}\w+)(\s+\1)+\b', r'\1', reply, flags=re.IGNORECASE)
        reply = re.sub(r'\s+', ' ', reply).strip()
        print(f"[Reasoning] {reply}")
        return {"response": reply}
        
    except Exception as e:
        print(f"[Reasoning] Error: {e}")
        return {"response": "I'm having trouble thinking clearly, sir."}


def process_command(text):
    """
    Unified Entry Point for all User Commands (Voice, UI, API).
    V24.5 Memory-First Cognitive Path.
    """
    if not text: return {"response": "I didn't hear anything, sir."}
    
    print(f"[Engine] 🧠 [V24.5] Processing Command: {text}")
    emit_state("thinking", text)
    emit_truth(text, "Interpreting user intent via Unified Brain...")

    try:
        # Use the V24.5 Cognitive Dispatcher
        result = dispatcher_v245.dispatch_sync(text)
        
        # Emit reasoning if available in result
        reasoning = result.get("reasoning", "Action complete.")
        emit_truth(text, reasoning, status="success", details=result)
        
        return result
    except Exception as e:
        print(f"[Engine Error] V24.5 Dispatch failure: {e}")
        emit_truth(text, f"Brain core failure: {str(e)}", status="failure")
        return {"response": "I encountered an error in my reasoning core, sir.", "error": str(e)}


# === JARVIS v22 MASTER UPGRADE - CORE LOOP ===

def conversation_mode_v22():
    """
    The J.A.R.V.I.S. v22 Master Loop.
    Uses whisper.cpp for low-latency STT (no SileroVAD).
    """
    print("\n" + "*"*40)
    print("  ENTERING MASTER CONVERSATION MODE (v22)")
    print("*"*40 + "\n")
    
    global audio, stream
    
    try:
        if audio is None:
            audio = pyaudio.PyAudio()
            
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=MIC_INDEX
        )
        print("[Hearing] Faster-Whisper STT initialized (conversation_mode_v22).")
    except Exception as e:
        print(f"[Hearing] Failed to init audio stream: {e}")
        speak("I'm having trouble with my hearing subroutines, sir.")
        return

    while True:
        try:
            # Listening for speech using whisper.cpp engine
            emit_state("listening")
            text = stt_engine.listen_and_transcribe(stream)
            
            if not text:
                continue
                
            text = text.strip()
            
            if not text:
                continue
                
            print(f"\n[User] {text}")
            
            # --- PHASE 1: PRE-PROCESSING & GLOBAL MISHEARING CORRECTIONS ---
            text_lower = text.lower().strip()
            replacements = {
                "oven": "open", "urban": "open", "open search": "search",
                "summer": "summary", "summit": "summary", "of england": ""
            }
            for word, replacement in replacements.items():
                if word in text_lower:
                    text_lower = text_lower.replace(word, replacement)

            # --- PHASE 2: PRIORITY 0 SHORTCUTS (Instant) ---
            # A. Exit Commands
            if any(cmd in text_lower for cmd in ["go to sleep", "stop listening", "goodbye jarvis", "that will be all"]):
                play_deactivation_sound()
                speak("Of course, sir. Standing by.")
                break
                
            # B. Mechanical Controls (Volume/Mute)
            if any(k in text_lower for k in ["volume", "mute", "unmute", "louder", "quieter"]):
                from automation_engine_v2 import automation_v2
                if "up" in text_lower or "louder" in text_lower: automation_v2.hotkey("command", "up")
                elif "down" in text_lower or "quieter" in text_lower: automation_v2.hotkey("command", "down")
                speak("Adjusting volume, sir.")
                continue

            # --- PHASE 3: PRIORITY 1 PERSONALITY (Greeting) ---
            if CREWAI_ROUTING_AVAILABLE:
                personality_matches = re.search(r'\b(hi|hello|hey jarvis|how are you|who are you|how are your|how are u|at your service)\b', text_lower)
                if personality_matches:
                    response = query_gemini(text)
                    speak(response, blocking=False)
                    continue

            # --- PHASE 4 & 5: V24.5 COGNITIVE PIPELINE ---
            # Replaces CrewAI, specialists, and fallbacks with the 6-step cognitive loop.
            try:
                print(f"[Engine] 🧠 Engaging V24.5 Cognitive Pipeline...")
                
                # Execute the full pipeline (Memory -> Plan -> Exec -> Review -> Update -> Respond)
                # Using Sync version for the main loop execution
                result = dispatcher_v245.dispatch_sync(text)
                
                response_text = result.get("response", "I'm not sure how to respond, sir.")
                speak(response_text)
                continue
                
            except Exception as e:
                print(f"[Engine Error] V24.5 Cognitive failure: {e}")
                speak("I encountered a glitch in my cognitive loop, sir.")
                continue

        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"[Master Loop] ❌ CRITICAL ERROR: {e}")
            traceback.print_exc() # Show exactly where it's breaking
            speak("I've encountered a glitch in my reasoning core, sir.")
            time.sleep(2)

    # Cleanup
    try:
        recorder.stop()
        recorder.close()
    except:
        pass
    print("\n[Mode] Exited Master Conversation Mode.\n")

# =============================================================================
# MAIN JARVIS EXECUTION
# =============================================================================

def run_jarvis():
    """Main interaction loop for J.A.R.V.I.S. v22"""
    global audio, stream, zero_level_count
    
    # Visual Header
    print("\n" + "="*60, flush=True)
    print("  J.A.R.V.I.S. v22 - ACTIVE LISTENING MODE", flush=True)
    print("="*60, flush=True)
    print(f"  Voice Engine: Coqui XTTS V3 (READY)", flush=True)
    print(f"  Wake Word: 'Hey Jarvis' (READY)", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Audio Device Discovery - helps user find correct MIC_INDEX
    print("[Audio] Listing all audio devices...")
    _temp_audio = pyaudio.PyAudio()
    for i in range(_temp_audio.get_device_count()):
        dev = _temp_audio.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"  [ID {i}] {dev.get('name')} (Inputs: {dev.get('maxInputChannels')})")
    _temp_audio.terminate()
    print(f"[Audio] Selected MIC_INDEX: {MIC_INDEX}")
    print()
    
    while True:
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, 
                                input=True, frames_per_buffer=CHUNK, input_device_index=MIC_INDEX)
            print("[Mode] Waiting for 'Hey Jarvis'...")
            break
        except Exception as e:
            print(f"[Audio] Failed to open mic ({e}). Retrying in 3 seconds...")
            time.sleep(3)
            if audio:
                try: audio.terminate()
                except: pass
            audio = None
    
    try:
        while True:
            # Check if system is busy with heavy tasks
            if IS_BUSY:
                time.sleep(1)
                continue

            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # === ZERO LEVEL DETECTION ===
                level = np.abs(audio_np).mean()

                if level < 5:
                    zero_level_count += 1
                    if zero_level_count > 100: # ~2 seconds of zeros
                        print("\n[Audio] ⚠️ Mic appears disconnected (all zeros), waiting for reconnect...")
                        raise OSError("Mic appears disconnected")
                else:
                    zero_level_count = 0
                    
            except OSError as e:
                print(f"[Audio] Host Error (likely device busy or disconnected): {e}")
                time.sleep(3) # Longer wait for hardware to settle
                try:
                    stream.stop_stream()
                    stream.close()
                except: pass
                
                # Re-initialize PyAudio entirely for fresh hardware handshake
                print("[Audio] Re-initializing PyAudio hardware...")
                try:
                    audio.terminate()
                except: pass
                
                # Persistent retry loop for mic reconnect
                while True:
                    try:
                        audio = pyaudio.PyAudio()
                        stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, 
                                            input=True, frames_per_buffer=CHUNK, input_device_index=MIC_INDEX)
                        emit_sys_log("[Audio] [FIXED] Hardware re-initialized.", "INFO")
                        zero_level_count = 0
                        break
                    except Exception as ex:
                        emit_sys_log(f"[Audio] Still waiting for mic... ({ex})", "WARN")
                        time.sleep(3)
                continue
            except Exception as e:
                print(f"[Audio] Read Error: {e}")
                continue

            

            # Get wake word score

            score = 0

            if wake_model:

                score = wake_model.predict(audio_np).get('hey_jarvis', 0)

            else:

                # In minimal mode, we skip wake word

                time.sleep(0.1)

                continue

            

            if score > 0.01:
                emit_sys_log(f"Wake Score: {score:.2f} (Target: {WAKE_WORD_SENSITIVITY})", "DEBUG")

            # Threshold = configurable sensitivity
            if score >= WAKE_WORD_SENSITIVITY:
                emit_sys_log(f"JARVIS activated! ({score:.2f})", "INFO")

                if wake_model:
                    wake_model.reset()

                # Stop and close stream before speaking to prevent hardware conflicts
                try:
                    if stream:
                        stream.stop_stream()
                        stream.close()
                except Exception as e:
                    print(f"[Wake] Stream closure warning: {e}")
                finally:
                    stream = None

                # INSTANT RESPONSE - Use cached greeting
                speak("Yes sir?")

                # Enter conversation mode (BLOCKING - DEAF DURING ACTION)

                # Enter conversation mode (BLOCKING - DEAF DURING ACTION)
                try:
                    conversation_mode_v22()
                except Exception as e:
                    print(f"[Conversation] Master mode failure: {e}")
                
                # Reopen stream for wake word detection (After action complete)

                # Reopen stream for wake word detection (After action complete)
                time.sleep(0.3)

                

                # Reset wake model FIRST to clear any buffered audio

                if wake_model:

                    wake_model.reset()

                

                # Reopen audio stream
                try:
                    stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, 
                                        input=True, frames_per_buffer=CHUNK, input_device_index=MIC_INDEX)
                    emit_sys_log("[Audio] [OK] Stream reopened", "DEBUG")
                except Exception as e:
                    emit_sys_log(f"[Audio] Reopening stream failed: {e}, recreating PyAudio...", "WARN")
                    try:
                        audio.terminate()
                    except:
                        pass
                    audio = pyaudio.PyAudio()
                    stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, 
                                        input=True, frames_per_buffer=CHUNK, input_device_index=MIC_INDEX)
                    emit_sys_log("[Audio] [OK] Stream recreated", "INFO")
                

                # Flush any stale audio data

                for _ in range(5):

                    try:

                        stream.read(CHUNK, exception_on_overflow=False)

                    except:

                        pass

                
                print("\n[Mode] Waiting for 'Hey Jarvis'...")

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping JARVIS...")
    finally:
        try:
            if 'audio' in locals() and audio: audio.terminate()
            if 'stream' in locals() and stream: stream.close()
        except: pass
        print("[Shutdown] Goodbye!")
        # sys is imported at top level, but making it explicit for safety in finally
        import sys
        sys.exit(0)

# Consolidating startup at the end of file for API integration

# =============================================================================

# FASTAPI / WEBSOCKET SERVER - For Electron App Integration

# =============================================================================



class ConnectionManager:

    def __init__(self):

        self.active_connections: list[WebSocket] = []



    async def connect(self, websocket: WebSocket):

        await websocket.accept()

        self.active_connections.append(websocket)



    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)



    async def broadcast(self, message: dict):

        for connection in self.active_connections[:]:

            try:

                await connection.send_json(message)

            except Exception as e:

                print(f"[WS] Broadcast failed for a client: {e}")

                self.disconnect(connection)



manager = ConnectionManager()



@app.websocket("/ws")

async def websocket_endpoint(websocket: WebSocket):

    await manager.connect(websocket)

    try:

        while True:

            data = await websocket.receive_text()

            message = json.loads(data)

            response = await process_ws_message(message, websocket)

            if response:

                await websocket.send_json(response)

    except WebSocketDisconnect:

        manager.disconnect(websocket)

    except Exception as e:

        print(f"[WS] Error: {e}")

        manager.disconnect(websocket)



async def process_ws_message(message: dict, websocket: WebSocket) -> dict:

    """Process incoming WebSocket messages from Electron app."""

    msg_type = message.get("type")

    

    if msg_type == "command" or msg_type == "voice_command":

        transcript = message.get("transcript", "")

        # Unified processing: Python Router first, then Gemini if complex

        result = process_command(transcript)

        return {"type": "executed", "result": result}

    

    elif msg_type == "snapshot" or msg_type == "snapshot_command":

        # Vision task from Electron app

        image_data = message.get("image")

        transcript = message.get("transcript", "")

        context = message.get("context", {})

        

        if llm_client:

            print("[WS] Processing Vision/Snapshot with Local LLaVA...")

            actions = await llm_client.get_actions(transcript, context, image_data)

            return {"type": "executed", "result": {"actions": actions}}

        return {"type": "error", "message": "Vision client not available"}



    elif msg_type == "code_context":

        # Coding assistant logic

        code_ctx = message.get("context", {})

        if llm_client:

            suggestions = await llm_client.get_code_suggestions(code_ctx)

            return {"type": "suggestions", "result": suggestions}

        return {"type": "error", "message": "LLM client not available"}



    elif msg_type == "automation":

        # Direct automation command from Electron frontend

        result = handle_automation_command(message.get("command", {}))

        return {"type": "automation_result", "result": result}

    

    elif msg_type == "ping":

        return {"type": "pong", "time": time.time()}

    

    return {"type": "ack", "received": msg_type}



def handle_automation_command(cmd: dict) -> dict:

    """Handle PyAutoGUI automation commands."""

    action = cmd.get('action')

    try:

        if action == 'click':

            x, y = cmd.get('x'), cmd.get('y')

            if x is not None and y is not None: pyautogui.click(x, y)

            else: pyautogui.click()

            return {'success': True}

        elif action == 'type':

            pyautogui.write(cmd.get('text', ''), interval=0.02)

            return {'success': True}

        elif action == 'pressKey':

            pyautogui.press(cmd.get('key', '').lower())

            return {'success': True}

        elif action == 'hotkey':

            pyautogui.hotkey(*cmd.get('keys', []))

            return {'success': True}

        elif action == 'screenshot':

            if not SCREENSHOT_ENABLED:

                return {'success': False, 'error': 'Advanced screenshots (mss) not available.'}

            with mss.mss() as sct:

                monitor = sct.monitors[1]

                img = sct.grab(monitor)

                pil_img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')

                buffer = BytesIO()

                pil_img.save(buffer, format='PNG')

                b64 = base64.b64encode(buffer.getvalue()).decode()

                return {'success': True, 'image': b64}

        # Add more actions as needed

        return {'success': False, 'error': f'Unsupported action: {action}'}

    except Exception as e:

        return {'success': False, 'error': str(e)}



@app.post("/api/automation")

async def api_automation(request: dict):

    return handle_automation_command(request)



class CommandRequest(BaseModel):
    text: str

@app.post("/api/command")

async def api_command(request: CommandRequest):

    print(f"[API] Command: {request.text}")

    result = process_command(request.text)

    return {"success": True, "result": result}



@app.post("/api/speak")

async def api_speak(request: CommandRequest):

    if JARVIS_VOICE_ENABLED:

        speak(request.text)

        return {"success": True}

    return {"error": "Voice disabled"}



@app.get("/health")

async def health():

    return {"status": "ok", "jarvis": "running"}



class ThreeDRequest(BaseModel):

    text: str

    image_data: str | None = None  # Base64 image data for reference

    unity_path: str | None = None

    params: dict | None = None



class NoteGenerateRequest(BaseModel):
    text: str

class LiveAudioRequest(BaseModel):
    audioData: str
    mimeType: str = "audio/webm"

class LiveFrameRequest(BaseModel):
    imageData: str
    context: str = ""

class FinalizeRequest(BaseModel):
    transcript: str = ""
    visualNotes: str = ""

class SaveNoteRequest(BaseModel):
    title: str
    notes: dict
    transcript: str = ""
    visualNotes: str = ""

DESIGN_SYSTEM_PROMPT = """You are JARVIS, a NASA-Grade Aerospace Systems Engineer.

You convert natural language into precise structural parameters.



RULE #1: You ONLY use these geometric primitives:

- box: [x, y, z] (main bodies, housings)

- cylinder: [radius, height] (arms, tanks, rotors)

- tube: [outer_r, inner_r, height] (pipes, conduits)

- panel: [x, y, thickness] (plates, solar wings)

- sphere: [radius] (sensors, joints)

- wing: [span, chord, thickness] (aero surfaces)

- nozzle: [radius, length] (engines)

- cone: [radius, height] (nosecones)



RULE #2: OUTPUT ONLY VALID JSON.

Reject any conversational text. No "Right away sir," no "Here is the design."

Only the JSON object.



OUTPUT SCHEMA:

{

  "project": "ProjectName",

  "detail_level": 5,

  "symmetry": true,

  "material_override": "nasa_white",

  "systems": {

    "structures": { "complexity": 1.0, "scale": 1.0 },

    "propulsion": { "complexity": 1.0, "scale": 1.0 },

    "avionics": { "complexity": 1.0, "scale": 1.0 }

  }

}

"""



@app.post("/api/generate_3d_model")

async def api_generate_3d(request: ThreeDRequest):

    """FastAPI endpoint for 3D structural generation"""

    print(f"[API] 3D Generation Requested: {request.text[:50]}...")

    try:

        image_path = None

        if request.image_data:

            # Save temp image for LLaVA

            try:

                image_path = "_temp_3d_ref.png"

                img_bytes = base64.b64decode(request.image_data.split(",")[-1])

                with open(image_path, "wb") as f:

                    f.write(img_bytes)

            except Exception as e:

                print(f"[API] Image decode failed: {e}")

                image_path = None



        async def progress_reporter(status):

            try:

                await manager.broadcast({"type": "3d_progress", "status": status})

            except:

                pass



        # Use persisted parameters if the request didn't provide specific ones

        # FIX: Check if params is actually populated, not just an empty dict

        has_params = request.params and len(request.params) > 0

        gen_params = request.params if has_params else LAST_DESIGN_PARAMS

        

        print(f"[API] Dispatching to process_3d_generation. has_request_params={has_params}, has_persisted={len(LAST_DESIGN_PARAMS)>0}")

        result = await process_3d_generation(request.text, image_path, progress_reporter, request.unity_path, gen_params)

        

        # Clean up temp image if it was created

        if image_path and os.path.exists(image_path):

            try: os.remove(image_path)

            except: pass

            

        return result

    except Exception as e:

        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "response": "I encountered a critical error during 3D assembly."}

@app.get("/status")
async def status():
    return {
        "status": "ok",
        "brain": MODEL_BRAIN,
        "voice": JARVIS_VOICE_ENABLED,

        "automation": AUTOMATION_ENABLED,

        "memory": MEMORY_ENABLED

    }



@app.get("/api/latest_model")

async def latest_model():

    """Get the last successfully generated model to persist session state."""

    if LAST_GENERATED_MODEL:

        return LAST_GENERATED_MODEL

    return {"success": False, "error": "No model generated yet in this session."}





@app.post("/api/chat")
async def api_chat(request: CommandRequest):
    # Legacy endpoint - Chat is now SocketIO
    return {"status": "Use WebSocket for chat"}

# --- Note Engine Endpoints ---

@app.post("/api/notes/audio")
async def api_notes_audio(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    transcript = "Audio processing is handled via local Whisper engine."
    notes = await jarvis_note_engine.generate_structured_notes(transcript)
    return {"transcript": transcript, "notes": notes}

@app.post("/api/notes/image")
async def api_notes_image(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    notes = await jarvis_note_engine.analyze_image_notes(img_b64, "Extract handwritten notes and organize them.")
    return {"notes": notes}

@app.post("/api/notes/screenshot")
async def api_notes_screenshot(screenshot: UploadFile = File(...)):
    img_bytes = await screenshot.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    notes = await jarvis_note_engine.analyze_image_notes(img_b64, "Extract key points from this screenshot.")
    return {"notes": notes}

@app.post("/api/notes/generate")
async def api_notes_generate(request: NoteGenerateRequest):
    notes = await jarvis_note_engine.generate_structured_notes(request.text)
    return {"notes": notes}

@app.post("/api/live/audio")
async def api_live_audio(request: LiveAudioRequest):
    notes = await jarvis_note_engine.process_live_audio_chunk(request.audioData)
    return {"notes": notes}

@app.post("/api/live/frame")
async def api_live_frame(request: LiveFrameRequest):
    notes = await jarvis_note_engine.process_live_frame(request.imageData, request.context)
    return {"notes": notes}

@app.post("/api/live/finalize")
async def api_live_finalize(request: FinalizeRequest):
    notes = await jarvis_note_engine.finalize_session(request.transcript, request.visualNotes)
    return {"notes": notes}

@app.post("/api/notes/save")
async def api_notes_save(request: SaveNoteRequest):
    filename = jarvis_note_engine.save_note(request.title, request.notes, request.transcript, request.visualNotes)
    return {"success": True, "filename": filename}

@app.get("/api/notes/saved")
async def api_notes_list():
    notes = jarvis_note_engine.list_notes()
    return {"notes": notes}

@app.get("/api/notes/saved/{filename}")
async def api_notes_get(filename: str):
    note = jarvis_note_engine.get_note(filename)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

    """Unified chat endpoint."""

    print(f"[API] Chat: {request.text}")

    result = process_command(request.text)

    return {"success": True, "result": result}



@app.post("/api/vision")

async def api_vision(request: CommandRequest):

    """Vision analysis endpoint."""

    print(f"[API] Vision requested: {request.text}")

    if ENHANCED_VISION_ENABLED:

        result = enhanced_vision(request.text)

        return {"success": True, "result": result}

    return {"success": False, "error": "Enhanced vision not enabled"}



@app.get("/api/health")

async def api_health():

    """Detailed health check."""

    return {

        "status": "ok",

        "brain": MODEL_BRAIN,

        "vision": ENHANCED_VISION_ENABLED,

        "structural_3d": STRUCTURAL_3D_ENABLED

    }




# =============================================================================
# FEATURE HUB API ENDPOINTS
# =============================================================================

class VisionRequest(BaseModel):
    image: str # Base64
    prompt: str = "Describe this image"

@app.post("/api/vision/describe")
async def api_vision_describe(request: VisionRequest):
    """Image Hub Description Endpoint"""
    print(f"[API] Vision Describe: {request.prompt}")
    
    try:
        # Decode image
        if "base64," in request.image:
            img_data = base64.b64decode(request.image.split("base64,")[1])
        else:
            img_data = base64.b64decode(request.image)
            
        # Save temp
        temp_path = "temp_vision_upload.png"
        with open(temp_path, "wb") as f:
            f.write(img_data)
            
        # Use Vision Engine (which uses ModelManager)
        from vision_engine import vl_describe, unload_vl_model
        
        # 1. Switch directly via manager if needed, or rely on engine
        # Engine calls manager.switch_to_vision() now.
        
        description = vl_describe(temp_path, request.prompt)
        
        # 2. Return to Brain
        unload_vl_model() # calls Switch to Brain
        
        # Cleanup
        try: os.remove(temp_path)
        except: pass
        
        return {"success": True, "description": description}
        
    except Exception as e:
        print(f"[API] Error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/parts/analyze")
async def api_parts_analyze(request: VisionRequest):
    """Parts Analyzer Endpoint - Same logic but specialized prompt handling"""
    print(f"[API] Parts Review: {request.prompt}")
    
    # Just reuse the describe logic for now, frontend sends specific prompts
    return await api_vision_describe(request)






import asyncio
import uvicorn

api_loop = None

def start_fastapi_server():
    global api_loop
    api_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(api_loop)
    
    # Initialize Logger Refs
    set_logger_refs(sio, api_loop)
    
    from config import HOST, API_PORT
    
    # Run Telemetry Worker in the background
    # Assuming telemetry_worker is defined or imported elsewhere
    # api_loop.create_task(telemetry_worker()) 
    
    print(f"[API] Unified Backend running on http://{HOST}:{API_PORT}")
    
    # Disable uvicorn's logging that conflicts with our manual redirection
    config = uvicorn.Config(socket_app, host=HOST, port=API_PORT, log_level="warning", log_config=None)
    server = uvicorn.Server(config)
    api_loop.run_until_complete(server.serve())


# VISION TO IMAGE GENERATION
# =============================================================================

@app.post("/api/vision/generate-image")
async def api_vision_generate_image(request: VisionRequest):
    """
    Vision -> Prompt -> Image Generation Pipeline
    1. Vision analyzes image (VL)
    2. Suggests prompt (VL)
    3. Generates Image (SD)
    """
    print(f"[API] Vision-to-Image: {request.prompt}")
    
    if not IMAGE_GEN_ENABLED:
        return {"success": False, "error": "Image Generation module disabled"}

    try:
        # 1. Decode Image
        if "base64," in request.image:
            img_data = base64.b64decode(request.image.split("base64,")[1])
        else:
            img_data = base64.b64decode(request.image)
            
        temp_path = os.path.abspath("temp_gen_source.png")
        with open(temp_path, "wb") as f:
            f.write(img_data)

        # 2. RUN VISION ANALYSIS (Qwen3-VL)
        # We ask VL to describe it specifically for an image generator
        prompt_for_vl = f"Describe this image in detail so I can recreate it with an AI image generator. Focus on visual style, lighting, and composition. User additional request: {request.prompt}"
        
        from vision_engine import vl_describe, unload_vl_model
        
        print("[API] 1/3 Analyzing source image...")
        vl_response = vl_describe(temp_path, prompt_for_vl)
        
        if not vl_response:
            return {"success": False, "error": "Vision analysis failed"}
            
        # 3. UNLOAD VISION & BRAIN (Clear VRAM)
        print("[API] 2/3 Clearing VRAM for Image Gen...")
        # model_manager is available inside vision_engine, but we can access the instance here if imported
        # But vision_engine imports it. 
        # Easier: unload_vl_model() returns to Brain. We DON'T want that.
        # We need to force unload.
        
        from model_manager import model_manager
        model_manager.unload_all() 
        
        # 4. GENERATE IMAGE (Stable Diffusion)
        from image_generator import generate_image, unload_model as unload_sd
        
        final_prompt = vl_response
        print(f"[API] 3/3 Generating Image: {final_prompt[:50]}...")
        
        paths = generate_image(prompt=final_prompt, count=1, mode="fast")
        
        # 5. Cleanup
        unload_sd() # Unload SD
        # model_manager.switch_to_brain() # Optional: Reload brain now or let next request do it
        # User said: "vision unloaded once they finish"
        # We'll leave it empty (Brain loads on demand anyway)
        
        try: os.remove(temp_path)
        except: pass
        
        if paths:
            # Convert absolute path to URL
            # Server mounts 'generated_images' ? No, verify mount
            # Existing mount: app.mount("/models", ...) 
            # We need to verify if /images is mounted or just return path for local
            # Let's return local path for now, frontend might need `file://` or we add a mount
            
            # QUICK FIX: Mount the generated_images directory if not mounted
            # We can't easily add mount at runtime.
            # But likely server.py has extensive mounts.
            # We'll return full path
            return {"success": True, "image_path": str(paths[0]), "prompt_used": final_prompt}
        else:
            return {"success": False, "error": "Generation failed"}

    except Exception as e:
        print(f"[API] Error: {e}")
        return {"success": False, "error": str(e)}






# 3D CREATOR STUDIO API
# =============================================================================

class ThreeDGenRequest(BaseModel):
    prompt: str

class ImageGenRequest(BaseModel):
    prompt: str

class ThreeDAnalyzeRequest(BaseModel):
    image: str # Base64

@app.post("/api/image/generate")
async def api_image_generate(request: ImageGenRequest):
    print("--------------------------------------------------")
    print(f"[API] 🔴 RECEIVED IMAGE GEN REQUEST: {request.prompt}")
    print("--------------------------------------------------")
    from image_generator import process_image_request
    try:
        # Run in threadpool to avoid blocking async loop
        result = await asyncio.to_thread(process_image_request, request.prompt)
        print(f"[API] 🟢 GENERATION COMPLETE: {result.get('success')}")
        return result
    except Exception as e:
        print(f"[API] Image Gen Failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/system/models")
async def api_system_models():
    from config import MODEL_BRAIN, MODEL_ROUTER, MODEL_EXPERT, MODEL_VISION, WHISPER_MODEL
    return {
        "brain": MODEL_BRAIN,
        "routing": MODEL_ROUTER,
        "expert": MODEL_EXPERT,
        "vision": MODEL_VISION,
        "stt": WHISPER_MODEL,
        "image": "SDXL Lightning (DreamShaper XL)"
    }

@app.post("/api/3d/generate")
async def api_3d_generate(request: ThreeDGenRequest):
    print(f"[API] 3D Generation Request: {request.prompt}")
    from threed_llm import threed_llm
    
    try:
        # Run in threadpool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        glb_path = await loop.run_in_executor(None, threed_llm.generate_from_prompt, request.prompt)
        return {"success": True, "model_path": str(glb_path)}
    except Exception as e:
        print(f"[API] 3D Gen Failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/3d/analyze")
async def api_3d_analyze(request: ThreeDAnalyzeRequest):
    print(f"[API] 3D Analysis Request")
    from vision_engine_v3 import vision_v3
    
    try:
        # Run in threadpool
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(None, vision_v3.analyze_base64, request.image, "Analyze this 3D model screenshot. Describe the structure, potential materials, and engineering purpose.")
        return {"success": True, "description": analysis}
    except Exception as e:
        print(f"[API] 3D Analysis Failed: {e}")
        return {"success": False, "error": str(e)}

print("[Main] Reached entry point")
if __name__ == "__main__":
    from config import HOST, API_PORT
    import uvicorn
    
    print(f"[Main] Starting J.A.R.V.I.S. on http://{HOST}:{API_PORT}")
    uvicorn.run(socket_app, host=HOST, port=API_PORT, log_level="info")



# =============================================================================
