"""
JARVIS Vision Engine V3 - Hybrid Efficiency Controller (PaddleOCR + OpenCV)
==========================================================================
STRATEGY: "The Scout & The Professor"
1. Scout (PaddleOCR + OpenCV): Runs fast (every 3s). Detects text/layout changes.
2. Professor (VLM): Runs deeply (only on change). Analyzes context/code.

UPGRADE from V2:
- EasyOCR → PaddleOCR (faster, more accurate on screens)
- Added OpenCV preprocessing (grayscale + adaptive threshold) for slide text
- Qwen3-VL 8B as the Professor for deep visual reasoning
"""

import os
import sys
import time
import base64
import asyncio
import difflib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
from PIL import Image
import requests
import json
import logging

# Silence PaddleOCR spam
logging.getLogger("ppocr").setLevel(logging.ERROR)

from config import MODEL_VISION

# --- CONFIGURATION ---
VLM_MODEL = MODEL_VISION  # The "Professor"
OCR_CONFIDENCE_THRESHOLD = 0.5
CHANGE_THRESHOLD = 0.8     # < 80% similarity triggers VLM
MIN_VLM_INTERVAL = 10      # Seconds between VLM calls (cooldown)
FORCE_VLM_INTERVAL = 60    # Seconds to force VLM even if static

class VisualChangeDetector:
    """The Scout: Detects if the screen has changed meaningfully."""
    
    def __init__(self):
        self.last_text = ""
        self.last_check_time = 0
        
    def has_changed(self, current_text: str) -> bool:
        """Compares current OCR text with previous frame."""
        curr = " ".join(current_text.split()).lower()
        prev = " ".join(self.last_text.split()).lower()
        
        if not prev:
            self.last_text = current_text
            return True  # First run always triggers
            
        similarity = difflib.SequenceMatcher(None, curr, prev).ratio()
        has_changed = similarity < CHANGE_THRESHOLD
        
        if has_changed:
            print(f"[Vision Scout] 👁️ Change Detected! Similarity: {similarity:.2f}")
            self.last_text = current_text
            
        return has_changed

class HybridVisionEngine:
    """The Controller: Manages PaddleOCR + OpenCV and VLM orchestration."""
    
    def __init__(self):
        self.scout = VisualChangeDetector()
        self.paddle_reader = None
        self.last_vlm_run = 0
        self.screenshot_path = Path(__file__).parent / "current_screen_v3.png"
        self._init_ocr()
        
    def _init_ocr(self):
        """Initialize PaddleOCR (replaces EasyOCR for better screen reading)."""
        try:
            from paddleocr import PaddleOCR
            print("[Vision V3] Loading Scout (PaddleOCR + OpenCV)...")
            self.paddle_reader = PaddleOCR(
                use_angle_cls=True,   # Handle rotated text
                lang='en',
                show_log=False
            )
            print("[Vision V3] ✅ Scout Ready (PaddleOCR)")
        except Exception as e:
            print(f"[Vision V3] ❌ PaddleOCR Init Failed: {e}")
            print("[Vision V3] ⚠️ OCR will not be available.")
            self.paddle_reader = None

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        OpenCV preprocessing to enhance text readability for OCR.
        Critical for slides with colored backgrounds or low contrast.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Adaptive thresholding (handles varying backgrounds on slides)
        # This makes dark text on any background appear as black on white
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # 3. Light denoise (removes speckle without losing text)
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        
        return denoised

    def capture_screen(self) -> str:
        """Capture screen to file."""
        try:
            subprocess.run(["screencapture", "-x", str(self.screenshot_path)], check=True)
            return str(self.screenshot_path)
        except Exception as e:
            print(f"[Vision V3] Capture failed: {e}")
            return ""

    def _run_scout(self, image_path: str) -> str:
        """Run PaddleOCR with OpenCV preprocessing to get raw text."""
        if self.paddle_reader:
            try:
                # Preprocess with OpenCV for better text extraction
                preprocessed = self._preprocess_image(image_path)
                
                if preprocessed is not None:
                    # Run PaddleOCR on preprocessed image
                    results = self.paddle_reader.ocr(preprocessed, cls=True)
                else:
                    # Fallback: run on original image
                    results = self.paddle_reader.ocr(image_path, cls=True)
                
                # Extract text from PaddleOCR results
                texts = []
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]       # The recognized text
                            confidence = line[1][1]  # Confidence score
                            if confidence > OCR_CONFIDENCE_THRESHOLD:
                                texts.append(text)
                
                return " ".join(texts)
            except Exception as e:
                print(f"[Vision V3] PaddleOCR failed: {e}")
                return ""
        return ""

    def _run_professor(self, image_path: str, prompt: str = "Describe the UI and analyze any code.") -> str:
        """Send frame to Ollama VLM (Qwen3-VL 8B)."""
        print(f"[Vision Professor] 🧠 Analyzing frame with {VLM_MODEL}...")
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return self._call_vlm(img_b64, prompt)
                
        except Exception as e:
            print(f"[Vision Professor] ❌ Failed: {e}")
            return ""

    def analyze_base64(self, base64_image: str, prompt: str = "Describe this 3D model in detail.") -> str:
        """Analyze a base64 encoded image directly."""
        # Clean header if present
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
            
        print(f"[Vision Professor] 🧠 Analyzing 3D Frame...")
        return self._call_vlm(base64_image, prompt)

    def _call_vlm(self, img_b64: str, prompt: str) -> str:
        """Internal helper to call Ollama."""
        try:
            payload = {
                "model": VLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }],
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Precise
                    "num_predict": 256,   # Concise
                    "repeat_penalty": 1.1 # No repetition
                }
            }
            
            res = requests.post("http://localhost:11434/api/chat", json=payload)
            if res.status_code == 200:
                response = res.json()
                content = response.get("message", {}).get("content", "")
                print(f"[Vision Professor] ✅ Analysis: {content[:100]}...")
                return content
            else:
                print(f"[Vision Professor] ❌ API Error: {res.text}")
                return "Analysis failed due to API error."
        except Exception as e:
            print(f"[Vision Professor] ❌ Call Failed: {e}")
            return f"Analysis failed: {str(e)}"

    def analyze(self) -> Dict:
        """
        Main Loop:
        1. Capture
        2. Scout (PaddleOCR + OpenCV)
        3. If Changed -> Professor (VLM)
        4. Return combined insight.
        """
        path = self.capture_screen()
        if not path: return {"error": "Capture failed"}
        
        # 1. Scout Run
        ocr_text = self._run_scout(path)
        
        # 2. Decision Logic
        now = time.time()
        time_since_vlm = now - self.last_vlm_run
        should_trigger_vlm = False
        
        is_changed = self.scout.has_changed(ocr_text)
        
        if is_changed and time_since_vlm > MIN_VLM_INTERVAL:
            should_trigger_vlm = True
            reason = "Visual Change"
        elif time_since_vlm > FORCE_VLM_INTERVAL:
            should_trigger_vlm = True
            reason = "Force Interval"
            
        vlm_analysis = ""
        
        # 3. Professor Run
        if should_trigger_vlm:
            print(f"[Vision V3] 🚀 Triggering VLM [Reason: {reason}]")
            prompt = "Analyze this screen. If there is code, explain the logic. If it is a slide, summarize the key points."
            vlm_analysis = self._run_professor(path, prompt)
            self.last_vlm_run = now
            
        return {
            "timestamp": now,
            "ocr_text": ocr_text,
            "vlm_analysis": vlm_analysis,
            "triggered_vlm": should_trigger_vlm,
            "image_path": path
        }

# Singleton
vision_v3 = HybridVisionEngine()
