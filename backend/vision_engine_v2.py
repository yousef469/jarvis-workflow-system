"""
JARVIS Vision Engine V2 - Lightweight Screen Analysis
=====================================================
- EasyOCR for text detection
- YOLO for icon/UI element detection
- macOS native screencapture for precise app capture
- Reasoning is delegated to the primary brain (Gemma 1B)
"""
import os
import sys
import time
import base64
import io
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# OmniParser path setup
# Try finding backend-relative first, then project-root relative
_current_dir = Path(__file__).resolve().parent
if (_current_dir / "OmniParser").exists():
    OMNIPARSER_DIR = _current_dir / "OmniParser"
else:
    OMNIPARSER_DIR = _current_dir.parent / "OmniParser"

sys.path.insert(0, str(OMNIPARSER_DIR))
print(f"[Vision V2] OmniParser Directory: {OMNIPARSER_DIR}")

# Models (lazy loaded)
_yolo_model = None
_easyocr_reader = None
OMNIPARSER_AVAILABLE = False


def _init_easyocr():
    """Lazy initialize EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    
    try:
        import easyocr
        print("[Vision V2] Loading EasyOCR (English)...")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        print("[Vision V2] ✅ EasyOCR ready")
        return _easyocr_reader
    except Exception as e:
        print(f"[Vision V2] EasyOCR init failed: {e}")
        return None


def _init_yolo():
    """Lazy initialize YOLO model."""
    global _yolo_model, OMNIPARSER_AVAILABLE
    
    if _yolo_model is not None:
        return True
    
    try:
        from ultralytics import YOLO
        
        weights_dir = OMNIPARSER_DIR / "weights"
        icon_detect_path = weights_dir / "icon_detect" / "model.pt"
            
        if not icon_detect_path.exists():
            print(f"[Vision V2] YOLO weights not found at {weights_dir / 'icon_detect'}")
            return False
        
        print("[Vision V2] Loading YOLO model for icon detection...")
        # task="detect" avoids the "unable to guess task" warning
        _yolo_model = YOLO(str(icon_detect_path), task="detect")
        OMNIPARSER_AVAILABLE = True
        print("[Vision V2] ✅ YOLO ready")
        return True
    except Exception as e:
        print(f"[Vision V2] YOLO init failed: {e}")
        return False


class JarvisVision:
    def __init__(self, parser_dir: str = None):
        """
        Initializes the vision engine.
        Uses EasyOCR + YOLO for detection.
        """
        self.parser_dir = Path(parser_dir) if parser_dir else OMNIPARSER_DIR
        self.screenshot_path = Path(__file__).parent / "current_screen.png"
        print(f"[Vision V2] Vision Engine ready. OmniParser at: {self.parser_dir}")

    def pre_warm(self):
        """Pre-load OCR and YOLO models to avoid first-run delay."""
        print("[Vision V2] Pre-warming models...")
        _init_easyocr()
        _init_yolo()
        print("[Vision V2] ✅ Pre-warming complete.")

    def capture_screen(self, region: str = "full") -> str:
        """Takes a full-screen screenshot using macOS native tool."""
        print("[Vision V2] Capturing screen via macOS screencapture...")
        try:
            # -x: silent (no shutter sound)
            # This captures apps and windows properly on macOS
            subprocess.run(["screencapture", "-x", str(self.screenshot_path)], check=True)
            
            if region == "center":
                screenshot = Image.open(self.screenshot_path)
                w, h = screenshot.size
                left = int(w * 0.2)
                top = int(h * 0.2)
                right = int(w * 0.8)
                bottom = int(h * 0.8)
                screenshot = screenshot.crop((left, top, right, bottom))
                screenshot.save(str(self.screenshot_path))
                
            print(f"[Vision V2] Screenshot saved to {self.screenshot_path}")
            return str(self.screenshot_path)
        except Exception as e:
            print(f"[Vision V2] Screenshot failed: {e}")
            # Fallback to pyautogui if screencapture fails
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save(str(self.screenshot_path))
            return str(self.screenshot_path)

    def _run_ocr(self, image_path: str) -> Tuple[List[str], List]:
        """Run EasyOCR on image and return text + bounding boxes."""
        reader = _init_easyocr()
        if reader is None:
            return [], []
        
        print("[Vision V2] Running EasyOCR text detection...")
        import numpy as np
        image = Image.open(image_path)
        image_np = np.array(image)
        
        results = reader.readtext(image_np, paragraph=False)
        
        texts = [r[1] for r in results]
        bboxes = [r[0] for r in results]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        
        print(f"[Vision V2] ✅ OCR detected {len(texts)} text elements")
        return texts, bboxes

    def _run_yolo(self, image_path: str) -> List[Dict]:
        """Run YOLO for icon/element detection."""
        if not _init_yolo():
            return []
        
        print("[Vision V2] Running YOLO icon detection...")
        results = _yolo_model.predict(source=image_path, conf=0.05, verbose=False, device='cpu')
        
        elements = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                elements.append({
                    "index": i,
                    "type": "icon",
                    "bbox": xyxy,
                    "confidence": conf
                })
        
        print(f"[Vision V2] ✅ YOLO detected {len(elements)} UI elements")
        return elements

    def analyze_ui(self, query: str = None, image_path: str = None) -> Dict:
        """
        Full vision pipeline:
        1. Use provided image_path or take screenshot
        2. Run EasyOCR for text
        3. Run YOLO for icons
        """
        path = image_path if image_path else self.capture_screen()
        
        result = {
            "status": "success",
            "image_path": path,
            "elements": [],
            "text_detected": [],
            "summary": "",
            "query": query or "Analyze screen"
        }
        
        try:
            # Step 1: OCR
            texts, bboxes = self._run_ocr(path)
            result["text_detected"] = texts
            
            # Step 2: YOLO
            yolo_elements = self._run_yolo(path)
            
            # Combine elements for the brain to reason over
            combined_elements = []
            
            # Text Elements
            for i, (text, bbox) in enumerate(zip(texts, bboxes)):
                combined_elements.append({
                    "id": i,
                    "type": "text",
                    "content": text,
                    "bbox": bbox
                })
                
            # Icon Elements
            for elem in yolo_elements:
                combined_elements.append({
                    "id": len(combined_elements),
                    "type": "icon",
                    "content": "UI Element (Icon/Button)",
                    "bbox": elem["bbox"]
                })
            
            result["elements"] = combined_elements
            
            # Create a textual summary for the Brain (Gemma 1B)
            text_context = ", ".join(texts[:30]) if texts else "No text detected."
            result["summary"] = f"Detected {len(texts)} text items and {len(yolo_elements)} UI icons on screen. Main text includes: {text_context}"
            
            print(f"[Vision V2] ✅ Full analysis complete: {len(combined_elements)} total symbols found.")
            
            # Auto-index in Memory
            try:
                from memory_engine_v2 import memory_v2
                memory_v2.add_asset(
                    type="screenshot",
                    path=os.path.abspath(path),
                    description=f"UI Analysis: {query or 'General Scan'}"
                )
            except:
                pass
                
        except Exception as e:
            print(f"[Vision V2] Analysis failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        
        return result

    def find_element(self, description: str) -> Optional[Dict]:
        """Finds a UI element by description."""
        analysis = self.analyze_ui(query=description)
        
        if not analysis.get("elements"):
            return None
        
        description_lower = description.lower()
        for element in analysis["elements"]:
            if description_lower in str(element.get("content", "")).lower():
                return element
        
        return None

    def click_element(self, description: str) -> bool:
        """Finds an element by description and clicks it."""
        element = self.find_element(description)
        if not element or not element.get("bbox"):
            print(f"[Vision V2] Element not found: {description}")
            return False
        
        bbox = element["bbox"]
        if isinstance(bbox[0], list):
            # EasyOCR format [[x1,y1], ...]
            center_x = int((bbox[0][0] + bbox[2][0]) / 2)
            center_y = int((bbox[0][1] + bbox[2][1]) / 2)
        else:
            # YOLO format [x1, y1, x2, y2]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
        
        print(f"[Vision V2] Clicking element '{description}' at ({center_x}, {center_y})")
        # Ensure focus before click
        import pyautogui
        pyautogui.click(center_x, center_y)
        return True

    def get_screen_text(self) -> List[str]:
        """Returns all text detected on screen."""
        analysis = self.analyze_ui()
        return analysis.get("text_detected", [])


# Singleton instance
vision_v2 = JarvisVision()
