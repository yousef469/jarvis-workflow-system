"""Vision Worker V24 - EasyOCR + YOLO (No Instructor)"""

import asyncio
from typing import Dict, Any


class VisionWorker:
    """Handles screen analysis using EasyOCR + YOLO directly"""
    
    def __init__(self):
        self.name = "vision"
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute vision analysis task.
        Returns raw OCR + YOLO data for the brain to interpret.
        """
        print(f"[VisionWorker] 👁️ Analyzing screen for: {query}")
        
        try:
            from vision_engine_v2 import vision_v2
            
            # Perform visual analysis in thread
            result = await asyncio.to_thread(vision_v2.analyze_ui, query)
            
            # Extract text detected by OCR
            text_detected = result.get("text_detected", [])
            elements = result.get("elements", [])
            
            # Count element types
            text_elements = len([e for e in elements if e.get("type") == "text"])
            icon_elements = len([e for e in elements if e.get("type") == "icon"])
            
            # Build a structured summary
            if text_detected:
                # Get first 20 text items for summary
                text_preview = ", ".join(text_detected[:20])
                if len(text_detected) > 20:
                    text_preview += f"... (+{len(text_detected) - 20} more)"
            else:
                text_preview = "No text detected"
            
            return {
                "source": "vision",
                "query": query,
                "text_on_screen": text_detected[:30],  # Limit for response size
                "text_count": len(text_detected),
                "icon_count": icon_elements,
                "total_elements": len(elements),
                "summary": f"Found {len(text_detected)} text items and {icon_elements} UI elements. Text: {text_preview[:200]}",
                "image_path": result.get("image_path", ""),
                "status": "success"
            }
            
        except Exception as e:
            print(f"[VisionWorker] ❌ Vision Engine error: {e}")
            return {
                "source": "vision",
                "error": str(e),
                "summary": f"Vision analysis failed: {str(e)[:100]}",
                "status": "error"
            }
    
    async def take_screenshot(self) -> Dict[str, Any]:
        """Just take a screenshot without analysis"""
        try:
            from vision_engine_v2 import vision_v2
            path = await asyncio.to_thread(vision_v2.capture_screen)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Singleton instance  
vision_worker = VisionWorker()
