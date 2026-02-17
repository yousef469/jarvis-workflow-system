"""Image Generation Worker - SDXL Lightning (DreamShaper XL)"""

import asyncio
from typing import Dict, Any


class ImageGenWorker:
    """Handles image generation via SDXL Lightning pipeline"""
    
    def __init__(self):
        self.name = "image_gen"
    
    async def execute(self, prompt: str) -> Dict[str, Any]:
        """
        Execute image generation task.
        Passes the prompt to the SDXL Lightning pipeline.
        """
        print(f"[ImageGenWorker] 🎨 Generating image for: {prompt}")
        
        try:
            from image_generator import process_image_request
            
            # Direct generation - no extra prompt refinement
            # The brain already crafted the prompt
            result = await asyncio.to_thread(process_image_request, prompt)
            
            if result.get("success"):
                return {
                    "source": "image_gen",
                    "prompt": prompt,
                    "image_path": result.get("path", ""),
                    "refined_prompt": result.get("prompt", prompt),
                    "summary": f"Successfully created image: {prompt[:50]}",
                    "status": "success"
                }
            else:
                return {
                    "source": "image_gen",
                    "prompt": prompt,
                    "error": result.get("error", "Unknown error"),
                    "summary": f"Image generation failed: {result.get('error', 'Unknown error')}",
                    "status": "error"
                }
                
        except Exception as e:
            print(f"[ImageGenWorker] ❌ Generation error: {e}")
            return {
                "source": "image_gen",
                "prompt": prompt,
                "error": str(e),
                "summary": f"Image generation encountered an error: {str(e)[:100]}",
                "status": "error"
            }


# Singleton instance
image_gen_worker = ImageGenWorker()
