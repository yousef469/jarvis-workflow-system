import os

SERVER_PATH = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"

NEW_ENDPOINT = """
# =============================================================================
# VISION TO IMAGE GENERATION
# =============================================================================

@app.post("/api/vision/generate-image")
async def api_vision_generate_image(request: VisionRequest):
    \"\"\"
    Vision -> Prompt -> Image Generation Pipeline
    1. Vision analyzes image (VL)
    2. Suggests prompt (VL)
    3. Generates Image (SD)
    \"\"\"
    print(f"[API] Vision-to-Image: {request.prompt}")
    
    if not IMAGE_GEN_ENABLED:
        return {"success": False, "error": "Image Generation module disabled"}

    try:
        # 1. Decode Image
        if "base64," in request.image:
            img_data = base64.b64decode(request.image.split("base64,")[1])
        else:
            img_data = base64.b64decode(request.image)
            
        temp_path = "temp_gen_source.png"
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
"""

with open(SERVER_PATH, "r", encoding="utf-8") as f:
    content = f.read()

if "api_vision_generate_image" in content:
    print("Endpoint already exists.")
    exit()

# Append to end of file
with open(SERVER_PATH, "a", encoding="utf-8") as f:
    f.write("\n" + NEW_ENDPOINT + "\n")
    print("Injected api_vision_generate_image.")
