import os

SERVER_PATH = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"

NEW_ENDPOINTS = """
# =============================================================================
# FEATURE HUB API ENDPOINTS
# =============================================================================

class VisionRequest(BaseModel):
    image: str # Base64
    prompt: str = "Describe this image"

@app.post("/api/vision/describe")
async def api_vision_describe(request: VisionRequest):
    \"\"\"Image Hub Description Endpoint\"\"\"
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
    \"\"\"Parts Analyzer Endpoint - Same logic but specialized prompt handling\"\"\"
    print(f"[API] Parts Review: {request.prompt}")
    
    # Just reuse the describe logic for now, frontend sends specific prompts
    return await api_vision_describe(request)
"""

with open(SERVER_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Check avoidance
if "api_vision_describe" in content:
    print("Endpoints already present.")
    exit(0)

# Anchor: app.mount("/models"
ANCHOR = 'app.mount("/models", StaticFiles(directory="generated_models"), name="generated_models")'

if ANCHOR in content:
    idx = content.find(ANCHOR) + len(ANCHOR)
    content = content[:idx] + "\n\n" + NEW_ENDPOINTS + "\n\n" + content[idx:]
    print("Injected new endpoints.")
    
    with open(SERVER_PATH, "w", encoding="utf-8") as f:
        f.write(content)
else:
    print(f"Could not find insertion point: {ANCHOR}")
    # Fallback to appending if anchor fails but we know it's a python file
    # But reckless appending might be outside the main block if there is one. 
    # server.py usually has implicit global scope for @app.post, so appending is actually safe-ish
    # provided it's before any __main__ block if it strictly runs from there.
