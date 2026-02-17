
import json
import os
import sys
import time

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from three_d_generator import Structural3DGenerator
from generate_mars_starship_125k import generate_mars_starship

def run_v3_generation():
    print("🚀 STARTING V3 (FINAL) FORCE RE-GENERATION...")
    
    # 1. Update Blueprint with Flaps & Nose Fix
    generate_mars_starship() 
    
    blueprint_path = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_125k.json"
    with open(blueprint_path, "r") as f:
        recipe = json.load(f)

    # 2. Verify Flaps Exist
    has_flaps = any("Flap" in p["name"] for p in recipe["parts"])
    if not has_flaps:
        print("❌ CRITICAL ERROR: Flaps NOT found in blueprint JSON!")
        return
    else:
        print("✅ Flaps DETECTED in blueprint.")

    # 3. Generate NEW GLB
    generator = Structural3DGenerator()
    start_time = time.time()
    
    # Run Assembly (Silence progress for speed)
    scene = generator.assemble_hierarchical(recipe, progress_callback=lambda x: None)
    
    # 4. Save to FINAL PATH
    output_glb = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_v3_FINAL.glb"
    
    scene.export(output_glb)
    
    elapsed = time.time() - start_time
    print(f"\n🎉 V3 GENERATION COMPLETE: {elapsed:.2f}s")
    print(f"💾 SAVED TO: {output_glb}")

if __name__ == "__main__":
    run_v3_generation()
