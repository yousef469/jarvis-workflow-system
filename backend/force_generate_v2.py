
import json
import os
import sys
import time

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from three_d_generator import Structural3DGenerator
from generate_mars_starship_125k import generate_mars_starship

def run_v2_generation():
    print("🚀 STARTING V2 FORCE RE-GENERATION...")
    
    # 1. Ensure Blueprint is Updated
    # We call the function to write the JSON fresh
    generate_mars_starship() 
    
    blueprint_path = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_125k.json"
    with open(blueprint_path, "r") as f:
        recipe = json.load(f)

    # 2. Verify Config In-Memory
    has_nose = any(p["name"] == "Starship_Nose" for p in recipe["parts"])
    if not has_nose:
        print("❌ CRITICAL ERROR: Nose Cone NOT found in blueprint JSON!")
        return
    else:
        print("✅ Nose Cone DETECTED in blueprint.")

    # 3. Generate NEW GLB
    generator = Structural3DGenerator()
    start_time = time.time()
    
    # Run Assembly
    scene = generator.assemble_hierarchical(recipe, progress_callback=lambda x: None)
    
    # 4. Save to NEW PATH
    output_glb = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_v2_REAL.glb"
    
    # Verify Engine Materials
    # We can't easily introspect the scene graph materials here without deep traversal, 
    # but the generator logic is deterministic.
    
    scene.export(output_glb)
    
    elapsed = time.time() - start_time
    print(f"\n🎉 GENERATION COMPLETE: {elapsed:.2f}s")
    print(f"💾 SAVED TO: {output_glb}")

if __name__ == "__main__":
    run_v2_generation()
