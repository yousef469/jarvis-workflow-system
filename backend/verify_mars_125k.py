import time
import json
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from three_d_generator import Structural3DGenerator

def run_benchmark():
    generator = Structural3DGenerator()
    
    blueprint_path = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_125k.json"
    
    if not os.path.exists(blueprint_path):
        print(f"Error: Blueprint not found at {blueprint_path}")
        return

    with open(blueprint_path, "r") as f:
        recipe = json.load(f)

    print(f"🚀 Starting 125k Benchmark: {recipe.get('project')}")
    print(f"📦 Total Parts in Recipe: {len(recipe.get('parts', []))}")
    
    start_time = time.time()
    
    # Generate GLB
    # We use a dummy lambda for progress to avoid WebSocket overhead during pure benchmark
    scene = generator.assemble_hierarchical(recipe, progress_callback=lambda x: None)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "="*50)
    print(f"✅ BENCHMARK COMPLETE")
    print(f"⏱️  Time Elapsed: {elapsed:.2f} seconds")
    print(f"📊 Speed: {len(scene.graph.nodes) / elapsed:.2f} parts/second")
    print(f"🌳 Scene Graph Nodes: {len(scene.graph.nodes):,}")
    print("="*50)

    # Save test GLB
    output_glb = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_125k.glb"
    scene.export(output_glb)
    print(f"💾 GLB Exported: {output_glb}")

if __name__ == "__main__":
    run_benchmark()
