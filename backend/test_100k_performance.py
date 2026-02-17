import json
import time
import os
import sys

# Add the backend dir to sys.path to import our generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from three_d_generator import Structural3DGenerator
from server import flatten_hierarchy

def test_100k():
    print("--- 100,000 PART PERFORMANCE TEST ---")
    blueprint_path = "starship_100k_detailed.json"
    
    with open(blueprint_path, "r") as f:
        blueprint = json.load(f)
        
    start_time = time.time()
    print("[Test] Flattening hierarchy...")
    flat_data = flatten_hierarchy(blueprint)
    flatten_time = time.time() - start_time
    print(f"[Test] Flattening complete in {flatten_time:.2f}s")
    
    generator = Structural3DGenerator(output_dir="test_output")
    
    print("[Test] Starting 100k Generation...")
    gen_start = time.time()
    
    # We use assemble_hierarchical directly as it's the core of the 100k logic
    scene = generator.assemble_hierarchical(flat_data)
    
    gen_time = time.time() - gen_start
    print(f"[Test] Generation complete in {gen_time:.2f}s")
    
    # Analyze results
    print(f"Total nodes in scene graph: {len(scene.graph.nodes)}")
    print(f"Total mass: {scene.metadata['total_mass_kg']:.2f} kg")
    
    if len(scene.graph.nodes) >= 100000:
        print("\n[SUCCESS] Milestone Hit: 100,000+ parts generated!")
    else:
        print(f"\n[WARNING] Only {len(scene.graph.nodes)} parts found.")

if __name__ == "__main__":
    test_100k()
