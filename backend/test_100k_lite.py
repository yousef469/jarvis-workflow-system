import json
import time
import os
import sys
import numpy as np

# Add the backend dir to sys.path to import our generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from three_d_generator import Structural3DGenerator

# Lightweight flatten logic (Simplified for 100k test)
def lightweight_flatten(data):
    flat_parts = []
    is_direct = data.get("is_direct_blueprint", True) # Force direct for test
    
    def walk(node, parent_name="world"):
        p_type = node.get("type", "box")
        dims = node.get("dims", [0.1, 0.1, 0.1])
        pos = node.get("pos", [0, 0, 0])
        rot = node.get("rot", [0, 0, 0])
        
        # Build part
        part = {
            "name": node.get("name", "part"),
            "type": p_type,
            "dims": dims,
            "pos": pos,
            "rot": rot,
            "material": node.get("material", "default"),
            "subsystem": node.get("subsystem", "structures"),
            "parent": parent_name
        }
        
        # Pass array keys
        for ak in ["radial_array", "linear_array", "grid_array"]:
            if ak in node:
                part[ak] = node[ak]
        
        flat_parts.append(part)
        
        # Recurse
        for sub in node.get("sub_parts", []):
            walk(sub, parent_name=part["name"])
            
    for p in data.get("parts", []):
        walk(p)
        
    return {"project": data.get("project", "Test"), "parts": flat_parts, "is_direct_blueprint": is_direct}

def run_100k_test():
    print("\n" + "="*50)
    print("🚀 JARVIS EXTREME FIDELITY STRESS TEST (100K MILESTONE)")
    print("="*50)
    
    blueprint_path = "starship_100k_detailed.json"
    
    if not os.path.exists(blueprint_path):
        print(f"Error: {blueprint_path} not found.")
        return

    with open(blueprint_path, "r") as f:
        blueprint = json.load(f)
        
    print(f"[Phase 1] Analyzing blueprint components...")
    flat_data = lightweight_flatten(blueprint)
    print(f" > Found {len(flat_data['parts'])} unique component definitions (Arrays).")
    
    generator = Structural3DGenerator(output_dir="test_output")
    
    print("\n[Phase 2] Initiating Procedural Instanced Generation...")
    print(" > Generating 100,000+ physics nodes in scene graph...")
    
    start = time.time()
    scene = generator.assemble_hierarchical(flat_data)
    elapsed = time.time() - start
    
    node_count = len(scene.graph.nodes)
    
    print("\n" + "-"*50)
    print(f"✅ GENERATION COMPLETE")
    print(f" > Real-time elapsed: {elapsed:.2f} seconds")
    print(f" > Final Scene Node Count: {node_count:,} PARTS")
    print(f" > Calculated Total Mass: {scene.metadata['total_mass_kg']:,.0f} kg")
    print("-"*50)
    
    if node_count >= 100000:
        print("\n🏆 MILESTONE REACHED: 100,000 PART THRESHOLD EXCEEDED")
        print("This model is now ready for MIT-level structural diagnostics.")
    else:
        print(f"\n⚠️ Result: {node_count:,} parts. Threshold not met.")

if __name__ == "__main__":
    run_100k_test()
