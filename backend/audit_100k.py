import json
import time
import os
import sys
import psutil

# Add the backend dir to sys.path to import our generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from three_d_generator import Structural3DGenerator

def lightweight_flatten(data):
    flat_parts = []
    is_direct = data.get("is_direct_blueprint", True)
    
    def walk(node, parent_name="world"):
        p_type = node.get("type", "box")
        dims = node.get("dims", [0.1, 0.1, 0.1])
        pos = node.get("pos", [0, 0, 0])
        rot = node.get("rot", [0, 0, 0])
        
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
        
        for ak in ["radial_array", "linear_array", "grid_array"]:
            if ak in node:
                part[ak] = node[ak]
        
        flat_parts.append(part)
        for sub in node.get("sub_parts", []):
            walk(sub, parent_name=part["name"])
            
    for p in data.get("parts", []):
        walk(p)
        
    return {"project": data.get("project", "Test"), "parts": flat_parts, "is_direct_blueprint": is_direct}

def audit_100k():
    print("\n" + "="*50)
    print("🚀 JARVIS 100K PERFORMANCE AUDIT")
    print("="*50)
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    blueprint_path = "starship_100k_detailed.json"
    with open(blueprint_path, "r") as f:
        blueprint = json.load(f)
        
    flat_data = lightweight_flatten(blueprint)
    
    generator = Structural3DGenerator(output_dir="test_output")
    
    print(f"[Audit] Starting Generation of 116,000 Nodes...")
    start_time = time.time()
    
    # We run the actual assembly logic
    scene = generator.assemble_hierarchical(flat_data)
    
    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    gen_time = end_time - start_time
    mem_used = end_mem - start_mem
    node_count = len(scene.graph.nodes)
    
    print("\n--- RESULTS ---")
    print(f"Total Nodes: {node_count:,}")
    print(f"Generation Time: {gen_time:.2f} seconds")
    print(f"RAM Consumed: {mem_used:.2f} MB")
    print(f"Engine Efficiency: {node_count / gen_time:.0f} parts/sec")
    print(f"Memory Efficiency: { (mem_used * 1024) / node_count:.2f} KB/node")
    print("----------------\n")

if __name__ == "__main__":
    audit_100k()
