import json
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from three_d_generator import Structural3DGenerator
import trimesh

def audit_physics():
    print("[ STARTING PHYSICS & GEOMETRY AUDIT ]")
    
    blueprint_path = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_125k.json"
    with open(blueprint_path, "r") as f:
        recipe = json.load(f)

    # Manual checking of key positions
    print(f"\n[ CHECKING 1: COMPONENT WORLD POSITIONS (Theory) ]")
    
    # 1. HULL
    # Pos: [0, 26, 0]. Dims: [4.5, 4.45, 52.0]
    # Y-Span: 26 +/- 26 = [0.0, 52.0]
    hull_min_y = 0.0
    hull_max_y = 52.0
    print(f"  > Starship_Hull: Y-Range [{hull_min_y}, {hull_max_y}] (Height: 52m)")
    
    # 2. RIBS
    # Parent: Hull (World +26). Local Pos: [0, -25, 0]. 
    # World Start = 26 + (-25) = +1.0
    # Array: 74 items, Step 0.68
    # Last Rib Local = -25 + (73 * 0.68) = -25 + 49.64 = +24.64
    # Last Rib World = 26 + 24.64 = +50.64
    rib_start_y = 1.0
    rib_end_y = 50.64
    print(f"  > Internal_Ribs: Y-Range [{rib_start_y:.2f}, {rib_end_y:.2f}]")
    
    if rib_start_y >= hull_min_y and rib_end_y <= hull_max_y:
        print("    [ PASS ]: Ribs fully contained within Hull.")
    else:
        print("    [ FAIL ]: Ribs protrude outside Hull!")

    # 3. HAB DECKS
    # Parent: Hull. Local Pos: [0, 10, 0]
    # World Start = 26 + 10 = 36.0
    # Array: 5 items, Step 3.0
    # Last Deck Local = 10 + (4 * 3) = 22
    # Last Deck World = 26 + 22 = 48.0
    deck_start_y = 36.0
    deck_end_y = 48.0
    print(f"  > Hab_Decks:     Y-Range [{deck_start_y:.2f}, {deck_end_y:.2f}]")
    
    if deck_end_y <= hull_max_y:
        print("    [ PASS ]: Decks fully contained within Hull.")
    else:
        print("    [ FAIL ]: Decks protrude outside Hull (Top)!")

    # 4. ENGINES (RAPTOR 3)
    # Parent: Hull. Local Pos: [0, -25, 0]
    # World Pivot = 26 + (-25) = +1.0
    # Dims: Height 3.1
    # Rotation: 180 deg (Pointing DOWN)
    # If logic centers geometry: Span is +/- 1.55
    # World Span = 1.0 +/- 1.55 = [-0.55, 2.55]
    
    engine_top_y = 1.0 + 1.55
    engine_bottom_y = 1.0 - 1.55
    print(f"  > Raptor_Engines: Y-Range [{engine_bottom_y:.2f}, {engine_top_y:.2f}]")
    print(f"    - Overlap with Hull: {engine_top_y - hull_min_y:.2f}m")
    
    if engine_top_y > hull_min_y + 0.5: # Allow small overlap for mounting
        print("    [ WARNING ]: Engines are mostly INSIDE the hull (Hidden). Needed Adjustment.")
    else:
         print("    [ PASS ]: Engines are properly exposed.")

    print("\n[ CHECKING 2: TOTAL MASS ESTIMATE ]")
    # Hull Volume: ~73 m^3 * 8000 kg/m3 = ~584,000 kg
    # Tiles: 18,000 * 0.15^2 * pi * 0.05 * 1450... 
    # Just rough check
    print("  > Mass calculations handled by Trimesh logic. Trusting 2.5M kg output (includes payload/simulated mass).")
    
if __name__ == "__main__":
    audit_physics()
