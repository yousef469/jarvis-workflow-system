import json
import numpy as np

def generate_starship_100k():
    blueprint = {
        "project": "SPACEX STARSHIP - 100,000 PART MASSIVE ARRAY",
        "is_direct_blueprint": True,
        "parts": []
    }

    # --- 1. MAIN HULL ---
    hull_height = 50.0
    hull_radius = 4.5
    ship_hull = {
        "name": "Starship_Main_Hull",
        "type": "tube",
        "dims": [hull_radius, hull_radius - 0.05, hull_height],
        "pos": [0, 0, hull_height / 2],
        "material": "stainless_steel_304",
        "sub_parts": []
    }

    # --- 2. THERMAL SHIELD (33,000 TILE ARRAY) ---
    # We use 1 part with a massive radial array repeated over rows
    # 110 rows * 300 tiles = 33,000 tiles
    for r in range(110):
        ship_hull["sub_parts"].append({
            "name": f"HeatShield_Band_{r}",
            "type": "panel",
            "dims": [0.08, 0.08, 0.01],
            "pos": [0, 0, (r * 0.1) + 5.0 - (hull_height / 2)],
            "radial_array": {
                "count": 300,
                "radius": hull_radius + 0.02,
                "axis": "z"
            },
            "subsystem": "thermal"
        })

    # --- 3. INTERNAL FASTENERS (50,000 BOLT ARRAY) ---
    # Using 5 master panels, each with 10,000 bolts
    for p in range(5):
        ship_hull["sub_parts"].append({
            "name": f"Internal_Structural_Matrix_{p}",
            "type": "panel",
            "dims": [0.1, 0.1, hull_height - 10],
            "pos": [0, 0, 0],
            "rot": [0, 0, p * 72],
            "sub_parts": [
                {
                    "name": f"Fastener_Array_{p}",
                    "type": "bolt",
                    "dims": [0.005, 0.01],
                    "pos": [hull_radius - 0.1, 0, 0],
                    "grid_array": {
                        "count_x": 10,
                        "count_y": 1000,
                        "step_x": [0, 0.1, 0],
                        "step_y": [0, 0, 0.04]
                    },
                    "material": "steel_chrome"
                }
            ]
        })

    # --- 4. ENGINE PLUMBING (33,000 TUBES) ---
    # 33 Raptors * 1000 plumbing lines
    for e in range(33):
        angle = np.radians(e * (360/33))
        r_inner = 1.2 if e < 3 else 3.8
        ex, ey = r_inner * np.cos(angle), r_inner * np.sin(angle)
        
        ship_hull["sub_parts"].append({
            "name": f"Raptor_{e}",
            "type": "nozzle",
            "dims": [1.5 if e >= 3 else 0.8, 3.0],
            "pos": [ex, ey, -1.8],
            "sub_parts": [
                {
                    "name": f"Internal_Fuel_Lines_{e}",
                    "type": "tube",
                    "dims": [0.01, 0.008, 0.6],
                    "pos": [0, 0, 0],
                    "radial_array": {
                        "count": 1000,
                        "radius": 0.4,
                        "axis": "z"
                    },
                    "material": "steel_dark"
                }
            ]
        })

    # Total: 33k (tiles) + 50k (fasteners) + 33k (engines) = 116,000 parts.
    blueprint["parts"].append(ship_hull)

    with open("starship_100k_detailed.json", "w") as f:
        json.dump(blueprint, f, indent=2)

if __name__ == "__main__":
    generate_starship_100k()
    print("Successfully generated 116,000-part Starship blueprint.")
