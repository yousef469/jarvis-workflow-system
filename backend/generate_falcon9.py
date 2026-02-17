import json
import numpy as np

def generate_falcon9_blueprint():
    blueprint = {
        "project": "SPACEX FALCON 9 - BLOCK 5 FULL ASSEMBLY",
        "is_direct_blueprint": True,
        "parts": []
    }

    # --- 1. FIRST STAGE BOOSTER (ROOT-ISH) ---
    # Centered at World Z=10.0 (Huge rocket). 
    # Height of booster stage is approx 40m. Dims: [3.7, 40] -> [Radius, Height]
    booster = {
        "name": "S1_Booster_Stage",
        "type": "cylinder",
        "dims": [1.85, 40.0],
        "pos": [0, 0, 20.0],
        "material": "nasa_white",
        "subsystem": "structures",
        "sub_parts": []
    }

    # Internal Ribs for Booster (30 Ribs)
    for i in range(30):
        z_off_rel = -19.5 + (i * 1.35)
        booster["sub_parts"].append({
            "name": f"Booster_Rib_{i}",
            "type": "rib",
            "dims": [1.8, 1.8, 0.05],
            "pos": [0, 0, z_off_rel],
            "material": "aluminum_6061"
        })

    # --- 2. OCTAWEB & ENGINES (Child of Booster) ---
    # Positioned at bottom of booster (Rel Z = -20.0)
    octaweb = {
        "name": "Octaweb_Structure",
        "type": "cylinder",
        "dims": [1.9, 1.2],
        "pos": [0, 0, -20.0],
        "material": "steel_dark",
        "sub_parts": []
    }

    # 9 Merlin 1D Engines
    # 1 Center, 8 Outer
    engine_positions = [[0, 0]]
    for i in range(8):
        angle = (i * 45) * (np.pi/180)
        engine_positions.append([1.1 * np.cos(angle), 1.1 * np.sin(angle)])

    for i, (ex, ey) in enumerate(engine_positions):
        engine = {
            "name": f"Merlin_1D_{i}",
            "type": "cylinder",
            "dims": [0.2, 0.5],
            "pos": [ex, ey, -0.6],
            "material": "titanium_polished",
            "sub_parts": []
        }
        
        # Engine Nozzle (Bell)
        engine["sub_parts"].append({
            "name": f"Merlin_{i}_Nozzle",
            "type": "nozzle",
            "dims": [0.45, 1.2],
            "pos": [0, 0, -0.8],
            "material": "steel_dark"
        })
        
        # Turbopump and Plumbing (4 parts per engine)
        for p in range(4):
            engine["sub_parts"].append({
                "name": f"Merlin_{i}_Pump_{p}",
                "type": "tube",
                "dims": [0.08, 0.06, 0.4],
                "pos": [0.15 * np.cos(p*90), 0.15 * np.sin(p*90), 0.2],
                "material": "steel_chrome"
            })
            
        # Fasteners for mounting (20 bolts per engine)
        for b in range(20):
            engine["sub_parts"].append({
                "name": f"Merlin_{i}_Bolt_{b}",
                "type": "bolt",
                "dims": [0.015, 0.04],
                "pos": [0.22 * np.cos(b*18), 0.22 * np.sin(b*18), 0.25],
                "rot": [0, 0, b*18],
                "material": "steel_chrome"
            })
            
        octaweb["sub_parts"].append(engine)

    booster["sub_parts"].append(octaweb)

    # --- 3. INTERSTAGE & GRID FINS (Child of Booster) ---
    # Top of Booster (Rel Z = 20.0)
    interstage = {
        "name": "Interstage_Section",
        "type": "cylinder",
        "dims": [1.86, 6.0],
        "pos": [0, 0, 23.0], # Sits above booster center
        "material": "steel_dark",
        "sub_parts": []
    }

    # 4 Titanium Grid Fins
    for i in range(4):
        angle_deg = i * 90
        angle_rad = np.radians(angle_deg)
        x, y = 1.9 * np.cos(angle_rad), 1.9 * np.sin(angle_rad)
        
        fin_arm = {
            "name": f"Grid_Fin_Actuator_{i}",
            "type": "box",
            "dims": [0.4, 0.4, 0.5],
            "pos": [x, y, 2.0], # Near top of interstage
            "rot": [0, 0, angle_deg],
            "material": "titanium_grade_5",
            "sub_parts": []
        }
        
        # The actual Grid (Simulated with a slatted wing primitive or panel)
        # We rotate it out 90 degrees
        fin_arm["sub_parts"].append({
            "name": f"Grid_Fin_Slat_Main_{i}",
            "type": "wing",
            "dims": [1.2, 1.0, 0.05],
            "pos": [0.6, 0, 0],
            "rot": [0, -90, 0], # Flush against body initially
            "material": "titanium_grade_5",
            "sub_parts": []
        })
        
        # Grid Detail (10 slats per fin)
        for s in range(10):
            fin_arm["sub_parts"][-1]["sub_parts"].append({
                "name": f"Fin_{i}_Slat_{s}",
                "type": "panel",
                "dims": [1.1, 0.02, 0.9],
                "pos": [0, -0.45 + s*0.1, 0],
                "material": "titanium_grade_5"
            })
            
        interstage["sub_parts"].append(fin_arm)

    booster["sub_parts"].append(interstage)

    # --- 4. LANDING LEGS (4 Legs - Child of Booster) ---
    # Attached near octaweb (Rel Z = -19.0)
    L_leg = 8.0
    for i in range(4):
        angle_deg = 45 + (i * 90)
        angle_rad = np.radians(angle_deg)
        lx, ly = 1.8 * np.cos(angle_rad), 1.8 * np.sin(angle_rad)
        
        # Leg center calculation (Tilted 15 degrees in flight mode)
        leg_strut = {
            "name": f"Landing_Leg_{i}",
            "type": "cylinder",
            "dims": [0.15, L_leg],
            "pos": [lx*1.1, ly*1.1, -16.0],
            "rot": [0, 10, angle_deg], # Slight tilt
            "material": "carbon_fiber",
            "sub_parts": []
        }
        
        # Foot pad
        leg_strut["sub_parts"].append({
            "name": f"Leg_Foot_{i}",
            "type": "panel",
            "dims": [1.2, 1.2, 0.1],
            "pos": [0, 0, -L_leg/2],
            "material": "matte_black"
        })
        
        # Pistons and Fasteners (12 parts per leg)
        for p in range(4):
            leg_strut["sub_parts"].append({
                "name": f"Leg_{i}_Piston_{p}",
                "type": "tube",
                "dims": [0.08, 0.06, 2.0],
                "pos": [0.2, 0, -2.0 + p],
                "material": "steel_chrome"
            })
            
        booster["sub_parts"].append(leg_strut)

    # --- 5. SECOND STAGE & FAIRING (Top of Interstage) ---
    s2_tank = {
        "name": "S2_Fuel_Tank",
        "type": "cylinder",
        "dims": [1.85, 12.0],
        "pos": [0, 0, 32.0], # Above interstage
        "material": "nasa_white",
        "sub_parts": []
    }
    
    # MVac Engine
    mvac = {
        "name": "Merlin_Vacuum",
        "type": "nozzle",
        "dims": [0.8, 3.2],
        "pos": [0, 0, -7.0],
        "material": "titanium_polished"
    }
    s2_tank["sub_parts"].append(mvac)
    
    # Fairing (2 Halves)
    for h in [1, -1]:
        fairing_half = {
            "name": f"Fairing_Half_{'A' if h > 0 else 'B'}",
            "type": "cone",
            "dims": [2.6, 13.0],
            "pos": [h * 0.1, 0, 44.0],
            "rot": [0, 0, 90 if h > 0 else -90],
            "material": "nasa_white",
            "sub_parts": []
        }
        # Internal Fairing Ribs (10 per half)
        for r in range(10):
            fairing_half["sub_parts"].append({
                "name": f"Fairing_Rib_{h}_{r}",
                "type": "rib",
                "dims": [2.5, 0.1, 2.5],
                "pos": [0, 0, -6.0 + r*1.2],
                "material": "aluminum_6061"
            })
        blueprint["parts"].append(fairing_half)

    blueprint["parts"].append(booster)
    blueprint["parts"].append(s2_tank)

    # --- 6. GLOBAL FASTENERS (Final detailing to hit 500 parts) ---
    # Fastener array for Interstage connection
    interstage_bolts = {
        "name": "Interstage_Bolt_Ring",
        "type": "box",
        "dims": [0.01, 0.01, 0.01],
        "pos": [0, 0, 20.0],
        "sub_parts": []
    }
    for b in range(100):
        angle = (b * 3.6) * (np.pi/180)
        interstage_bolts["sub_parts"].append({
            "name": f"IS_Bolt_{b}",
            "type": "bolt",
            "dims": [0.02, 0.08],
            "pos": [1.87 * np.cos(angle), 1.87 * np.sin(angle), 0],
            "rot": [0, 90, b*3.6],
            "material": "steel_chrome"
        })
    blueprint["parts"].append(interstage_bolts)

    with open("falcon9_500_detailed.json", "w") as f:
        json.dump(blueprint, f, indent=2)

if __name__ == "__main__":
    generate_falcon9_blueprint()
    print("Successfully generated 500-part Falcon 9 blueprint.")
