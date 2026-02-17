import json
import numpy as np

def generate_mars_helicopter():
    blueprint = {
        "project": "MARS INGENUITY - HYPER-FIDELITY ASSEMBLY",
        "is_direct_blueprint": True,
        "parts": []
    }

    # --- 1. MAIN FUSELAGE (ROOT) ---
    chassis = {
        "name": "Fuselage_Chassis",
        "type": "box",
        "dims": [0.14, 0.14, 0.14],
        "pos": [0, 0, 0.1],
        "material": "gold_foil",
        "subsystem": "structures",
        "sub_parts": []
    }

    # --- 2. ROTOR MAST (Child of Chassis) ---
    mast = {
        "name": "Rotor_Mast",
        "type": "cylinder",
        "dims": [0.018, 0.45],
        "pos": [0, 0, 0.25], 
        "material": "carbon_fiber",
        "sub_parts": []
    }

    # Rotor Hubs (Children of Mast)
    for level, z_hub_rel in [("Lower", -0.07), ("Upper", 0.13)]:
        hub = {
            "name": f"{level}_Hub",
            "type": "cylinder",
            "dims": [0.045, 0.04],
            "pos": [0, 0, z_hub_rel],
            "material": "titanium_grade_5",
            "sub_parts": []
        }
        # Blades (Horizontal Wings)
        for b in range(2):
            angle_deg = b * 180 + (45 if level == "Upper" else 0)
            angle_rad = np.radians(angle_deg)
            hub["sub_parts"].append({
                "name": f"{level}_Blade_{b}",
                "type": "wing",
                "dims": [1.2, 0.12, 0.01],
                "pos": [0.6 * np.cos(angle_rad), 0.6 * np.sin(angle_rad), 0],
                "rot": [0, 0, angle_deg],
                "material": "carbon_fiber",
                "sub_parts": []
            })
            # Blade Ribs
            for r in range(15):
                r_pos_x = -0.55 + (r * 0.078)
                hub["sub_parts"][-1]["sub_parts"].append({
                    "name": f"{level}_Blade_{b}_Rib_{r}",
                    "type": "rib",
                    "dims": [0.001, 0.1, 0.006],
                    "pos": [r_pos_x, 0, 0],
                    "material": "aluminum_6061"
                })
        mast["sub_parts"].append(hub)

    # Solar Array
    solar_base = {
        "name": "Solar_Array_Frame",
        "type": "panel",
        "dims": [0.4, 0.4, 0.008],
        "pos": [0, 0, 0.23],
        "material": "carbon_fiber",
        "sub_parts": []
    }
    for x in range(10):
        for y in range(10):
            solar_base["sub_parts"].append({
                "name": f"Cell_{x}_{y}",
                "type": "panel",
                "dims": [0.038, 0.038, 0.001],
                "pos": [-0.18 + x*0.04, -0.18 + y*0.04, 0.005],
                "material": "solar_blue"
            })
    mast["sub_parts"].append(solar_base)
    chassis["sub_parts"].append(mast)

    # --- 3. LANDING GEAR (PRECISE ANCHORING) ---
    L_leg = 0.45
    theta_deg = 135 # Tilts "Down and Out" relative to Up-Vector
    theta_rad = np.radians(theta_deg)
    
    # Attachment point T at chassis bottom corners
    leg_coords = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    z_angles = [45, 135, 315, 225]
    
    for i, (lx, ly) in enumerate(leg_coords):
        # Anchor point T inside the lower corner of the gold foil chassis
        T = np.array([lx * 0.065, ly * 0.065, -0.06])
        
        # Axis A of the leg in the TR instance (at rot_z=0)
        # A = [sin(theta), 0, cos(theta)]
        A_local = np.array([np.sin(theta_rad), 0, np.cos(theta_rad)])
        
        # Center of cylinder C = T + A*(L/2)
        # BUT wait, the rotation rot_z is applied to the WHOLE cylinder.
        # So it's easier to calculate C in world space if there was no rotation, then rotate C?
        # No, let's keep it simple: C relative to Chassis center.
        
        # Vector from anchor to center in "unrotated-Yaw" frame
        V_to_C = A_local * (L_leg / 2)
        
        # Full relative pos matrix (Rotate V_to_C by rot_z)
        phi = np.radians(z_angles[i])
        R_z = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi),  np.cos(phi), 0],
            [0, 0, 1]
        ])
        pos_rel = T + R_z @ V_to_C
        
        leg_strut = {
            "name": f"Leg_Strut_{i}",
            "type": "cylinder",
            "dims": [0.015, L_leg],
            "pos": pos_rel.tolist(),
            "rot": [0, theta_deg, z_angles[i]],
            "material": "carbon_fiber",
            "sub_parts": [
                {
                    "name": f"Leg_Foot_{i}",
                    "type": "panel",
                    "dims": [0.07, 0.07, 0.005],
                    "pos": [0, 0, L_leg/2], # Foot is at the +Z end (the bottom end after 135 deg tilt)
                    "material": "matte_black"
                }
            ]
        }
        chassis["sub_parts"].append(leg_strut)

    # --- 4. HULL DETAILS (Fasteners) ---
    for side in range(4):
        for b in range(25):
            bx = -0.06 + (b % 5) * 0.03
            bz = -0.06 + (b // 5) * 0.03
            if side == 0: pos, rot = [0.071, bx, bz], [0, 90, 0]
            if side == 1: pos, rot = [-0.071, bx, bz], [0, -90, 0]
            if side == 2: pos, rot = [bx, 0.071, bz], [90, 0, 0]
            if side == 3: pos, rot = [bx, -0.071, bz], [-90, 0, 0]
            chassis["sub_parts"].append({
                "name": f"Bolt_{side}_{b}",
                "type": "bolt",
                "dims": [0.005, 0.012],
                "pos": pos,
                "rot": rot,
                "material": "steel_chrome"
            })

    blueprint["parts"].append(chassis)

    with open("mars_helicopter_500.json", "w") as f:
        json.dump(blueprint, f, indent=2)

if __name__ == "__main__":
    generate_mars_helicopter()
    print("Successfully generated Structured Mars Helicopter blueprint.")
