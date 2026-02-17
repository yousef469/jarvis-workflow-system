import json
import os

def generate_mars_starship():
    """
    Generates a 125,000+ part Mars-spec Starship blueprint.
    Adheres to the "Golden Rule": Semantic subsystems, parametric rules.
    """
    
    parts = []

    # 1. MAIN HULL (Reference Frame)
    parts.append({
        "name": "Starship_Hull",
        "type": "tube",
        "dims": [4.5, 4.45, 52.0], # Radius: 4.5m (9m Dia), Thickness: 0.05m, Height: 52m
        "pos": [0, 26.0, 0], # Centered
        "material": "stainless_steel",
        "subsystem": "structures"
    })

    # 1b. NOSE CONE (Fixing the "Eaten Top")
    parts.append({
        "name": "Starship_Nose",
        "parent": "Starship_Hull",
        "type": "cone",
        "dims": [4.5, 18.0], # Radius 4.5m, Height 18m
        "material": "stainless_steel",
        "subsystem": "structures",
        "pos": [0, 26.0 + (52.0/2) + (18.0/2), 0], # Stack on top. 26(center)+26(half_h)+9(half_cone) = 61.0
        "rot": [0, 0, 0] # Ensure upright
    })

    # 1c. FORWARD FLAPS (Aerodynamic Control)
    parts.append({
        "name": "Fwd_Flap_Left",
        "parent": "Starship_Nose",
        "type": "box", # Simplified Aerodynamic Surface
        "dims": [1.5, 5.0, 0.3], # Width 1.5, Height 5, Thin
        "material": "stainless_steel",
        "subsystem": "structures",
        "pos": [-4.5, 0, 0], # Side of nose
        "rot": [0, 0, 45] # Angled back
    })
    parts.append({
        "name": "Fwd_Flap_Right",
        "parent": "Starship_Nose",
        "type": "box",
        "dims": [1.5, 5.0, 0.3],
        "material": "stainless_steel",
        "subsystem": "structures",
        "pos": [4.5, 0, 0],
        "rot": [0, 0, -45]
    })

    # 1d. AFT FLAPS (Main Control)
    parts.append({
        "name": "Aft_Flap_Left",
        "parent": "Starship_Hull",
        "type": "box",
        "dims": [3.0, 8.0, 0.5],
        "material": "stainless_steel",
        "subsystem": "structures",
        "pos": [-4.6, -15.0, 0], # Near bottom
        "rot": [0, 0, 15]
    })
    parts.append({
        "name": "Aft_Flap_Right",
        "parent": "Starship_Hull",
        "type": "box",
        "dims": [3.0, 8.0, 0.5],
        "material": "stainless_steel",
        "subsystem": "structures",
        "pos": [4.6, -15.0, 0],
        "rot": [0, 0, -15]
    })

    # 2. THERMAL PROTECTION SYSTEM (The Heavy Lifter - ~92,100 Parts)
    # 18,000 Hexagonal Tiles aligned to hull curvature
    parts.append({
        "name": "TPS_Hex_Tiles",
        "parent": "Starship_Hull",
        "type": "hex_tile",
        "dims": [0.15, 0.05], # 15cm radius, 5cm thickness
        "material": "heat_shield_phenolic",
        "subsystem": "thermal",
        "placement": {
            "method": "surface_hex",
            "surface": "Starship_Hull",
            "coverage": "windward",
            "normal_offset": 0.012 # 12mm air gap/insulation
        },
        "instance_count": 18000
    })

    # NESTED ARRAY: 4 Mounting Pins PER Tile
    # 18,000 * 4 = 72,000 parts
    parts.append({
        "name": "TPS_Mounting_Pins",
        "parent": "TPS_Hex_Tiles",
        "type": "cylinder",
        "dims": [0.005, 0.08], # 5mm radius pin
        "material": "stainless_steel_304",
        "subsystem": "thermal",
        "radial_array": {
            "count": 4,
            "radius": 0.08,
            "per_parent_instance": True
        },
        "pos": [0, 0, -0.04] # Recessed into tile
    })

    # 3. PROPULSION (Aft Bay - ~12,000 Parts)
    # 6 Raptor 3 Engines
    # Each engine uses a De Laval Bell Nozzle curve
    parts.append({
        "name": "Raptor_3_Core",
        "parent": "Starship_Hull",
        "type": "nozzle",
        "dims": [1.3, 3.1], # Radius (Base), Height
        "material": "steel_dark",
        "subsystem": "propulsion",
        "radial_array": {
            "count": 6,
            "radius": 4.0,
            "axis": "y"
        },
        "pos": [0, -27.2, 0], # Moved down 2.2m to expose engines (Pivot at -1.2, Top +0.35)
        "rot": [0, 0, 180] # Flip engines to point down
    })

    # 6 Engines * 2000 Cooling Channels = 12,000 parts
    parts.append({
        "name": "Cooling_Channels",
        "parent": "Raptor_3_Core",
        "type": "rod",
        "dims": [0.005, 3.0],
        "material": "steel_dark", # Changed from copper to avoid "red barrel" look
        "subsystem": "propulsion",
        "channel_density": {
            "count": 2000,
            "distribution": "uniform_radial",
            "radius": 0.65
        },
        "pos": [0, 0, 0]
    })

    # 4. INTERNAL STRUCTURE (Skeleton - ~7,400 Parts)
    # 74 Rings * 100 internal stringers/ribs
    parts.append({
        "name": "Internal_Rib",
        "parent": "Starship_Hull",
        "type": "box",
        "dims": [0.05, 1.8, 0.1],
        "material": "aluminum_6061",
        "subsystem": "structures",
        "placement": {
            "frame": "tank_volume",
            "method": "axial_radial"
        },
        "linear_array": {
            "count": 74,
            "step": [0, 0.68, 0] # Adjusted step to fit 52m
        },
        "radial_array": {
            "count": 100,
            "radius": 4.45
        },
        "pos": [0, -25.0, 0] # Start at bottom of hull (-26 + 1m clearance)
    })

    # 5. MARS HABITATION (15,000 Parts)
    # 5 Decks, each with 3,000 fasteners/brackets
    parts.append({
        "name": "Hab_Deck",
        "parent": "Starship_Hull",
        "type": "cylinder",
        "dims": [8.8, 0.2],
        "material": "titanium",
        "subsystem": "gnc",
        "linear_array": {
            "count": 5,
            "step": [0, 3.0, 0]
        },
        "pos": [0, 10, 0] # Moved down to fit top deck (10 + 3*4 = 22 < 26)
    })

    parts.append({
        "name": "Hab_Fasteners",
        "parent": "Hab_Deck",
        "type": "bolt",
        "dims": [0.02, 0.02],
        "material": "steel_chrome",
        "subsystem": "gnc",
        "grid_array": {
            "count_x": 50,
            "count_y": 60,
            "step_x": [0.15, 0, 0],
            "step_y": [0, 0, 0.15]
        },
        "pos": [0, 0.1, 0]
    })

    blueprint = {
        "project": "Mars_Starship_2026_HighFidelity",
        "is_direct_blueprint": True,
        "parts": parts
    }

    output_path = "c:\\Users\\Yousef\\projects\\live ai asistant\\jarvis-system\\backend\\mars_starship_125k.json"
    with open(output_path, "w") as f:
        json.dump(blueprint, f, indent=2)

    print(f"Generated Mars Starship Blueprint: {output_path}")
    
    # Calculate Total Parts
    total = 0
    total += 1 # Hull
    total += 18000 # Tiles
    total += 18000 * 4 # Pins
    total += 6 # Engine Cores
    total += 6 * 2000 # Channels
    total += 71 * 100 # Ribs
    total += 5 # Decks
    total += 5 * 3000 # Hab Fasteners
    
    print(f"Total Theoretical Part Count: {total:,}")

if __name__ == "__main__":
    generate_mars_starship()
