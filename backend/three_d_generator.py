import trimesh
import numpy as np
import json
import os
import asyncio
from pathlib import Path
import uuid

class Structural3DGenerator:
    """
    JARVIS Aerospace Structural 3D Engine.
    Converts hierarchical recipes into engineering-grade GLB models with physics metadata.
    Supports NASA/SpaceX-style subsystem organization and 6-DOF simulation readiness.
    """
    
    # Material Densities (kg/m^3) - Engineering Standard
    MATERIAL_DENSITIES = {
        "aluminum_6061": 2700.0,
        "stainless_steel_304": 8000.0,
        "titanium_grade_5": 4430.0,
        "carbon_fiber": 1600.0,
        "heat_shield_phenolic": 1450.0,
        "aerogel_space": 20.0,
        "electronics_avionic": 1200.0,
        "fuel_rp1": 810.0,
        "liquid_oxygen": 1140.0,
        "titanium_red": 4430.0,
        "titanium_gold": 4430.0,
        "steel_dark": 7800.0,
        "steel_chrome": 7800.0,
        "arc_glow": 1.0, # Visual only
        "matte_black": 1800.0,
        "default": 1000.0
    }

    MATERIAL_COLORS = {
        "titanium_red": [204, 34, 34, 255],
        "titanium_gold": [221, 170, 34, 255],
        "titanium_polished": [230, 230, 240, 255], # New Premium Material
        "steel_dark": [37, 37, 48, 255],
        "steel_chrome": [221, 221, 238, 255],
        "arc_glow": [68, 204, 255, 255],
        "matte_black": [8, 8, 8, 255],
        "carbon_fiber": [16, 16, 16, 255],
        "aluminum_6061": [190, 190, 190, 255],
        "solar_blue": [20, 40, 140, 255],
        "copper_winding": [184, 115, 51, 255],
        "nasa_white": [245, 245, 250, 255],
        "hazard_orange": [255, 120, 0, 255],
        "anodized_blue": [0, 100, 200, 255],
        "gold_foil": [212, 175, 55, 255],
        "nasa_glass": [200, 240, 255, 40], # Transparent Glass (Ultra Clear)
        "wireframe_grid": [50, 255, 50, 80], # Debug/Wireframe feel
        "default": [128, 128, 128, 255]
    }

    # PBR PROPERTIES: (Metallic, Roughness)
    # Metallic: 0.0 (Dielectric) to 1.0 (Metal)
    # Roughness: 0.0 (Mirror) to 1.0 (Matte)
    PBR_PROPERTIES = {
        "stainless_steel_304": (1.0, 0.15), # Shiny Metal
        "stainless_steel": (1.0, 0.15),
        "aluminum_6061": (0.9, 0.4), # Brushed Metal
        "titanium_grade_5": (0.9, 0.3),
        "titanium_polished": (1.0, 0.05), # Mirror
        "heat_shield_phenolic": (0.0, 0.9), # Matte Ceramic
        "steel_dark": (0.8, 0.4), # Engine Bell
        "steel_chrome": (1.0, 0.05),
        "copper": (1.0, 0.2),
        "nasa_glass": (0.0, 0.0), # Glass
        "default": (0.5, 0.5)
    }
    
    @staticmethod
    def enforce_symmetry(design_json):
        """
        Precision Engineering Guard:
        1. Grid Snapping: Forces 1mm alignment to prevent 'shaky' look.
        2. Symmetry Correction: Fixes L/R naming collisions.
        3. Vertical Alignment: Ensures rotors are UP and legs are DOWN.
        """
        if "parts" not in design_json:
            return design_json
            
        parts_map = {}
        for i, part in enumerate(design_json["parts"]):
            # SKIP SANITY CHECKS FOR DIRECT BLUEPRINTS (God-Mode Trust)
            if design_json.get("is_direct_blueprint"):
                part["pos"] = [round(p * 1000) / 1000 for p in part.get("pos", [0,0,0])]
                continue

            name = part.get("name", "").lower()
            pos = part.get("pos", [0,0,0])
            
            # 1. GRID SNAPPING (1mm precision for flush assembly)
            part["pos"] = [round(p * 1000) / 1000 for p in pos]
            
            # 2. STRUCTURAL SANITY BIAS (Assumes Z-up for Aerospace)
            if any(word in name for word in ["rotor", "blade", "mast", "solar", "antenna"]):
                if part["pos"][2] < 0: part["pos"][2] = abs(part["pos"][2])
                    
            if any(word in name for word in ["leg", "foot", "strut", "gear"]):
                if part["pos"][2] > 0: part["pos"][2] = -abs(part["pos"][2])

            # Map potential L/R pairs
            base_name = None
            side = None
            if "_l" in name or "_left" in name:
                base_name = name.replace("_l", "").replace("_left", "")
                side = "L"
            elif "_r" in name or "_right" in name:
                base_name = name.replace("_r", "").replace("_right", "")
                side = "R"
                
            if base_name:
                if base_name not in parts_map: parts_map[base_name] = {}
                parts_map[base_name][side] = i

        # 3. Symmetry Check
        for base, sides in parts_map.items():
            if "L" in sides and "R" in sides:
                idx_l, idx_r = sides["L"], sides["R"]
                part_l, part_r = design_json["parts"][idx_l], design_json["parts"][idx_r]
                pos_l, pos_r = part_l["pos"], part_r["pos"]
                
                if sum([(a-b)**2 for a, b in zip(pos_l, pos_r)]) < 0.001:
                    part_r["pos"][0] = -pos_l[0] if pos_l[0] != 0 else 0.2
                    
        return design_json

    @staticmethod
    def flatten_hierarchy(assembly_data):
        """Recursively flattens nested 'sub_parts' into a flat list for the generator."""
        flat_parts = []
        is_direct = False
        project_name = "Robotic_Assembly"
        
        if isinstance(assembly_data, dict):
            is_direct = assembly_data.get("is_direct_blueprint", False)
            project_name = assembly_data.get("project", "Robotic_Assembly")

        def walk(node, parent_pos=[0,0,0], parent_name="world"):
            p_type = node.get("type", node.get("shape", "box"))
            if isinstance(p_type, list): p_type = p_type[0]
            p_type = str(p_type).lower()
            
            dims = node.get("dims", node.get("dimensions", node.get("size", [0.1, 0.1, 0.1])))
            if isinstance(dims, dict):
                l = dims.get("length", dims.get("l", dims.get("x", 0.1)))
                w = dims.get("width", dims.get("w", dims.get("y", 0.1)))
                h = dims.get("height", dims.get("h", dims.get("z", 0.1)))
                r = dims.get("radius", dims.get("diameter", 0.1) / 2)
                if "radius" in dims or "diameter" in dims: dims = [r, l]
                else: dims = [l, w, h]
            
            positions_list = node.get("positions", [])
            if not positions_list:
                pos = node.get("pos", node.get("coordinates", node.get("position", [0, 0, 0])))
                positions_list = [pos]

            for i, local_pos in enumerate(positions_list):
                if isinstance(local_pos, dict):
                    local_pos = [local_pos.get("x", 0), local_pos.get("y", 0), local_pos.get("z", 0)]
                
                rel_pos = [parent_pos[0] + float(local_pos[0]), parent_pos[1] + float(local_pos[1]), parent_pos[2] + float(local_pos[2])]
                actual_parent = "world" if not is_direct else parent_name

                p_name = node.get("name", f"part_{len(flat_parts)}")
                if len(positions_list) > 1: p_name = f"{p_name}_{i+1}"

                part = {
                    "name": p_name, "type": p_type, "dims": dims, "pos": rel_pos, 
                    "rot": node.get("rot", [0, 0, 0]), "material": node.get("material", "default"),
                    "subsystem": node.get("subsystem", "structures"), "parent": actual_parent
                }
                
                for ak in ["radial_array", "linear_array", "grid_array"]:
                    if ak in node: part[ak] = node[ak]

                flat_parts.append(part)
                children = node.get("sub_parts", [])
                for child in children:
                    walk(child, parent_pos=rel_pos, parent_name=part["name"])
        
        if isinstance(assembly_data, list):
            for p in assembly_data: walk(p)
        elif isinstance(assembly_data, dict):
            if "parts" in assembly_data:
                for p in assembly_data["parts"]: walk(p)
            elif "assembly" in assembly_data:
                walk(assembly_data["assembly"])
            elif "project" in assembly_data:
                 if "type" in assembly_data or "shape" in assembly_data: walk(assembly_data)
                 elif "sub_parts" in assembly_data:
                     for p in assembly_data["sub_parts"]: walk(p)

        return {"project": project_name, "parts": flat_parts, "is_direct_blueprint": is_direct}

    def __init__(self, output_dir="generated_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mesh_cache = {} # Cache for instanced geometry reuse
        
    def _get_material_name(self, requested):
        """Fuzzy lookup to handle 'carbon_fiber_composite' -> 'carbon_fiber'"""
        if not requested: return "default"
        req = requested.lower().replace(" ", "_")
        
        # 1. Direct Match
        if req in self.MATERIAL_DENSITIES: return req
        
        # 2. Substring Match
        for m in self.MATERIAL_DENSITIES:
            if m in req or req in m: return m
            
        # 3. Known Aliases
        aliases = {
            "alloy": "aluminum_6061",
            "composite": "carbon_fiber",
            "ion": "electronics_avionic",
            "foam": "aerogel_space",
            "plastic": "matte_black",
            "glass": "nasa_glass"
        }
        for a, target in aliases.items():
            if a in req: return target
            
        return "default"

    # DETERMINISTIC PRIMITIVE SPECIFICATION (NASA STANDARD)
    EXPECTED_DIMS = {
        "box": 3,      # [x, y, z]
        "panel": 3,    # [x, y, thickness]
        "cylinder": 2, # [radius, height]
        "tube": 3,     # [outer_r, inner_r, height]
        "sphere": 1,   # [radius]
        "wing": 3,     # [span, chord, thickness]
        "nozzle": 2,   # [radius, length]
        "cone": 2,      # [radius, height]
        "bolt": 2,     # [radius, length] - Specialized primitive
        "rib": 3,       # [x, y, z] - Specialized primitive
        "rod": 2,       # [radius, length] - Alias for cylinder
        "hex_tile": 2   # [radius, thickness] - High-fidelity TPS tile
    }

    def validate_part(self, p, is_direct=False):
        """Rule #3: Validate BEFORE creating meshes (Deterministic Safety)"""
        p_name = p.get("name", "Unnamed Part")
        p_type = p.get("type")
        dims = p.get("dims", [])
        
        if p_type not in self.EXPECTED_DIMS:
            raise ValueError(f"CRITICAL: Invalid primitive type '{p_type}' in '{p_name}'. System only allows: {list(self.EXPECTED_DIMS.keys())}")

        # Rule #3.5: ADAPTIVE DIMENSIONS (For External AIs)
        if p_type == "tube" and len(dims) == 2:
            p["dims"] = [dims[0], dims[0]*0.8, dims[1]]
            dims = p["dims"]
        elif p_type == "wing" and len(dims) == 2:
            p["dims"] = [dims[0], dims[1], dims[1]*0.1]
            dims = p["dims"]
        elif p_type == "nozzle" and len(dims) == 3:
            p["dims"] = [dims[0], dims[2]]
            dims = p["dims"]
        elif p_type == "cone" and len(dims) == 3:
            p["dims"] = [dims[0], dims[2]]
            dims = p["dims"]

        if len(dims) != self.EXPECTED_DIMS[p_type]:
            raise ValueError(f"CRITICAL: Wrong dimensions for {p_type} in '{p_name}'. Expected {self.EXPECTED_DIMS[p_type]}, got {len(dims)}.")

        # Rule: Any dimension <= 0 or > 3 is rejected as unsafe scale
        # EXCEPTION: Direct blueprints or High-Fidelity assemblies allow "Orbital Scale" up to 100m.
        max_allowed = 100.0 if is_direct else 3.0
        if any(d <= 0 or d > max_allowed for d in dims):
            raise ValueError(f"CRITICAL: Unsafe scale detected in '{p_name}' dims: {dims}. Must be between 0.001 and {max_allowed} meters.")

    def create_primitive(self, p_type, dims, material_name="default"):
        """Rule #4: Geometry generation (DETERMINISTIC) with Caching for High-Fidelity"""
        material_name = self._get_material_name(material_name)
        
        # 0. Cache Lookup (Crucial for 100k parts)
        # We cache by rounded dims to ensure slight floating point variance doesn't break instancing
        cache_key = (p_type, tuple(np.round(dims, 5)), material_name)
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key].copy()
        
        # Adaptive sections for curves (Increased for High-Fidelity surface sampling)
        sections = 64
        
        if p_type == "box" or p_type == "panel" or p_type == "rib":
            # Panel and Rib are geometrically boxes in this deterministic system
            mesh = trimesh.creation.box(extents=dims)
            
        elif p_type == "cylinder":
            mesh = trimesh.creation.cylinder(radius=dims[0], height=dims[1], sections=sections)
            # Align with Y-axis
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            
        elif p_type == "sphere":
            mesh = trimesh.creation.icosphere(radius=dims[0], subdivisions=3)
            
        elif p_type == "tube":
            # Tube: [outer_r, inner_r, height]
            mesh = self._create_tube_mesh(dims[0], dims[1], dims[2], sections=sections)
            # Align with Y-axis
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            
        elif p_type == "wing":
            # Wing: [span, chord, thickness]
            # Mapping to our established airfoil logic for aero integrity
            mesh = self._create_airfoil_mesh(chord=dims[1], span=dims[0], thickness_m=dims[2])
            
        elif p_type == "nozzle":
            # Nozzle: [radius, length]
            # Optimized Bell curve
            mesh = self._create_nozzle_mesh(dims[0], dims[0]*0.4, dims[0]*0.8, dims[1], sections=sections)
            # Align with Y-axis (Points +Y instead of +Z)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            
        elif p_type == "cone":
            mesh = trimesh.creation.cone(radius=dims[0], height=dims[1], sections=sections)
            # 1. Center the cone (Base is at 0, Tip at H. Move to -H/2 .. +H/2)
            mesh.apply_translation([0, 0, -dims[1]/2])
            # 2. Align with Y-axis (Rotate -90 deg X to point UP)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
            
        elif p_type == "bolt":
            # Bolt Head + Shank (Standard Engineering)
            head = trimesh.creation.cylinder(radius=dims[0]*1.5, height=dims[1]*0.2, sections=sections)
            shank = trimesh.creation.cylinder(radius=dims[0], height=dims[1], sections=sections)
            head.apply_translation([0, 0, dims[1]*0.5 + dims[1]*0.1])
            mesh = trimesh.util.concatenate([head, shank])
            # Align with Y-axis
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            
        elif p_type == "rod":
            # Rod: [radius, height]
            mesh = trimesh.creation.cylinder(radius=dims[0], height=dims[1], sections=sections)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            
        elif p_type == "hex_tile":
            # Hex Tile: [radius, thickness] - A 6-sided cylinder
            mesh = trimesh.creation.cylinder(radius=dims[0], height=dims[1], sections=6)
            
        else:
            # Rule #4: If type isn't recognized → crash, don't fallback.
            raise ValueError(f"Unsupported primitive type: {p_type}")
            
        # Add Physical Properties (Engineering Metadata)
        density = self.MATERIAL_DENSITIES.get(material_name, self.MATERIAL_DENSITIES["default"])
        volume = mesh.volume
        mass = volume * density
        
        # PBR Material Assignment
        base_color = self.MATERIAL_COLORS.get(material_name, self.MATERIAL_COLORS["default"])
        metallic, roughness = self.PBR_PROPERTIES.get(material_name, self.PBR_PROPERTIES["default"])
        
        # Apply PBR only if supported (trimesh 3.9+ handles this in GLB export)
        # We manually set visual.material to PBR
        try:
            mesh.visual.material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=base_color,
                metallicFactor=metallic,
                roughnessFactor=roughness,
                alphaMode='BLEND' if base_color[3] < 255 else 'OPAQUE'
            )
        except Exception:
            # Fallback for older trimesh versions
            mesh.visual.face_colors = base_color

        mesh.metadata = {
            "mass_kg": float(mass),
            "material": material_name,
            "volume_m3": float(volume),
            "center_of_mass_local": mesh.center_mass.tolist()
        }
        
        # Store in cache
        self.mesh_cache[cache_key] = mesh.copy()
        
        return mesh
            



    def _create_nozzle_mesh(self, r_base, r_throat, r_exit, height, sections=32):
        """Generates a high-fidelity De Laval Bell Nozzle using revolution."""
        # Define 2D profile curve (y, z)
        # 0: Base, 0.3h: Throat, 1.0h: Exit
        t = np.linspace(0, 1, 20)
        
        # Simple bell curve approximation
        # throat is at 30% height
        h_throat = height * 0.3
        
        profile = []
        for i in range(20):
            z = t[i] * height
            if z < h_throat:
                # Converging section (Linearish)
                ratio = z / h_throat
                r = r_base + (r_throat - r_base) * ratio
            else:
                # Diverging section (Parabolic/Bell)
                ratio = (z - h_throat) / (height - h_throat)
                # Curve from throat to exit
                r = r_throat + (r_exit - r_throat) * (ratio ** 0.5) 
            profile.append([r, z])
            
        # Revolve profile
        return trimesh.creation.revolve(
            np.array(profile), 
            angle=2*np.pi, 
            sections=sections
        )

    def _create_tube_mesh(self, r_outer, r_inner, height, sections=64):
        """Generates a hollow tube using annulus extrusion."""
        return trimesh.creation.annulus(
            r_min=r_inner, 
            r_max=r_outer, 
            height=height,
            sections=sections
        )

    def _create_airfoil_mesh(self, chord, span, thickness_m, n_points=40):
        """
        Generates a symmetric NACA airfoil (e.g. NACA 00xx) extruded to form a wing/blade.
        Math: https://en.wikipedia.org/wiki/NACA_airfoil
        """
        # Calculate t (thickness ratio) assuming thickness_m is max thickness
        # NACA 00xx: t = thickness / chord
        t = thickness_m / chord if chord > 0 else 0.1
        if t < 0.02: t = 0.02 # Min thickness safety
        
        # Generate x points (cosine spacing for better curve at leading edge)
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2 # 0 to 1
        
        # NACA symmetric thickness distribution formula
        # yt = 5t * (0.2969sqrt(x) - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1015x^4)
        yt = 5 * t * (
            0.2969 * np.sqrt(x) - 
            0.1260 * x - 
            0.3516 * x**2 + 
            0.2843 * x**3 - 
            0.1015 * x**4
        )
        
        # Coordinates
        points_upper = np.column_stack((x * chord, yt * chord))
        points_lower = np.column_stack((x * chord, -yt * chord))
        
        from shapely.geometry import Polygon
        
        # Combine (Trailing edge -> Leading edge -> Trailing edge)
        points = np.vstack((points_upper[::-1], points_lower[1:]))
        poly = Polygon(points)
        
        # Extrude along Z (Span)
        # Shift to center
        transform = trimesh.transformations.translation_matrix([-chord/2, 0, -span/2])
        mesh = trimesh.creation.extrude_polygon(poly, height=span)
        mesh.apply_transform(transform)
        
        # Rotate to align with standard dims (Box-like orientation)
        # We rotate 90 deg around Y so that 'span' lies along the X axis (Horizontal)
        # Default extrude is Z (Vertical), we want result to be horizontal by default.
        rot_90 = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        mesh.apply_transform(rot_90)
        
        return mesh

    def extract_collision_mesh(self, mesh):
        """Generates a simplified convex hull for collision physics (NASA Module 2)"""
        try:
            return mesh.convex_hull
        except Exception:
            # Fallback to bounding box if hull fails
            return trimesh.creation.box(extents=mesh.bounding_box.extents)

    def calculate_inertia(self, mesh, mass):
        """Calculate inertia tensor for a given mass (Module 2)"""
        # Trimesh provides moment_inertia for density=1.0, scale by actual mass
        return (mesh.moment_inertia * (mass / mesh.volume)).tolist()

    def assemble_hierarchical(self, recipe, progress_callback=None):
        """
        Assembles a hierarchical model from a recipe.
        Includes engineering metadata for 6-DOF simulation.
        Subsystems: structures, propulsion, aero, gnc, thermal.
        """
        scene = trimesh.Scene()
        parts_list = recipe.get("parts", [])
        
        total_mass = 0.0
        weighted_com = np.zeros(3)
        
        subsystem_stats = {}
        
        print(f"[3DGen] Building assembly: {recipe.get('project', 'Unnamed Vehicle')}")
        print(f"[3DGen] Total expected parts: {len(parts_list)}")

        # 1. HIERARCHICAL SORTING (Ensure parents are created before children)
        sorted_parts = []
        visited = set()
        
        def add_recursive(p_name):
            if p_name in visited: return
            # Find the part config
            cfg = next((p for p in parts_list if p.get('name') == p_name), None)
            if not cfg: return
            
            # Add parent first
            parent = cfg.get('parent')
            if parent and parent != 'world':
                add_recursive(parent)
            
            if p_name not in visited:
                sorted_parts.append(cfg)
                visited.add(p_name)

        for p in parts_list:
            add_recursive(p.get('name'))
        
        parts_list = sorted_parts

        # 2. Safety: Scale Check (Detect meters vs centimeters hallucination)
        is_small_drone = any(word in recipe.get('project', '').lower() for word in ['drone', 'helicopter', 'quad', 'ingenuity', 'micro'])
        scale_factor = 1.0
        
        # If any major part is over 0.5m for a drone, it's likely a scale error
        # EXCEPTION: Direct blueprints or high-fidelity assemblies (>100 parts)
        is_direct = recipe.get("is_direct_blueprint") or len(parts_list) > 100
        if is_small_drone and not is_direct:
            for p in parts_list:
                d = p.get("dims", [1,1,1])
                if max(d) > 0.8: # Threshold: Drone body shouldn't be a 1m cylinder
                    scale_factor = 0.1
                    print(f"[3DGen] [SAFETY] Massive scale ({max(d)}m) detected for drone part. Applying 0.1x safety scale.")
                    break

        # Performance Upgrade: Matrix Caching (Fixes O(N^2) bottleneck)
        # We store the world matrix of every instance to avoid recursive graph lookups.
        # Key: instance_name, Value: 4x4 numpy matrix
        world_matrices = {"world": np.eye(4)}
        
        # Group tracker for nested instancing (e.g., 3 pins on EACH of the 18,000 tiles)
        # Key: part_name, Value: List of {"name": inst_name, "matrix": world_matrix}
        group_instances = {"world": [{"name": "world", "matrix": np.eye(4)}]}

        pos_tracker = {} # Track used coordinates to prevent overlapping
        
        for i, part_cfg in enumerate(parts_list):
            p_name = part_cfg.get("name", f"part_{i}")
            p_type = part_cfg.get("type", "box")
            
            # Rule #3: Validate BEFORE creating meshes (Deterministic Safety)
            self.validate_part(part_cfg, is_direct=is_direct)
            
            # 3. SAFETY: PART SIZE NORMALIZATION
            origin_dims = part_cfg.get("dims", [1, 1, 1])
            scaled_dims = [abs(float(d)) * scale_factor for d in origin_dims]
            if i == 0: self.root_max_dim = max(scaled_dims)
            
            is_micro = any(tag in p_name.lower() or tag in p_type.lower() for tag in ['bolt', 'screw', 'fastener', 'pin', 'nut', 'connector', 'sensor'])
            if i > 0 and not recipe.get("is_direct_blueprint"):
                if is_micro and any(d > 0.05 for d in scaled_dims):
                    scaled_dims = [min(d, 0.05) for d in scaled_dims]
                    print(f"[3DGen] [SAFETY] Micro-part Gigantism detected for {p_name}. Clamping to 0.05m.")
                elif any(d > getattr(self, 'root_max_dim', 2.0) for d in scaled_dims):
                    scaled_dims = [min(d, getattr(self, 'root_max_dim', 2.0)) for d in scaled_dims]
                    print(f"[3DGen] [SAFETY] Abnormal size detected for {p_name}. Clamping to root scale.")

            # 4. PARENT RESOLUTION (Reference Frame Logic)
            parent_id = part_cfg.get("parent", "world")
            # If the parent is "world", we have one parent instance.
            # If the parent is a previous part name (e.g. "Starship_Hull"), we fetch ALL its instances.
            parent_group = group_instances.get(parent_id, group_instances["world"])

            # 5. LOCAL ARRAY LOGIC (Parametric Placement)
            # This is the "Local" transform relative to the parent.
            local_pos = np.array(part_cfg.get("pos", [0, 0, 0])) * scale_factor
            local_rot = part_cfg.get("rot", [0, 0, 0])
            
            # Semantic Subsystem Mapping
            mat_name = part_cfg.get("material", "default").lower()
            name_lower = p_name.lower()
            if "bolt" in name_lower or "screw" in name_lower or "fastener" in name_lower: mat_name = "steel_chrome"
            elif "rib" in name_lower or "bracket" in name_lower: mat_name = "gold_foil" if "rib" in name_lower else "aluminum_6061"
            elif "blade" in name_lower or "rotor" in name_lower: mat_name = "carbon_fiber"
            elif "joint" in name_lower or "gear" in name_lower: mat_name = "steel_dark"
            elif "glass" in name_lower or "shroud" in name_lower: mat_name = "nasa_glass" # Removed 'hull' to prevent transparent starships
            elif "wire" in name_lower or "cable" in name_lower: mat_name = "hazard_orange"

            # Compute Local Instance Transforms
            # NEW: Iterative Array Processing (Multiplicative support)
            local_templates = [{"pos": local_pos, "rot": local_rot}]
            
            if "linear_array" in part_cfg:
                arr = part_cfg["linear_array"]; count = int(arr.get("count", 1)); step = np.array(arr.get("step", [0, 0, 0]))
                new_templates = []
                for temp in local_templates:
                    for a_i in range(count):
                        new_templates.append({"pos": np.array(temp["pos"]) + (step * a_i), "rot": temp["rot"]})
                local_templates = new_templates

            if "radial_array" in part_cfg:
                arr = part_cfg["radial_array"]
                count, radius, axis = int(arr.get("count", 1)), float(arr.get("radius", 0)), arr.get("axis", "z")
                start_angle = float(arr.get("start_angle", 0))
                new_templates = []
                for temp in local_templates:
                    orig_pos = np.array(temp["pos"])
                    for a_i in range(count):
                        angle_deg = start_angle + (a_i * (360.0 / count))
                        angle_rad = np.radians(angle_deg)
                        i_pos = np.copy(orig_pos)
                        if axis == "z":
                            i_pos += [radius * np.cos(angle_rad), radius * np.sin(angle_rad), 0]
                            i_rot = [temp["rot"][0], temp["rot"][1], temp["rot"][2] + angle_deg]
                        elif axis == "y":
                            i_pos += [radius * np.sin(angle_rad), 0, radius * np.cos(angle_rad)]
                            i_rot = [temp["rot"][0], temp["rot"][1] + angle_deg, temp["rot"][2]]
                        else:
                            i_pos += [0, radius * np.cos(angle_rad), radius * np.sin(angle_rad)]
                            i_rot = [temp["rot"][0] + angle_deg, temp["rot"][1], temp["rot"][2]]
                        new_templates.append({"pos": i_pos, "rot": i_rot})
                local_templates = new_templates

            if "grid_array" in part_cfg:
                arr = part_cfg["grid_array"]; cx, cy = int(arr.get("count_x", 1)), int(arr.get("count_y", 1))
                sx, sy = np.array(arr.get("step_x", [0, 0, 0])), np.array(arr.get("step_y", [0, 0, 0]))
                new_templates = []
                for temp in local_templates:
                    orig_pos = np.array(temp["pos"])
                    for ix in range(cx):
                        for iy in range(cy):
                            new_templates.append({"pos": orig_pos + (sx * ix) + (sy * iy), "rot": temp["rot"]})
                local_templates = new_templates

            if "channel_density" in part_cfg:
                dens = part_cfg["channel_density"]; d_count = int(dens.get("count", 100)); d_type = dens.get("distribution", "uniform_radial")
                if d_type == "uniform_radial":
                    radius = float(dens.get("radius", part_cfg.get("radial_array", {}).get("radius", 0.5)))
                    new_templates = []
                    for temp in local_templates:
                        orig_pos = np.array(temp["pos"])
                        for d_i in range(d_count):
                            angle_rad = (d_i / d_count) * (2 * np.pi)
                            i_pos = orig_pos + [radius * np.cos(angle_rad), radius * np.sin(angle_rad), 0]
                            new_templates.append({"pos": i_pos, "rot": [temp["rot"][0], temp["rot"][1], temp["rot"][2] + np.degrees(angle_rad)]})
                    local_templates = new_templates

            # Create shared mesh
            base_mesh = self.create_primitive(p_type, scaled_dims, mat_name)
            p_mass = base_mesh.metadata["mass_kg"]
            subsystem = part_cfg.get("subsystem", "structures")
            if subsystem not in subsystem_stats: subsystem_stats[subsystem] = {"mass": 0.0, "count": 0}

            # 6. NESTED ASSEMBLY LOOP (Surface-Aware Golden Rule Implementation)
            this_group_instances = []
            
            # --- SURFACE SAMPLING (New Rule-Based Engine) ---
            placement = part_cfg.get("placement", {})
            method = placement.get("method")
            
            if method == "surface_hex":
                # Sample the parent mesh for curvature-aligned placement
                surf_node = placement.get("surface", parent_id)
                # We need the parent MESH to sample. We'll search the scene nodes.
                try:
                    parent_node_data = scene.graph.get(parent_group[0]["name"])
                    target_geom_name = parent_node_data[1]
                    target_geom = scene.geometry.get(target_geom_name)
                    
                    if target_geom:
                        target_count = part_cfg.get("instance_count", 18000)
                        offset = float(placement.get("normal_offset", 0.0))
                        coverage = placement.get("coverage", "all")
                        
                        print(f"      - {p_name}: Sampling {target_count} points from {target_geom_name}...")
                        
                        # High-speed surface sampling
                        # Sample 20x the count to ensure dense coverage after aggressive filtering
                        samples, face_indices = trimesh.sample.sample_surface(target_geom, target_count * 20)
                        normals = target_geom.face_normals[face_indices]
                        
                        # Filter for "Windward" (Belly-side)
                        if coverage == "windward":
                            # Use more robust check: are we on the +X side of the hull?
                            # Also check that it's NOT on a cap (nx,nz should be significant compared to ny)
                            mask = (normals[:, 0] > 0.05) & (np.abs(normals[:, 1]) < 0.95)
                            samples, normals = samples[mask], normals[mask]
                            if len(samples) > target_count:
                                samples, normals = samples[:target_count], normals[:target_count]
                            print(f"      - {p_name}: Filtered to {len(samples)} windward tiles (Yield: {len(samples)/target_count:.1%})")
                        sampled_mats = []
                        for s, n in zip(samples, normals):
                            # Alig Part-Z with Surface-Normal
                            z_axis = n / np.linalg.norm(n)
                            x_axis = np.cross([0, 1, 0], z_axis)
                            if np.linalg.norm(x_axis) < 1e-6: x_axis = np.cross([1, 0, 0], z_axis)
                            x_axis /= np.linalg.norm(x_axis)
                            y_axis = np.cross(z_axis, x_axis)
                            
                            m = np.eye(4)
                            m[:3, 0] = x_axis
                            m[:3, 1] = y_axis
                            m[:3, 2] = z_axis
                            m[:3, 3] = s + (z_axis * offset)
                            sampled_mats.append(m)
                        
                        # Override local templates with sampled matrices
                        local_templates = [{"matrix": m} for m in sampled_mats]
                        
                        if not local_templates:
                            print(f"      [WARNING] {p_name}: No surface templates generated. Check parent mesh or coverage rules.")
                    else:
                        print(f"      [WARNING] {p_name}: Parent geometry '{target_geom_name}' not found in scene.")
                except Exception as e:
                    print(f"      [ERROR] {p_name} surface sampling failed: {str(e)}")
                
            if i % 10 == 0:
                print(f"  > [{i+1}/{len(parts_list)}] Constructing {p_name} ({p_type})... (Parent: {parent_id})")

            for p_idx, p_inst in enumerate(parent_group):
                p_world_matrix = p_inst["matrix"]
                
                for l_idx, l_temp in enumerate(local_templates):
                    # Sub-instance reporting (Every 5000 parts to avoid IO bottleneck)
                    total_idx = p_idx * len(local_templates) + l_idx
                    if total_idx > 0 and total_idx % 5000 == 0:
                        print(f"      - {p_name}: Instantiated {total_idx}/{len(parent_group)*len(local_templates)} parts...")
                        if progress_callback:
                            try: progress_callback(json.dumps({"message": f"Instantiating {p_name} ({total_idx})", "percentage": 10 + int((i / len(parts_list)) * 40)}))
                            except: pass

                    # Calculate Local Matrix (Either from matrix template or pos/rot template)
                    if "matrix" in l_temp:
                        i_matrix = l_temp["matrix"]
                    else:
                        i_matrix = trimesh.transformations.translation_matrix(l_temp["pos"])
                        if l_temp["rot"] != [0, 0, 0]:
                            rad_rot = [np.radians(r) for r in l_temp["rot"]]
                            i_matrix = trimesh.transformations.concatenate_matrices(i_matrix, trimesh.transformations.euler_matrix(*rad_rot))
                    
                    # 3. GLOBAL MATRIX (The Fix!)
                    world_matrix = p_world_matrix @ i_matrix
                    
                    # Unique Name
                    inst_name = f"{p_name}_{total_idx}" if len(parent_group) * len(local_templates) > 1 else p_name
                    
                    # Cache for children
                    this_group_instances.append({"name": inst_name, "matrix": world_matrix})
                    # world_matrices[inst_name] = world_matrix # Keep if needed for debug
                    
                    # Physics & CoM
                    world_pos = world_matrix[:3, 3]
                    total_mass += p_mass
                    subsystem_stats[subsystem]["mass"] += p_mass
                    subsystem_stats[subsystem]["count"] += 1
                    weighted_com += world_pos * p_mass

                    # Construct Scene Node
                    part_metadata = {
                        "name": inst_name,
                        "system": part_cfg.get("system", "default"),
                        "subsystem": subsystem,
                        "explode_dist": part_cfg.get("explode_dist", 1.0),
                        "engineering_data": base_mesh.metadata
                    }
                    
                    # NOTE: We add to scene using relative transform to parent_node for Explode Mode compatibility
                    # But we use the world_pos we COMPOTED for our internal physics metrics.
                    scene.add_geometry(
                        geometry=base_mesh, 
                        node_name=inst_name, 
                        parent_node_name=p_inst["name"],
                        transform=i_matrix, 
                        metadata=part_metadata
                    )

            # Record this group for future children (Reference Frame support)
            group_instances[p_name] = this_group_instances

        # Global Scene Metadata
        scene.metadata.update({
            "project": recipe.get("project", "SpaceVehicle"),
            "total_mass_kg": float(total_mass),
            "center_of_mass": (weighted_com / total_mass).tolist() if total_mass > 0 else [0,0,0],
            "subsystem_analysis": subsystem_stats,
            "simulation_ready": True,
            "physics_enabled": True,
            "units": "meters/kilograms"
        })
            
        return scene

    def generate_sdf(self, recipe, physics_data, filename_base):
        """
        Generates Gazebo/ROS compatible SDF file.
        Includes visual/collision links and inertia tensor.
        """
        project_name = recipe.get("project", "SpaceVehicle").replace(" ", "_")
        
        # Calculate inertia (Simplified diagonal for now)
        parts = physics_data["parts"]
        total_mass = physics_data["total_mass_kg"]
        
        # Assemble SDF XML
        sdf = [
            "<?xml version='1.0'?>",
            "<sdf version='1.7'>",
            f"  <model name='{project_name}'>",
            "    <static>false</static>",
            "    <link name='base_link'>",
            "      <inertial>",
            f"        <mass>{total_mass:.4f}</mass>",
            "        <inertia>",
            "          <ixx>1.0</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz>",
            "          <iyy>1.0</iyy> <iyz>0.0</iyz>",
            "          <izz>1.0</izz>",
            "        </inertia>",
            "      </inertial>",
            "      <collision name='collision'>",
            "        <geometry>",
            f"          <mesh><uri>model://{project_name}/meshes/{filename_base}_collision.stl</uri></mesh>",
            "        </geometry>",
            "      </collision>",
            "      <visual name='visual'>",
            "        <geometry>",
            f"          <mesh><uri>model://{project_name}/meshes/{filename_base}_visual.glb</uri></mesh>",
            "        </geometry>",
            "      </visual>",
            "    </link>",
            "    <!-- JARVIS Control Plugin Placeholder -->",
            "    <plugin name='jarvis_control' filename='libJarvisControlPlugin.so'/>",
            "  </model>",
            "</sdf>"
        ]
        
        return "\n".join(sdf)

    def generate_glb(self, recipe, filename=None, progress_callback=None, unity_path=None):
        """
        Unity Pipeline: Generates Visual GLB + Collision STL/FBX
        """
        if not filename:
            filename = f"space_{uuid.uuid4().hex[:8]}"
            
        parts = recipe.get("parts", [])
        total_parts = len(parts)
        
        # Calculate Total Steps: 
        # 1 (Init) + 1 (Build Visual) + 1 (Export Visual) + 1 (Hulls) + 1 (Export Collision) + 1 (Physics) + 1 (Unity)
        total_steps = 7
        current_step = 0

        def report(msg):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                # Send structured data: "Step X/Y: Message"
                try:
                    progress_data = json.dumps({
                        "message": msg,
                        "step": current_step,
                        "total": total_steps,
                        "percentage": int((current_step / total_steps) * 100)
                    })
                    progress_callback(progress_data)
                except Exception as e:
                    print(f"[3DGen] Progress Report Failed: {e}")

        report("Initializing Aerospace Assembly...")
        
        visual_path = self.output_dir / f"{filename}_Visual.glb"
        collision_path = self.output_dir / f"{filename}_Collision.stl"
        physics_path = self.output_dir / f"{filename}_Physics.json"
        
        physics_data = {
            "project": recipe.get("project", "SpaceVehicle"),
            "timestamp": str(uuid.uuid4()),
            "parts": []
        }

        # 1. BUILD VISUAL SCENE
        report(f"Building Visual Components...")
        print("[3DGen] Starting Hierarchical Assembly...")
        scene = self.assemble_hierarchical(recipe, progress_callback=progress_callback)
        print(f"[3DGen] Assembly Complete. Total Mass: {scene.metadata.get('total_mass_kg', 0):.2f}kg")
        
        # Export Visual Mesh (GLB)
        try:
            report("Exporting GLB (Visual)...")
            data = scene.export(file_type='glb')
            with open(visual_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"[3DGen] Visual export failed: {e}")

        # 2. GENERATE COLLISION HULLS
        report("Generating Collision Hulls (NASA Subsystems)...")
        collision_scene = trimesh.Scene()
        geom_items = list(scene.geometry.items())
        
        # Calculate scene scale for relative thresholds
        try:
            scene_extents = scene.extents
            max_dim = np.max(scene_extents) if scene_extents is not None else 1.0
        except:
            max_dim = 1.0
            
        detail_threshold = max_dim * 0.05 # 5% size threshold for "detail" parts
        
        # Iterate over nodes that actually contain geometry to ensure valid transforms
        for node_name in scene.graph.nodes:
             # Skip the root or nodes without geometry
             geom_name = scene.graph.transforms.node_data.get(node_name, {}).get('geometry')
             if not geom_name:
                 continue
                 
             geom = scene.geometry[geom_name]
             # print(f"  > Extracting NASA Collision Hull for: {node_name}...") # Silenced for 125k performance
             
             try:
                 # Extract world transform (absolute world position/orientation)
                 transform = scene.graph.get(node_name)[0]
             except Exception:
                 print(f"    [WARN] Transform lookup failed for {node_name}, using Identity.")
                 transform = np.eye(4)

             # Optimization: Smart Collision Detail
             # For small parts (bolts, wires), use fast OBB. For major parts, use accurate Convex Hull.
             bbox_diag = np.linalg.norm(geom.bounding_box.extents)
             is_detail = bbox_diag < detail_threshold
             
             if is_detail:
                 # Fast Path: Oriented Bounding Box
                 hull = geom.bounding_box_oriented
             else:
                 # High-Fidelity Path: Full Convex Hull
                 hull = self.extract_collision_mesh(geom)

             collision_scene.add_geometry(hull, node_name=node_name, transform=transform)
             
             physics_data["parts"].append({
                 "name": node_name,
                 "mass_kg": float(geom.metadata.get("mass_kg", 0)),
                 "center_of_mass_local": geom.metadata.get("center_of_mass_local", [0,0,0]),
                 "inertia_tensor_local": geom.metadata.get("inertia_tensor_local", []),
                 "volume_m3": geom.metadata.get("volume_m3", 0),
                 "density_kgm3": geom.metadata.get("density_kgm3", 0),
                 "material": geom.metadata.get("material", "default"),
                 "is_detail": bool(is_detail),
                 # New Simulation Fields
                 "joint": geom.metadata.get("joint"),
                 "actuators": geom.metadata.get("actuators"),
                 "sensors": geom.metadata.get("sensors")
             })

        try:
            # STL Export for Collision
            report("Exporting STL (Collision)...")
            collision_data = collision_scene.export(file_type='stl')
            with open(collision_path, 'wb') as f:
                f.write(collision_data)
        except Exception as e:
            print(f"[3DGen] Collision export failed: {e}")

        # 3. Export Physics Metadata
        report("Saving Physics Data...")
        with open(physics_path, 'w') as f:
            json.dump(physics_data, f, indent=2)

        # 4. UNITY AUTO-IMPORT
        if unity_path and os.path.exists(unity_path):
            try:
                report(f"Copying to Unity...")
                import shutil
                dest_dir = Path(unity_path)
                shutil.copy2(visual_path, dest_dir / f"{filename}_Visual.glb")
                shutil.copy2(collision_path, dest_dir / f"{filename}_Collision.stl")
                print(f"[3DGen] Copied assets to Unity: {dest_dir}")
            except Exception as e:
                print(f"[3DGen] Unity Copy Failed: {e}")
        else:
            current_step += 1 # Maintain step count consistency

        report("Generation Complete.")

        return {
            "visual": str(visual_path.absolute()),
            "collision": str(collision_path.absolute()),
            "physics": str(physics_path.absolute()),
            "sdf": "" 
        }
# --- AEROSPACE ENGINEERING RECIPES ---

MARS_ROVER_ENGINEERING = {
    "project": "MARS ROVER - PERSEVERANCE CLASS",
    "parts": [
        # Chassis (Subsystem: Structures)
        {"name": "Main_Chassis", "type": "box", "dims": [1.5, 1.0, 2.2], "pos": [0, 0.5, 0], "material": "aluminum_6061", "subsystem": "structures", "system": "frame"},
        # Power System (Subsystem: GNC/Power)
        {"name": "MMRTG_Generator", "type": "cylinder", "dims": [0.3, 0.8], "pos": [0, 1.0, -1.0], "material": "steel_dark", "subsystem": "gnc", "system": "energy"},
        # Mobility System (Subsystem: Mechanical)
        {"name": "Wheel_Front_Left", "type": "cylinder", "dims": [0.25, 0.2], "pos": [0.9, 0, 0.8], "rot": [1.57, 0, 0], "material": "titanium_grade_5", "subsystem": "mechanical", "system": "mechanical"},
        {"name": "Wheel_Front_Right", "type": "cylinder", "dims": [0.25, 0.2], "pos": [-0.9, 0, 0.8], "rot": [1.57, 0, 0], "material": "titanium_grade_5", "subsystem": "mechanical", "system": "mechanical"},
        # Sensor Mast (Subsystem: GNC)
        {"name": "Pancam_Mast", "type": "cylinder", "dims": [0.05, 1.2], "pos": [0, 1.2, 0.6], "material": "titanium_red", "subsystem": "gnc", "system": "default"}
    ]
}

STARSHIP_ENGINEERING = {
    "project": "SPACEX STARSHIP - HLS",
    "parts": [
        # Main Hull
        {"name": "Propellant_Tank_Main", "type": "cylinder", "dims": [4.5, 12.0], "pos": [0, 0, 0], "material": "stainless_steel_304", "subsystem": "structures", "system": "frame"},
        # Nosecone
        {"name": "Fairing_Nose", "type": "cone", "dims": [4.5, 6.0], "pos": [0, 9.0, 0], "material": "stainless_steel_304", "subsystem": "aero", "system": "armor"},
        # Raptors
        {"name": "Raptor_Center_1", "type": "cone", "dims": [0.7, 1.5], "pos": [0, -6.5, 0], "material": "steel_chrome", "subsystem": "propulsion", "system": "engine"},
        {"name": "Raptor_Vacuum_1", "type": "cone", "dims": [1.1, 2.5], "pos": [2.0, -7.0, 0], "material": "steel_dark", "subsystem": "propulsion", "system": "engine"}
    ]
}

if __name__ == "__main__":
    gen = Structural3DGenerator()
    path = gen.generate_glb(MARS_ROVER_ENGINEERING, "mars_rover_sim.glb")
    print(f"Generated Simulation-Ready Space Model: {path}")
