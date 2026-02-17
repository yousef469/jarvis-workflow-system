import json
import requests
import re
from pathlib import Path
from three_d_generator import Structural3DGenerator

class JarvisThreeDEngine:
    """
    AI-Powered 3D Model Generator.
    Connects User Prompts -> LLM (Qwen) -> JSON Recipe -> Structural3DGenerator -> GLB
    """
    
    def __init__(self):
        self.generator = Structural3DGenerator()
        self.output_dir = Path("generated_models")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_from_prompt(self, prompt: str) -> str:
        """
        Generates a 3D model from a text prompt or a direct JSON recipe.
        Returns the absolute path to the generated GLB file.
        """
        print(f"[3D-LLM] Processing input: '{prompt[:50]}...'")
        
        # 0. Check if input is a direct JSON blueprint
        is_json = False
        recipe = None
        
        stripped_prompt = prompt.strip()
        
        # Try parsing as JSON first for direct blueprint bypass
        try:
            # Clean markdown if present
            clean_json = stripped_prompt
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.strip("`").strip()
            
            recipe = json.loads(clean_json)
            print(f"[3D-LLM] JSON parsed OK. Keys: {list(recipe.keys())}")
            # Normalize complex blueprints (e.g., Ingenuity style)
            recipe = self._normalize_recipe(recipe)
            
            # Basic validation: must have 'parts' after normalization
            if isinstance(recipe, dict) and "parts" in recipe:
                is_json = True
                print(f"[3D-LLM] Direct JSON Blueprint detected. {len(recipe['parts'])} parts found. Bypassing LLM.")
            else:
                print(f"[3D-LLM] JSON parsed but no 'parts' found after normalization. Keys: {list(recipe.keys()) if isinstance(recipe, dict) else 'not-dict'}")
        except Exception as e:
            print(f"[3D-LLM] JSON parse/normalize failed: {e}")
            import traceback; traceback.print_exc()

        # 1. Get JSON Recipe from LLM if not provided
        if not is_json:
            recipe = self._query_llm(prompt)
            if not recipe:
                raise Exception("Failed to generate valid 3D recipe from LLM.")
            
        print(f"[3D-LLM] Recipe ready: {recipe.get('project', 'Unknown')}")
        
        # 2. Generate GLB
        filename = f"{recipe.get('project', 'model').replace(' ', '_')}_{hash(str(recipe))}"
        result = self.generator.generate_glb(recipe, filename=filename)
        
        return result["visual"]

    def _normalize_recipe(self, data: dict) -> dict:
        """
        Attempts to find 'parts' inside a complex nested JSON structure.
        Supports:
          - Flat: { parts: [...] }
          - Assemblies: { assemblies: { name: { parts: [...] } } }
          - Structure: { structure: { root: { components: [{ parts: [...] }] } } }
          - Groups:  { parts: [{ group: "X", parts: [...] }] }
        """
        if not isinstance(data, dict):
            return data
            
        # If already flat with valid parts, return as-is
        if "parts" in data and isinstance(data["parts"], list):
            # But check if parts need normalization (shape mapping etc.)
            data["parts"] = self._normalize_parts_list(data["parts"])
            return data

        all_parts = []
        
        # Strategy 1: assemblies -> {assembly_name} -> parts OR [ {name: ..., parts/details: ...} ]
        assemblies = data.get("assemblies", {})
        
        # Helper to process a single assembly dict
        def process_assembly_dict(asm_data, asm_name="Unknown"):
            if not isinstance(asm_data, dict): return
            
            # 1a. Standard "parts" list
            asm_parts = asm_data.get("parts", [])
            if isinstance(asm_parts, list):
                for item in asm_parts:
                    if isinstance(item, dict):
                        # Check for nested groups
                        group_key = "group" if "group" in item else ("leg" if "leg" in item else None)
                        if group_key and "parts" in item:
                            group_parts = item.get("parts", [])
                            if isinstance(group_parts, list):
                                for gp in group_parts:
                                    if isinstance(gp, dict):
                                        gp["_assembly"] = asm_name
                                        gp["_group"] = item.get(group_key, "")
                                        all_parts.append(gp)
                        else:
                            item["_assembly"] = asm_name
                            all_parts.append(item)
            
            # 1b. Manifest "details" list of strings
            if "details" in asm_data and isinstance(asm_data["details"], list):
                 for det in asm_data["details"]:
                     if isinstance(det, str):
                         generated_parts = self._explode_manifest_entry(det, asm_name)
                         all_parts.extend(generated_parts)

        # Handle Dict-style assemblies
        if isinstance(assemblies, dict):
            for asm_name, asm_data in assemblies.items():
                process_assembly_dict(asm_data, asm_name)
        
        # Handle List-style assemblies (New Manifest feature)
        elif isinstance(assemblies, list):
            for asm_data in assemblies:
                if isinstance(asm_data, dict):
                    name = asm_data.get("name", "Assembly")
                    process_assembly_dict(asm_data, name)
        
        # Strategy 2: structure -> root -> components -> parts
        if not all_parts:
            structure = data.get("structure", {})
            if isinstance(structure, dict):
                root = structure.get("root", {})
                if isinstance(root, dict):
                    components = root.get("components", [])
                    if isinstance(components, list):
                        for comp in components:
                            if isinstance(comp, dict):
                                parts = comp.get("parts", [])
                                if isinstance(parts, list):
                                    all_parts.extend(parts)

        # Strategy 3: Recursive search as final fallback
        if not all_parts:
            all_parts = self._find_all_parts_recursive(data)

        if all_parts:
            data["parts"] = self._normalize_parts_list(all_parts)
            # Generate project name if missing
            if "project" not in data:
                data["project"] = data.get("model_name", "Generated_Model")
            print(f"[3D-LLM] Normalized {len(data['parts'])} parts from recipe")
        
        return data

    def _find_all_parts_recursive(self, obj):
        """Recursively find ALL parts arrays and combine them."""
        results = []
        if isinstance(obj, dict):
            if "parts" in obj and isinstance(obj["parts"], list):
                for item in obj["parts"]:
                    if isinstance(item, dict):
                        # Nested groups (e.g., leg or group)
                        group_key = "group" if "group" in item else ("leg" if "leg" in item else None)
                        if group_key and "parts" in item:
                            results.extend(item.get("parts", []))
                        else:
                            results.append(item)
            else:
                for v in obj.values():
                    results.extend(self._find_all_parts_recursive(v))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._find_all_parts_recursive(item))
        return results

    def _normalize_parts_list(self, parts_list):
        """Normalize a list of part dicts: map shapes, parse dims/pos."""
        
        # Comprehensive shape -> type mapping
        shape_map = {
            # Box variants
            "box": "box", "cube": "box", "box_small": "box", "chamfered": "box",
            "enclosure": "box", "bracket": "box", "housing": "box", "box_angled": "box",
            "internal_frame": "box", "avionics": "box", "structure": "box",
            # Panel/plate variants
            "rectangle": "panel", "square": "panel", "plate": "panel", "thin_rectangle": "panel",
            "panel": "panel", "layer": "panel", "face": "panel", "curved_plate": "panel",
            "grid_overlay": "panel", "solar_cell": "panel", "substrate": "panel",
            # Cylinder variants
            "cylinder": "cylinder", "cylinder_complex": "cylinder", "cylinder_small": "cylinder",
            "cylinder_flat": "cylinder", "mast": "cylinder", "hub": "cylinder", "small_cylinder": "cylinder",
            "rod": "rod", "strut": "cylinder", "pipe": "cylinder", "wire_rod": "rod",
            "joint_connector": "cylinder", "coupling": "cylinder", "hinge": "cylinder",
            # Sphere variants
            "sphere": "sphere", "hemisphere": "sphere", "dome": "sphere",
            "sphere_joint": "sphere", "ball": "sphere", "joint": "sphere",
            # Cone variants
            "cone": "cone", "cone_spike": "cone", "spike": "cone", "pointed_cone": "cone",
            "cap": "cone", "point": "cone", "tip": "cone", "foot": "cone",
            # Wing/airfoil variants
            "wing": "wing", "airfoil": "wing", "tapered_airfoil": "wing",
            "blade": "wing", "fin": "wing", "rotor_blade": "wing",
            # Tube variants
            "tube": "tube", "hollow_cylinder": "tube", "hollow_tube": "tube", "tapered_tube": "tube",
            # Torus -> tube (closest approximation)
            "torus": "tube", "ring": "tube", "swashplate": "tube",
            # Nozzle variants
            "nozzle": "nozzle", "thruster": "nozzle", "exhaust": "nozzle",
            # Bolt
            "bolt": "bolt", "screw": "bolt", "fastener": "bolt",
            # Specialized
            "hex_tile": "hex_tile", "rib": "rib", "pin": "rod", "wire": "rod",
            "small_dot": "sphere", "recessed_circle": "cylinder", "clamping_joint": "cylinder"
        }
        
        normalized = []
        for i, p in enumerate(parts_list):
            if not isinstance(p, dict):
                continue
            
            # 1. Map shape -> type
            raw_shape = p.get("shape", p.get("type", "box")).lower().strip()
            p_type = shape_map.get(raw_shape)
            if not p_type:
                # Fuzzy match: check if any key is IN the shape name
                p_type = "box"  # default
                for key, val in shape_map.items():
                    if key in raw_shape:
                        p_type = val
                        break
            p["type"] = p_type
            
            # 2. Parse dims (handle radius, thickness, taper, scale scalar)
            raw_dims = p.get("dims", p.get("scale", None))
            
            # Airfoil special handling
            if p_type == "wing" and ("width_root" in p or "taper" in p or "length" in p):
                length = float(p.get("length", 0.6))
                width = float(p.get("width_root", 0.1))
                taper = float(p.get("taper", 0.5))
                thickness = width * 0.1
                raw_dims = [length, width, thickness]
            
            # Radius/Thickness special handling
            elif "radius" in p or "thickness" in p or "r" in p:
                r = float(p.get("radius", p.get("r", 0.05)))
                h = float(p.get("height", p.get("h", p.get("length", 0.2))))
                if p_type == "tube":
                    r_in = float(p.get("r_inner", r * 0.8))
                    raw_dims = [r, r_in, h]
                elif p_type in ("cylinder", "rod", "cone", "nozzle", "bolt"):
                    raw_dims = [r, h]
                elif p_type == "sphere":
                    raw_dims = [r]
                else:
                    thickness = float(p.get("thickness", 0.01))
                    raw_dims = [r*2, r*2, thickness]
            
            elif raw_dims is None or isinstance(raw_dims, (int, float)):
                # Handle scalar scale or missing dims
                val = float(raw_dims) if raw_dims is not None else None
                length_val = p.get("length")
                
                if val is not None and length_val is None:
                    # Scalar scale [val, val, val]
                    if p_type in ("box", "panel"): raw_dims = [val, val, val if p_type=="box" else 0.01]
                    elif p_type in ("cylinder", "rod", "cone", "nozzle", "bolt"): raw_dims = [val, val*4]
                    elif p_type == "sphere": raw_dims = [val]
                    else: raw_dims = [val, val, val]
                elif length_val is not None:
                    # Handle "length" field
                    l = float(length_val)
                    if p_type in ("cylinder", "rod", "cone", "nozzle", "bolt"): raw_dims = [0.05, l]
                    elif p_type == "tube": raw_dims = [0.05, 0.04, l]
                    elif p_type == "wing": raw_dims = [l, 0.1, 0.01]
                    else: raw_dims = [l, l, l]
            
            p["dims"] = self._parse_dims(raw_dims, p_type)
            
            # 3. Parse position (handle string positions)
            raw_pos = p.get("pos", p.get("position", [0, 0, 0]))
            if isinstance(raw_pos, list):
                p["pos"] = []
                for x in raw_pos:
                    try: p["pos"].append(float(x))
                    except: p["pos"].append(0.0)
                while len(p["pos"]) < 3: p["pos"].append(0.0)
            else:
                p["pos"] = [0, 0, 0]
            
            # 4. Parse rotation
            raw_rot = p.get("rotation", p.get("rot", [0, 0, 0]))
            if isinstance(raw_rot, list):
                p["rot"] = []
                for x in raw_rot:
                    try: p["rot"].append(float(x))
                    except: p["rot"].append(0.0)
                while len(p["rot"]) < 3: p["rot"].append(0.0)
            else:
                p["rot"] = [0, 0, 0]
            
            # 5. Map color/material
            mat = p.get("material", p.get("color", "default")).lower()
            mat_map = {
                "black": "carbon_fiber", "carbon": "carbon_fiber", "matte": "carbon_fiber",
                "silver": "aluminum_6061", "grey": "aluminum_6061", "titanium": "titanium_red",
                "gold": "gold_foil", "kapton": "gold_foil", "solar": "solar_blue",
                "blue": "solar_blue", "white": "titanium_white", "iridescent": "solar_blue",
                "glass": "nasa_glass"
            }
            p["material"] = "default"
            for k, v in mat_map.items():
                if k in mat:
                    p["material"] = v
                    break
            
            # 6. Set name
            p["name"] = str(p.get("id", f"part_{i}"))
            
            normalized.append(p)
        
        return normalized

    def _parse_dims(self, d, p_type):
        """Converts strings like '1.2m' or '14cm x 14cm' to float lists."""
        if d is None:
            defaults = {
                "box": [0.2, 0.2, 0.2], 
                "panel": [0.5, 0.5, 0.01], 
                "cylinder": [0.05, 0.2], 
                "sphere": [0.1], 
                "wing": [1.0, 0.2, 0.02],
                "cone": [0.1, 0.2],
                "nozzle": [0.1, 0.2],
                "tube": [0.1, 0.08, 0.2],
                "bolt": [0.01, 0.05],
                "rib": [0.1, 0.1, 0.01],
                "rod": [0.02, 0.5],
                "hex_tile": [0.1, 0.02]
            }
            return defaults.get(p_type, [0.1, 0.1, 0.1])

        if isinstance(d, (int, float)):
            if p_type in ["cylinder", "rod", "cone", "nozzle", "bolt", "hex_tile"]:
                return [0.05, float(d)] # radius + height
            elif p_type == "sphere":
                return [float(d)]
            else:
                return [float(d), float(d), float(d) if p_type == "box" else 0.01]
            
        if isinstance(d, list): 
            # Ensure it has the right number of dimensions
            expected = {
                "box": 3, "panel": 3, "cylinder": 2, "sphere": 1, "wing": 3, 
                "tube": 3, "cone": 2, "nozzle": 2, "rod": 2, "bolt": 2, "hex_tile": 2
            }
            count = expected.get(p_type, 3)
            current = [float(x) for x in d if isinstance(x, (int, float, str))]
            if len(current) >= count: return current[:count]
            # Pad with last value or 0.1
            while len(current) < count:
                current.append(current[-1] if current else 0.1)
            return current


    def _explode_manifest_entry(self, entry: str, assembly_name: str) -> list:
        """
        Parses descriptive strings like '48x_Solar_Cells' or '4x_Legs' 
        and generates procedural 3D parts with smart layouts.
        """
        # 1. Determine Global Assembly Offset (Vertical Stacking)
        asm_y = 0.0
        lower_asm = assembly_name.lower()
        if "solar" in lower_asm: asm_y = 0.8
        elif "upper" in lower_asm: asm_y = 0.5
        elif "lower" in lower_asm: asm_y = 0.3
        elif "landing" in lower_asm: asm_y = -0.2
        elif "avionics" in lower_asm: asm_y = 0.1
        
        # 1b. Parse Count: "48x..." -> 48
        count = 1
        name = entry
        match = re.match(r"^(\d+)x[-_ ]*(.*)", entry, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            name = match.group(2)
        
        # Clean name
        clean_name = name.replace("_", " ").strip()
        
        # 2. Determine Shape & Dimensions from Keywords
        shape = "box"
        dims = [0.1, 0.1, 0.1]
        material = "default"
        
        lower_name = clean_name.lower()
        if "solar" in lower_name or "cell" in lower_name:
            shape = "panel"
            dims = [0.05, 0.05, 0.002]
            material = "solar_blue"
        elif "leg" in lower_name or "strut" in lower_name:
            shape = "tube"
            dims = [0.02, 0.015, 0.3]
            material = "carbon_fiber"
        elif "rotor" in lower_name or "blade" in lower_name:
            shape = "wing"
            dims = [1.2, 0.15, 0.02]
            material = "carbon_fiber"
        elif "mast" in lower_name or "shaft" in lower_name:
            shape = "cylinder"
            dims = [0.05, 1.0]
            material = "aluminum_6061"
        elif "motor" in lower_name:
            shape = "cylinder"
            dims = [0.1, 0.15]
            material = "silver"
        elif "battery" in lower_name or "cell" in lower_name:
            shape = "cylinder"
            dims = [0.018, 0.065] # 18650 size-ish
            material = "green"
        elif "pcb" in lower_name or "board" in lower_name:
            shape = "panel"
            dims = [0.1, 0.1, 0.005]
            material = "green_pcb"
        elif "screw" in lower_name or "bolt" in lower_name or "rivet" in lower_name:
            shape = "bolt"
            dims = [0.005, 0.02]
            material = "silver"
            
        generated = []
        
        # 3. Procedural Layout
        if count == 1:
            # Single item at origin (plus assembly offset)
            generated.append({
                "name": clean_name,
                "shape": shape,
                "dims": dims,
                "pos": [0, asm_y, 0],
                "color": material,
                "_assembly": assembly_name
            })

        
        # 3b. Add a central "Core" for the assembly to anchor parts
        generated.append({
            "name": f"{assembly_name}_Core_Anchor",
            "shape": "cylinder",
            "dims": [0.04, 0.1],
            "pos": [0, asm_y, 0],
            "color": "grey",
            "_assembly": assembly_name
        })
            
        if "solar" in lower_name:
            # Grid Layout for Solar Cells
            cols = int(count**0.5) # Square-ish grid
            rows = (count // cols) + (1 if count % cols > 0 else 0)
            spacing = dims[0] * 1.1
            
            for i in range(count):
                r = i // cols
                c = i % cols
                x = (c - cols/2) * spacing
                z = (r - rows/2) * spacing
                generated.append({
                    "name": f"{clean_name}_{i+1}",
                    "shape": shape,
                    "dims": dims,
                    "pos": [x, asm_y, z], # Assembly offset
                    "rot": [0, 0, 0],
                    "color": material,
                    "_assembly": assembly_name
                })
                
        elif "leg" in lower_name:
            # Radial Symmetry for Legs
            radius = 0.5
            for i in range(count):
                angle = (360 / count) * i
                # Convert polar to cartesian
                import math
                rad = math.radians(angle)
                x = radius * math.cos(rad)
                z = radius * math.sin(rad)
                
                generated.append({
                    "name": f"{clean_name}_{i+1}",
                    "shape": shape,
                    "dims": dims,
                    "pos": [x, asm_y - 0.1, z],
                    "rot": [0, -angle, 35], # Splayed out
                    "color": material,
                    "_assembly": assembly_name
                })
                
        elif "blade" in lower_name:
            # Rotor Blades - Radial
            for i in range(count):
                angle = (360 / count) * i
                generated.append({
                    "name": f"{clean_name}_{i+1}",
                    "shape": shape,
                    "dims": dims,
                    "pos": [0, asm_y, 0], 
                    "rot": [0, angle, -5], # Slight pitch
                    "color": material,
                    "_assembly": assembly_name
                })
        else:
            # Default: Random/Cluster pile (or Stack for batteries)
            for i in range(count):
                offset_x = (i % 5) * 0.05
                offset_z = (i // 5) * 0.05
                generated.append({
                    "name": f"{clean_name}_{i+1}",
                    "shape": shape,
                    "dims": dims,
                    "pos": [offset_x, asm_y + (0.1 * (i%3)), offset_z], 
                    "color": material,
                    "_assembly": assembly_name
                })
                
        return generated
        
        if isinstance(d, str):
            # Extract numbers
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", d.replace("_", " "))
            floats = [float(x) for x in nums]
            
            # Simple unit conversion
            if "cm" in d.lower(): floats = [x * 0.01 for x in floats]
            if "mm" in d.lower(): floats = [x * 0.001 for x in floats]
            
            if floats: 
                return self._parse_dims(floats, p_type) # Recurse to handle count
            
        # Final fallback
        return self._parse_dims(None, p_type)

    def _query_llm(self, user_prompt: str) -> dict:
        """Queries Ollama to convert text -> Structural3D JSON."""
        
        system_prompt = """You are a 3D Engineering AI. Your goal is to convert user descriptions into JSON recipes for the Jarvis Structural 3D Engine.
        
        OUTPUT FORMAT:
        You must output ONLY valid JSON. Do not include markdown blocks (```json).
        
        SCHEMA:
        {
            "project": "Short Project Name",
            "parts": [
                {
                    "name": "PartName",
                    "type": "box|cylinder|sphere|cone|tube|nozzle|wing",
                    "dims": [0.0-100.0, ...], 
                    "pos": [x, y, z],
                    "rot": [x_deg, y_deg, z_deg],
                    "material": "aluminum_6061|steel_dark|solar_blue|nasa_glass|default",
                    "parent": "world" (or name of other part)
                }
            ]
        }
        
        RULES:
        1. Use 'box' (d=[x,y,z]), 'cylinder' (d=[r,h]), 'sphere' (d=[r]), 'cone' (d=[r,h]), 'tube' (d=[r_out, r_in, h]), 'nozzle' (d=[r,h]).
        2. 'dims' are in meters. detailed parts should be 0.1-1.0m. Vehicles can be 5-50m.
        3. Be creative but structurally sound.
        """
        
        payload = {
            "model": "qwen2.5-coder:7b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a 3D model recipe for: {user_prompt}"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }
        
        try:
            res = requests.post("http://localhost:11434/api/chat", json=payload)
            if res.status_code != 200:
                print(f"[3D-LLM] API Error: {res.text}")
                return None
                
            content = res.json()["message"]["content"]
            
            # Clean Markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            return json.loads(content)
            
        except Exception as e:
            print(f"[3D-LLM] Error parsing LLM response: {e}")
            print(f"[DEBUG] Raw content: {content if 'content' in locals() else 'N/A'}")
            return None

# Singleton for easy import
threed_llm = JarvisThreeDEngine()
