
import numpy as np
import trimesh
import sys
import os

# Debugging Cone Alignment
def debug_cone():
    print("📐 DEBUG: Cone Primitive Alignment")
    
    # 1. Standard Cone (Z-Up)
    # Trimesh default: Base at Z=0, Tip at Z=Height
    cone = trimesh.creation.cone(radius=1.0, height=2.0)
    print(f"Standard Cone Bounds: {cone.bounds}")
    # Expect Z: [0, 2]
    
    # 2. My Rotation Fix (+90 X)
    cone_rot = cone.copy()
    cone_rot.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    print(f"Rotated (+90 X) Bounds: {cone_rot.bounds}")
    # Z -> -Y?
    # X -> X
    # Y -> Z
    # Check bounds to see orientation
    
    # 3. Y-Up Target (Tip at +Y)
    # Base should be at Y=0 (or centered?)
    
    # 4. Correct Rotation (-90 X?)
    cone_fix = cone.copy()
    cone_fix.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    print(f"Rotated (-90 X) Bounds: {cone_fix.bounds}")

    # 5. Position Check
    # If Hull is height 52, centered at 26.
    # Hull Bounds Y: [0, 52] (Correct?)
    # Cone Height 18.
    # If Cone Base is at Y=0 (after rotation), we need to translate Base to 52.
    # New Center = 52 + (Height/2)? Depends on where the pivot is.
    
if __name__ == "__main__":
    debug_cone()
