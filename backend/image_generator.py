"""
JARVIS Image Generator - SDXL Lightning (DreamShaper XL)
=========================================================
High-quality 1024x1024 generation optimized for 8GB Mac.
Uses distilled 4-step generation for near-instant results.
"""

import torch
import os
import gc
import json
import time
import re
from pathlib import Path
from typing import Optional, List
import psutil
import ollama

# Workaround for 'torch' has no attribute 'xpu' error in some diffusers versions
if not hasattr(torch, "xpu"):
    class MockXPU:
        def is_available(self): return False
        def get_device_properties(self, *args, **kwargs): return None
        def empty_cache(self): pass
        def synchronize(self): pass
        def device_count(self): return 0
        def current_device(self): return 0
        def get_device_name(self, *args, **kwargs): return "None"
        def manual_seed(self, *args, **kwargs): pass
        def manual_seed_all(self, *args, **kwargs): pass
        def set_device(self, *args, **kwargs): pass
    torch.xpu = MockXPU()

# Configuration
SDXL_MODEL = "Lykon/dreamshaper-xl-lightning"
OUTPUT_DIR = Path(__file__).parent.parent / "generated_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model state
pipe = None
MODEL_LOADED = False

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def load_model():
    """Load SDXL Lightning with 8GB Mac optimizations."""
    global pipe, MODEL_LOADED
    if MODEL_LOADED: return pipe
    
    print(f"[ImageGen] 🚀 Loading SDXL Lightning (DreamShaper XL)...")
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    
    # Use MPS if on Mac, else CUDA/CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Force float32 on MPS (Intel Mac PyTorch 2.2 lacks float16 upsampling support)
    dtype = torch.float32 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32)
    
    try:
        # High Watermark fix for 8GB Macs - CRITICAL
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL,
            torch_dtype=dtype,
            variant="fp16", # Use fp16 weights to save 6GB disk space
            use_safetensors=True
        )
        
        # SDXL Lightning works best with this scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
        # Mac / GPU Optimizations
        if device == "mps":
            print("[ImageGen] 🛠 Optimizing for Mac (MPS)...")
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            pipe.to("mps")
            pipe.enable_attention_slicing()
            # Upcast VAE to float32 to fix MPS upsampling issues
            try:
                pipe.vae.to(dtype=torch.float32)
            except Exception as vae_e:
                print(f"[ImageGen] ⚠️ Could not upcast VAE to float32: {vae_e}. Attempting to proceed with float16.")
        elif device == "cuda":
            pipe.enable_model_cpu_offload() 
        else:
            pipe.to("cpu")
            
        MODEL_LOADED = True
        print(f"[ImageGen] ✅ SDXL Lightning Ready. Runtime: {device}")
        return pipe
    except Exception as e:
        import traceback
        print(f"[ImageGen] ❌ Model load failed: {e}")
        traceback.print_exc()
        return None

def unload_model():
    global pipe, MODEL_LOADED
    if pipe:
        del pipe
        pipe = None
        MODEL_LOADED = False
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        print("[ImageGen] ♻️ Model unloaded.")

def refine_prompt(prompt: str) -> str:
    """Enhance prompt using LLM with performance monitoring and reliable fallbacks."""
    from config import MODEL_BRAIN
    from jarvis_logger import emit_sys_log
    
    STYLE_SUFFIX = "photorealistic, cinematic lighting, masterpiece, 1024x1024, sharp focus, highly detailed textures"
    refined_fallback = f"{prompt}, {STYLE_SUFFIX}"
    
    try:
        print(f"[Brain Comfy] 🧠 Refining prompt: '{prompt}'")
        start_t = time.time()
        
        # Test connection quickly before trying to refine
        try:
            # We don't want to use the global 'ollama' directly if it hasn't initialized correctly
            # Late import inside the function to avoid module-level hangs
            import ollama as ollama_lib
        except ImportError:
            return refined_fallback

        refine_instruction = (
            f"As a master visual artist, enhance this image prompt into a descriptive, artistic scene: '{prompt}'\n"
            f"Focus on lighting, textures, and atmosphere for a high-end SDXL model.\n"
            f"Output ONLY the descriptive prompt in one short paragraph. No talk."
        )
        
        # Try to call Ollama with a shorter timeout window (if supported by library version)
        response = ollama_lib.chat(
            model=MODEL_BRAIN, 
            messages=[{'role': 'user', 'content': refine_instruction}],
            options={"temperature": 0.5, "num_predict": 100}
        )
        
        refined = response['message']['content'].strip()
        elapsed = time.time() - start_t
        emit_sys_log(f"[Performance] Image Prompt Refinement took {elapsed:.2f}s", "INFO")
        
        # Basic cleaning
        refined = re.sub(r'here is.*?[:\.]|prompt:.*?[:\.]', '', refined, flags=re.IGNORECASE).strip()
        refined = re.sub(r'^["\'\s]+|["\'\s]+$', '', refined)
        
        # If refined is too short or empty, use fallback
        if len(refined) < 3:
            return refined_fallback
            
        final_prompt = f"{refined}, {STYLE_SUFFIX}"
        return final_prompt
        
    except Exception as e:
        print(f"[Brain Comfy] ⚠️ Refinement failed: {e}")
        return refined_fallback

def generate_local(prompt: str) -> Optional[Path]:
    """Primary generator for JARVIS (SDXL Lightning)."""
    try:
        pipe = load_model()
        if not pipe: return None
        
        print(f"[ImageGen] 🎨 Generating: {prompt[:60]}...")
        start_t = time.time()
        
        # 4 STEPS for Lightning! High quality 1024x1024.
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, ugly, distorted, low resolution, text, watermark, bad anatomy",
            num_inference_steps=4,
            guidance_scale=1.0, # Lightning needs low CFG
            height=1024,
            width=1024
        ).images[0]
        
        elapsed = time.time() - start_t
        print(f"[ImageGen] ✨ Generation COMPLETE in {elapsed:.2f}s")
        
        filename = f"gen_{int(time.time())}.png"
        save_path = OUTPUT_DIR / filename
        result.save(save_path)
        
        # Index in Memory
        try:
            from memory_engine_v2 import memory_v2
            memory_v2.add_asset("image", str(save_path.absolute()), prompt)
        except: pass
        
        return save_path
    except Exception as e:
        import traceback
        print(f"[ImageGen] ❌ Generation failed: {e}")
        traceback.print_exc()
        return None
    finally:
        # Aggressive RAM Policy: ALWAYS unload after generation on 8GB Macs
        if torch.backends.mps.is_available():
            unload_model()

def process_image_request(text: str) -> dict:
    """Unified entry point for JARVIS brain."""
    prompt = text.lower()
    for trigger in ["generate image of", "create image of", "draw a", "make an image for", "make an image of"]:
        prompt = prompt.replace(trigger, "")
    prompt = prompt.strip()
    
    final_prompt = refine_prompt(prompt)
    
    path = generate_local(final_prompt)
    
    if path:
        return {"success": True, "path": str(path.resolve()), "prompt": final_prompt}
    return {"success": False, "error": "Generation failed"}

# Compatibility for server.py
def generate_image(prompt: str, count: int = 1, mode: str = "normal") -> List[Path]:
    final_prompt = refine_prompt(prompt)
    path = generate_local(final_prompt)
    return [path] if path else []
