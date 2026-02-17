import os

SERVER_PATH = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"

with open(SERVER_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Inject Import
if "from model_manager import model_manager" not in content:
    content = content.replace("import ollama", "import ollama\nfrom model_manager import model_manager")
    print("Injected import.")

# 2. Replace Legacy Logic Block
# We identify the start and end of the block we want to replace
START_MARKER = "# Track which model is currently loaded"
END_MARKER = "# VAD & AUDIO SETTINGS"

# The new logic to insert
NEW_LOGIC = """
# =============================================================================
# MODEL MANAGER INTEGRATION
# =============================================================================

def load_brain():
    \"\"\"Wrapper for ModelManager - ensures brain is loaded\"\"\"
    model_manager.switch_to_brain()
    return True
    
def force_unload_model():
    \"\"\"Wrapper for ModelManager - ensures heavy models unloaded\"\"\"
    model_manager.switch_to_brain()

# Helper for compatibility
def load_model(model_name):
    if model_name == model_manager.MODEL_BRAIN: # Use manager constants if possible or global
        model_manager.switch_to_brain()
    elif model_name == model_manager.MODEL_VISION:
        model_manager.switch_to_vision()
    # If using globals from this file
    elif model_name == "qwen3:4b":
         model_manager.switch_to_brain()
    elif model_name == "qwen3-vl:4b":
         model_manager.switch_to_vision()
    else:
         # Fallback for other models
         model_manager._preload_model(model_name)
    return True

# Ensure Startup Load
model_manager.switch_to_brain()

# Cleanup old globals if mistakenly referenced
_current_loaded_model = None
"""

if START_MARKER in content and END_MARKER in content:
    # Find indices
    start_idx = content.find(START_MARKER)
    # We want to find the LAST occurrence of END_MARKER just in case, or the first after start
    end_idx = content.find(END_MARKER, start_idx)
    
    if start_idx != -1 and end_idx != -1:
        # Check if we are deleting too much?
        # The block ends before VAD settings.
        # We need to make sure we don't delete too much.
        # Let's verify the content between.
        
        # Actually, let's look for known strings to be safer.
        # "def load_model(model_name):"
        
        # Let's simplify. We will replace the whole chunk from START_MARKER to just before END_MARKER.
        # We need to preserve END_MARKER.
        
        pre_block = content[:start_idx]
        post_block = content[end_idx:]
        
        content = pre_block + NEW_LOGIC + "\n\n" + post_block
        print("Replaced logic block.")
    else:
        print("Could not find start/end markers.")
else:
    print("Markers not found.")

with open(SERVER_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("Done.")
