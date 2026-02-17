
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

from jarvis_brain_v245 import brain_v245

def test_routing():
    user_text = "Open Chrome"
    print(f"--- START TEST ---")
    print(f"Testing routing for: '{user_text}'")
    
    # Mock memory context
    memory_context = {
        "relevant_facts": [],
        "past_lessons": [],
        "preferences": [],
        "past_mistakes": [],
        "recent_actions": []
    }
    
    try:
        print(f"Calling step_1_plan...")
        plan = brain_v245.step_1_plan(user_text, memory_context)
        print("\nPlan Result:")
        print(json.dumps(plan, indent=2))
        print(f"--- END TEST ---")
    except Exception as e:
        print(f"--- ERROR: {e} ---")

if __name__ == "__main__":
    test_routing()
