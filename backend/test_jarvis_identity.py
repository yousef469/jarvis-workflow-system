import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from jarvis_brain_v245 import brain_v245

def test_identity():
    print("\n--- JARVIS IDENTITY TEST ---")
    user_text = "Who are you?"
    memory_context = brain_v245.step_0_memory_fetch(user_text)
    response = brain_v245.generate_chat_response(user_text, memory_context)
    print(f"User: {user_text}")
    print(f"Jarvis: {response}")
    
    if "JARVIS" in response.upper():
        print("✅ Identity verified: System knows it is JARVIS.")
    else:
        print("⚠️ Warning: System did not use its new name in the response.")

if __name__ == "__main__":
    test_identity()
