import os
import sys

# Ensure backend dir is in path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from coder_engine import AdvancedCoderEngine

def test_engine():
    print("--- 1. Initializing Engine ---")
    project_path = os.getcwd()
    engine = AdvancedCoderEngine(project_path)
    
    print("\n--- 2. Testing Structural Info (on server.py) ---")
    server_path = os.path.join(project_path, "backend", "server.py")
    if os.path.exists(server_path):
        info = engine.get_structural_info(server_path)
        print(info)
    else:
        print("server.py not found for testing.")

    print("\n--- 3. Testing RAG Query (Stream) ---")
    user_query = "How is the 3D model generation handled in this project?"
    try:
        print("Stream started...")
        for chunk in engine.query_stream(user_query):
            # chunk is a Partial[CoderResponse] object
            # accessing .thought_process might fail if it's None in the first chunk
            thought = chunk.thought_process if chunk.thought_process else "..."
            sys.stdout.write(f"\rThinking: {thought[-50:]}")
            sys.stdout.flush()
        
        print("\nStream finished.")
        print("\nFinal Operations:")
        if chunk.operations:
            for op in chunk.operations:
                print(f"- {op.operation}: {op.path}")
    except Exception as e:
        print(f"\nQuery failed: {e}")

if __name__ == "__main__":
    test_engine()
