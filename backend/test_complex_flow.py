
import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "jarvis-system/backend"))

from crew_orchestrator import orchestrator
from instructor_merger import merger

async def test_flow():
    print("🚀 Testing Complex Image Gen Flow...")
    
    # Simulate Router Decision: Complex -> Image Gen Worker
    query = "generate an image of a cybernetic cat in a neon city"
    workers = ["image_gen"]
    
    print(f"1. Query: {query}")
    print(f"2. Workers: {workers}")
    
    # 1. Execute Worker (CrewAI)
    print("\n[Step 1] Executing Worker...")
    results = await orchestrator.execute_parallel(query, workers)
    print(f"Worker Result: {results}")
    
    # 2. Merge Results (Instructor)
    print("\n[Step 2] Merging with Instructor...")
    final_report = merger.merge(results)
    print(f"\nFinal Report:\n{final_report}")
    
    if "I have created the image" in final_report:
        print("\n✅ SUCCESS: Instructor confirmed image creation.")
    else:
        print("\n❌ FAILURE: Instructor did not use the correct phrase.")

if __name__ == "__main__":
    asyncio.run(test_flow())
