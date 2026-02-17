"""
JARVIS V24.5 Dispatcher - Cognitive Pipeline Orchestrator
==========================================================
Orchestrates the full 6-step cognitive pipeline:

  Step 0: Memory Pre-fetch (brain loads context)
  Step 1: Plan (brain decides worker)
  Step 2: Execute (dispatcher runs worker)
  Step 3: Review (brain verifies result)
  Step 4: Update Memory (brain learns)
  Step 5: Respond (brain synthesizes)

This replaces jarvis_dispatcher_v24.py with the cognitive loop.
"""

import asyncio
from typing import Dict, Any
from jarvis_brain_v245 import brain_v245
from workers import (
    web_worker, vision_worker, memory_worker,
    image_gen_worker, automation_worker
)
from jarvis_logger import emit_sys_log


# Worker registry
WORKERS = {
    "web_search": web_worker,
    "vision": vision_worker,
    "memory": memory_worker,
    "image_gen": image_gen_worker,
    "automation": automation_worker,
}


async def dispatch_cognitive(user_text: str) -> Dict[str, Any]:
    """
    Full V24.5 Cognitive Pipeline.
    
    Steps:
    0. Brain pre-fetches memory context
    1. Brain plans (memory-informed)
    2. Dispatcher executes worker
    3. Brain reviews result
    5. Brain generates final response
    """
    emit_sys_log(f"\n{'='*60}\n[V24.5] COGNITIVE PIPELINE START: {user_text[:50]}...\n{'='*60}")
    
    # ========== STEP 0 + 1: Memory Fetch + Plan ==========
    cognitive_result = await brain_v245.process_cognitive(user_text)
    
    worker_name = cognitive_result.get("worker", "none")
    worker_input = cognitive_result.get("worker_input", "")
    initial_response = cognitive_result.get("initial_response", "")
    memory_context = cognitive_result.get("memory_context", {})
    
    print(f"[V24.5] Brain decided: {worker_name} | Reasoning: {cognitive_result.get('reasoning', '')[:50]}")
    
    # ========== STEP 2: EXECUTE WORKER ==========
    if worker_name == "none" or worker_name not in WORKERS:
        # No worker needed - DEDICATED conversation call
        print(f"[V24.5] Chat mode: generating dedicated response...")
        
        # Use the NEW dedicated chat method (not the router's response!)
        chat_response = await asyncio.to_thread(
            brain_v245.generate_chat_response,
            user_text, memory_context
        )
        
        # Still update action history
        await asyncio.to_thread(
            brain_v245.step_4_update_memory,
            user_text, "none", {"status": "success"}, {"success": True, "quality": "good"}
        )
        
        return {
            "response": chat_response,
            "worker": "none",
            "worker_result": None,
            "cognitive": True,
            "status": "success"
        }
    
    # Execute the worker
    worker = WORKERS[worker_name]
    emit_sys_log(f"[V24.5] Step 2: Executing {worker_name}...", "INFO")
    
    try:
        worker_result = await worker.execute(worker_input)
        print(f"[V24.5] Worker returned: {worker_result.get('status', 'unknown')}")
    except Exception as e:
        emit_sys_log(f"[V24.5] Worker error: {e}", "ERROR")
        worker_result = {
            "status": "error",
            "error": str(e),
            "summary": f"Worker failed: {str(e)[:100]}"
        }
    
    # ========== STEP 3: REVIEW ==========
    review = await asyncio.to_thread(
        brain_v245.step_3_review,
        user_text, worker_result, memory_context
    )
    print(f"[V24.5] Review: quality={review.get('quality', 'unknown')}")
    
    # ========== STEP 4: UPDATE MEMORY ==========
    await asyncio.to_thread(
        brain_v245.step_4_update_memory,
        user_text, worker_name, worker_result, review
    )
    
    # ========== STEP 5: RESPOND ==========
    final_response = await asyncio.to_thread(
        brain_v245.step_5_respond,
        initial_response, worker_name, worker_result, review
    )
    
    print(f"[V24.5] COGNITIVE PIPELINE COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        "response": final_response,
        "worker": worker_name,
        "worker_input": worker_input,
        "worker_result": worker_result,
        "review": review,
        "reasoning": cognitive_result.get("reasoning", ""),
        "cognitive": True,
        "status": worker_result.get("status", "success")
    }


def dispatch_cognitive_sync(user_text: str) -> Dict[str, Any]:
    """Synchronous wrapper for dispatch_cognitive()"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, dispatch_cognitive(user_text))
            return future.result()
    else:
        return loop.run_until_complete(dispatch_cognitive(user_text))


# ========== QUICK MODE (Skip review for simple commands) ==========
async def dispatch_quick(user_text: str) -> Dict[str, Any]:
    """
    Faster mode - skips step 3 (review) for simple commands.
    Use this for automation commands where review isn't needed.
    """
    print(f"\n[V24.5 Quick] Processing: {user_text[:50]}...")
    
    # Step 0 + 1: Memory + Plan
    cognitive_result = await brain_v245.process_cognitive(user_text)
    
    worker_name = cognitive_result.get("worker", "none")
    worker_input = cognitive_result.get("worker_input", "")
    memory_context = cognitive_result.get("memory_context", {})
    
    # Simple commands - use dedicated chat
    if worker_name == "none":
        chat_response = await asyncio.to_thread(
            brain_v245.generate_chat_response,
            user_text, memory_context
        )
        return {"response": chat_response, "worker": "none", "status": "success"}
    
    # Step 2: Execute
    if worker_name in WORKERS:
        worker_result = await WORKERS[worker_name].execute(worker_input)
    else:
        worker_result = {"status": "error", "error": "Unknown worker"}
    
    # Step 4: Quick memory update (no full review)
    await asyncio.to_thread(
        brain_v245.step_4_update_memory,
        user_text, worker_name, worker_result,
        {"success": worker_result.get("status") == "success", "quality": "quick"}
    )
    
    # Step 5: Simple response
    if worker_result.get("status") == "error":
        final_response = f"I'm sorry sir, {worker_result.get('summary', 'that failed')}."
    else:
        final_response = worker_result.get("summary", "Done, sir.")
    
    return {
        "response": final_response,
        "worker": worker_name,
        "worker_result": worker_result,
        "status": worker_result.get("status", "success")
    }


# ========== CLASS WRAPPER ==========
class JarvisDispatcherV245:
    """Class wrapper for the V24.5 cognitive dispatcher."""
    
    def __init__(self):
        print("[Dispatcher V24.5] Initialized cognitive dispatcher")
    
    async def dispatch(self, user_text: str) -> Dict[str, Any]:
        return await dispatch_cognitive(user_text)
    
    async def dispatch_quick(self, user_text: str) -> Dict[str, Any]:
        return await dispatch_quick(user_text)
    
    def dispatch_sync(self, user_text: str) -> Dict[str, Any]:
        return dispatch_cognitive_sync(user_text)
    
    # Convenience methods for memory management
    def add_preference(self, preference: str):
        brain_v245.add_preference(preference)
    
    def add_fact(self, fact: str):
        brain_v245.add_fact(fact)


# Singleton
dispatcher_v245 = JarvisDispatcherV245()
