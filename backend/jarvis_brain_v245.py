"""
JARVIS V24.5 - Memory-First Cognitive Brain
============================================
6-Step Cognitive Pipeline:
  Step 0: Memory Pre-fetch (get context BEFORE planning)
  Step 1: Plan (memory-informed tool selection)
  Step 2: Execute Workers (handled by dispatcher)
  Step 3: Review (verify results, detect issues)
  Step 4: Update Memory (store lessons, preferences, mistakes)
  Step 5: Respond (final synthesis)

The brain is always "grounded" in memory context before acting.
"""

import ollama
import json
import time
import chromadb
from typing import Dict, Any, List, Optional
from config import MODEL_BRAIN
from jarvis_logger import emit_sys_log


class JarvisCognitiveBrain:
    """
    V24.5 Memory-First Cognitive Brain
    
    Unlike V24 which just routes, V24.5:
    - Pre-fetches memory context before planning
    - Reviews results after execution
    - Learns from outcomes (stores lessons, mistakes)
    - Personalizes based on preferences
    """
    
    def __init__(self, memory_path: str = "./jarvis_memory_v2"):
        self.model = MODEL_BRAIN
        self._init_memory(memory_path)
        print(f"[Brain V24.5] Initialized cognitive brain with {self.model}")
    
    def _init_memory(self, memory_path: str):
        """Initialize ChromaDB with cognitive collections."""
        try:
            self.chroma = chromadb.PersistentClient(path=memory_path)
            
            # Existing collections
            self.facts = self.chroma.get_or_create_collection("long_term_facts")
            self.assets = self.chroma.get_or_create_collection("system_assets")
            self.contacts = self.chroma.get_or_create_collection("contacts")
            
            # NEW V24.5 Cognitive collections
            self.lessons = self.chroma.get_or_create_collection("lessons")
            self.preferences = self.chroma.get_or_create_collection("preferences")
            self.mistakes = self.chroma.get_or_create_collection("mistakes")
            self.actions = self.chroma.get_or_create_collection("action_history")
            
            print("[Brain V24.5] Memory collections ready: facts, assets, lessons, preferences, mistakes, actions")
            
            # WARM-UP: Force indices into RAM
            self._warm_up_memory()
            
        except Exception as e:
            emit_sys_log(f"[Brain V24.5] Memory init warning: {e}", "WARN")
            self.facts = self.lessons = self.preferences = self.mistakes = self.actions = None

    def _warm_up_memory(self):
        """Perform dummy queries to pre-load indices into RAM."""
        print("[Brain V24.5] Warming up memory indices...")
        try:
            warmup_query = ["warmup"]
            if self.facts: self.facts.query(query_texts=warmup_query, n_results=1)
            if self.lessons: self.lessons.query(query_texts=warmup_query, n_results=1)
            if self.preferences: self.preferences.query(query_texts=warmup_query, n_results=1)
            if self.mistakes: self.mistakes.query(query_texts=warmup_query, n_results=1)
            print("[Brain V24.5] Memory warm-up complete. Indices are now in RAM.")
        except:
            emit_sys_log("[Brain V24.5] Memory warm-up skipped", "DEBUG")
    
    # ========== STEP 0: MEMORY PRE-FETCH ==========
    def step_0_memory_fetch(self, user_text: str) -> Dict[str, Any]:
        """
        Load relevant context from memory BEFORE planning.
        This grounds the brain in past knowledge.
        """
        emit_sys_log(f"[Brain V24.5] Step 0: Fetching memory context...", "INFO")
        
        context = {
            "relevant_facts": [],
            "past_lessons": [],
            "preferences": [],
            "past_mistakes": [],
            "recent_actions": []
        }
        
        if not self.facts:
            return context
        
        try:
            # Query each collection for relevant info
            if self.facts:
                results = self.facts.query(query_texts=[user_text], n_results=3)
                context["relevant_facts"] = results["documents"][0] if results["documents"] else []
            
            if self.lessons:
                results = self.lessons.query(query_texts=[user_text], n_results=3)
                context["past_lessons"] = results["documents"][0] if results["documents"] else []
            
            if self.preferences:
                results = self.preferences.query(query_texts=[user_text], n_results=3)
                context["preferences"] = results["documents"][0] if results["documents"] else []
            
            if self.mistakes:
                results = self.mistakes.query(query_texts=[user_text], n_results=2)
                context["past_mistakes"] = results["documents"][0] if results["documents"] else []
            
            if self.actions:
                results = self.actions.query(query_texts=[user_text], n_results=3)
                context["recent_actions"] = results["documents"][0] if results["documents"] else []
            
            # Count what we found
            total = sum(len(v) for v in context.values())
            print(f"[Brain V24.5] Step 0 complete: {total} memory items loaded")
            
        except Exception as e:
            print(f"[Brain V24.5] Memory fetch error: {e}")
        
        return context
    
    # ========== STEP 1: PLAN (Lightweight Router ONLY) ==========
    def step_1_plan(self, user_text: str, memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        ROUTING ONLY. Decides which worker to use.
        Does NOT generate the user-facing response (that's step_chat).
        """
        emit_sys_log(f"[Brain V24.5] Step 1: Routing...", "INFO")
        
        system_prompt = f"""Pick the best worker for this user request. Output ONLY JSON.

Workers:
- "web_search": Internet lookups, news, weather
- "vision": Read what's on screen
- "memory": Recall saved info
- "image_gen": Create/generate images
- "automation": Open/close apps, volume, typing
- "none": Conversation, greetings, questions, or if unsure

Rules:
- "Open/Close/Launch/Run/Start [app]" = automation (STRICT)
- "What's on screen", "Read this" = vision
- "Search", "Find out", "What is" (if not in memory) = web_search
- "Generate", "Draw", "Create image" = image_gen
- Chat/questions = none

Examples:
- "Open Chrome" -> {{"worker": "automation", "worker_input": "open chrome", "response": "Certainly sir, opening Chrome now."}}
- "Search for cats" -> {{"worker": "web_search", "worker_input": "cats", "response": "Searching for cats, sir."}}
- "Who are you?" -> {{"worker": "none", "worker_input": "", "response": "I am JARVIS, sir."}}

JSON format:
{{"worker": "worker_name", "worker_input": "input for worker", "response": "Short acknowledgment like 'Certainly, sir. Opening Chrome.'", "reasoning": "1 line why"}}

User: {user_text}"""

        try:
            start_t = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': system_prompt}],
                format='json',
                options={
                    "temperature": 0.3,
                    "repeat_penalty": 1.2,
                    "num_predict": 128,
                    "num_ctx": 2048
                }
            )
            elapsed = time.time() - start_t
            emit_sys_log(f"[Performance] Routing (Step 1) took {elapsed:.2f}s", "INFO")
            
            result = json.loads(response['message']['content'].strip())
            
            # Validate
            valid_workers = ['web_search', 'vision', 'memory', 'image_gen', 'automation', 'none']
            if result.get('worker') not in valid_workers:
                result['worker'] = 'none'
            
            print(f"[Brain V24.5] Step 1 complete (Ollama): {result.get('worker')}")
            return result
            
        except Exception as e:
            emit_sys_log(f"[Brain V24.5] Step 1 error: {e}", "ERROR")
            return {
                "worker": "none",
                "worker_input": "",
                "reasoning": "Router error, defaulting to chat"
            }
    
    # ========== STEP CHAT: Dedicated Conversation ==========
    def generate_chat_response(self, user_text: str, memory_context: Dict[str, Any]) -> str:
        """
        DEDICATED conversation response. Called ONLY when worker='none'.
        Separated from routing so the model can focus 100% on answering.
        """
        emit_sys_log(f"[Brain V24.5] Chat: Generating response...", "INFO")
        
        # Build memory hints
        mem_hints = ""
        if memory_context.get("relevant_facts"):
            mem_hints = f"\nContext: {'; '.join(memory_context['relevant_facts'][:2])}"

        # [NEW] Check for image recall requests in chat
        if any(word in user_text.lower() for word in ["show last image", "last image", "cyberpunk image", "show on screen"]):
            last_image = self.get_last_image()
            if last_image:
                return f"Certainly, sir. Showing the {last_image['description']} on your screen now."
        
        try:
            start_t = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': f"You are JARVIS, a professional AI assistant powered by the Gemma 3:1B offline model. NEVER call yourself GPT-4 or OpenAI. Be concise and clear. NEVER repeat yourself.{mem_hints}"},
                    {'role': 'user', 'content': user_text}
                ],
                options={
                    "temperature": 0.7,
                    "repeat_penalty": 1.2,
                    "repeat_last_n": 128,
                    "num_predict": 512,
                    "num_ctx": 4096,
                    "top_k": 40,
                    "top_p": 0.9
                }
            )
            elapsed = time.time() - start_t
            emit_sys_log(f"[Performance] Chat Generation took {elapsed:.2f}s", "INFO")
            
            reply = response['message']['content'].strip()
            reply = self._clean_repetition(reply)
            print(f"[Brain V24.5] Chat response (Ollama): {reply[:80]}...")
            return reply
            
        except Exception as e:
            print(f"[Brain V24.5] Chat error: {e}")
            return "I'm having a momentary lapse, sir. Could you repeat that?"
    
    @staticmethod
    def _clean_repetition(text: str) -> str:
        """Post-process: remove repeated words/phrases from model output."""
        import re
        # Remove immediate word-level repetition: "the the" -> "the"
        text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
        # Remove repeated short phrases (2-4 words): "created by created by" -> "created by"
        text = re.sub(r'\b((?:\w+\s+){1,3}\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # ========== STEP 3: REVIEW ==========
    def step_3_review(self, goal: str, worker_result: Dict[str, Any], 
                      memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review results after worker execution.
        Check for mistakes, missing info, contradictions.
        Decide if retry/correction needed.
        """
        emit_sys_log(f"[Brain V24.5] Step 3: Reviewing results...", "INFO")
        
        # For simple commands or errors, skip detailed review
        if worker_result.get("status") == "error":
            return {
                "success": False,
                "issue": worker_result.get("error", "Unknown error"),
                "needs_retry": False,
                "lesson": f"Action failed: {worker_result.get('error', 'unknown')}"
            }
        
        # For successful results, quick validation
        review_prompt = f"""Review this task result:

Goal: {goal}
Result: {json.dumps(worker_result, indent=2)[:500]}

Answer in JSON:
{{
    "success": true/false,
    "quality": "good/partial/poor",
    "issue": "any problem found or null",
    "lesson": "what to remember for next time (1 line)"
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': review_prompt}],
                format='json',
                options={
                    "temperature": 0.7,
                    "repeat_penalty": 1.2,
                    "num_ctx": 4096
                }
            )
            
            review = json.loads(response['message']['content'].strip())
            print(f"[Brain V24.5] Step 3 complete: quality={review.get('quality', 'unknown')}")
            return review
            
        except Exception as e:
            print(f"[Brain V24.5] Step 3 error: {e}")
            return {"success": True, "quality": "unknown", "lesson": f"Task '{goal}' completed."}
    
    # ========== STEP 4: UPDATE MEMORY ==========
    def step_4_update_memory(self, goal: str, worker: str, 
                             worker_result: Dict[str, Any], 
                             review: Dict[str, Any]) -> None:
        """
        Store lessons learned, mistakes, and action history.
        This builds the brain's long-term learning.
        """
        emit_sys_log(f"[Brain V24.5] Step 4: Updating memory...", "INFO")
        
        timestamp = time.time()
        
        try:
            # Always log the action
            if self.actions:
                action_desc = f"{goal} -> {worker}: {review.get('quality', 'done')}"
                self.actions.add(
                    documents=[action_desc],
                    metadatas=[{"worker": worker, "timestamp": timestamp, "success": review.get("success", True)}],
                    ids=[f"action_{int(timestamp)}"]
                )
            
            # Store lesson if there is one
            lesson = review.get("lesson")
            if lesson and self.lessons:
                self.lessons.add(
                    documents=[lesson],
                    metadatas=[{"goal": goal[:100], "timestamp": timestamp}],
                    ids=[f"lesson_{int(timestamp)}"]
                )
                print(f"[Brain V24.5] Stored lesson: {lesson[:50]}")
            
            # Store mistake if action failed
            if not review.get("success", True) and review.get("issue") and self.mistakes:
                mistake = f"For '{goal[:50]}': {review['issue']}"
                self.mistakes.add(
                    documents=[mistake],
                    metadatas=[{"timestamp": timestamp}],
                    ids=[f"mistake_{int(timestamp)}"]
                )
                print(f"[Brain V24.5] Stored mistake for future avoidance")
            
            print(f"[Brain V24.5] Step 4 complete: memory updated")
            
        except Exception as e:
            print(f"[Brain V24.5] Step 4 error: {e}")
    
    # ========== STEP 5: RESPOND ==========
    def step_5_respond(self, initial_response: str, worker: str,
                       worker_result: Dict[str, Any], review: Dict[str, Any]) -> str:
        """
        Generate final response, possibly enhanced with worker results.
        For info-gathering workers, synthesize the findings.
        """
        emit_sys_log(f"[Brain V24.5] Step 5: Generating response...", "INFO")
        
        # For simple actions, use initial response
        if worker in ["automation", "image_gen", "none"]:
            # [NEW] Add a specific mention of memory recording
            self.add_fact(f"User asked to {worker} with input: {worker_result.get('summary', 'done')}")
            return initial_response
        
        # For info workers, synthesize if successful
        if worker_result.get("status") == "success":
            summary = worker_result.get("summary", "")
            if summary and len(summary) > 20:
                try:
                    synth_prompt = f"""Summarize this for the user in 1-2 sentences.
Be JARVIS (British, polite, concise).
Data: {summary[:800]}"""
                    
                    response = ollama.chat(
                        model=self.model,
                        messages=[{'role': 'user', 'content': synth_prompt}],
                        options={
                            "temperature": 0.7,
                            "repeat_penalty": 1.2,
                            "num_ctx": 4096,
                            "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
                        }
                    )
                    return response['message']['content'].strip()
                except:
                    pass
        
        # Fallback
        if worker_result.get("status") == "error":
            return f"I'm sorry sir, {worker_result.get('summary', 'something went wrong')}."
        
        return initial_response
    
    # ========== ASSET RECALL ==========
    def get_last_image(self) -> Optional[Dict[str, str]]:
        """Retrieve the last generated image from the assets collection."""
        if not self.assets:
            return None
        try:
            # Query assets specifically for "image" type
            results = self.assets.get(
                where={"type": "image"},
                limit=1
            )
            # chroma .get() doesn't guarantee order without metadata sort, but we usually add only a few
            # A better way is to query action_history or use timestamps in metadata
            if results["ids"]:
                # Actually, ChromaDB .get doesn't have sort. Let's use metadata query.
                all_images = self.assets.get(where={"type": "image"})
                if not all_images["ids"]: return None
                
                # Sort by timestamp in metadata (if available)
                indexed_images = []
                for i in range(len(all_images["ids"])):
                    indexed_images.append({
                        "path": all_images["documents"][i],
                        "description": all_images["metadatas"][i].get("description", "image"),
                        "timestamp": all_images["metadatas"][i].get("timestamp", 0)
                    })
                
                indexed_images.sort(key=lambda x: x["timestamp"], reverse=True)
                return indexed_images[0]
        except Exception as e:
            print(f"[Brain V24.5] Image recall failed: {e}")
        return None

    # ========== FULL COGNITIVE PIPELINE ==========
    async def process_cognitive(self, user_text: str) -> Dict[str, Any]:
        """
        Full 6-step cognitive pipeline (async).
        Steps 2 (execute) is handled externally by dispatcher.
        """
        import asyncio
        
        # Step 0: Memory pre-fetch
        memory_context = await asyncio.to_thread(self.step_0_memory_fetch, user_text)
        
        # Step 1: Plan
        plan_result = await asyncio.to_thread(self.step_1_plan, user_text, memory_context)
        
        return {
            "memory_context": memory_context,
            "plan": plan_result,
            "worker": plan_result.get("worker", "none"),
            "worker_input": plan_result.get("worker_input", ""),
            "initial_response": plan_result.get("response", ""),
            "reasoning": plan_result.get("reasoning", "")
        }
    
    def process_cognitive_sync(self, user_text: str) -> Dict[str, Any]:
        """Synchronous version of process_cognitive (steps 0-1 only)."""
        memory_context = self.step_0_memory_fetch(user_text)
        plan_result = self.step_1_plan(user_text, memory_context)
        
        return {
            "memory_context": memory_context,
            "plan": plan_result,
            "worker": plan_result.get("worker", "none"),
            "worker_input": plan_result.get("worker_input", ""),
            "initial_response": plan_result.get("response", ""),
            "reasoning": plan_result.get("reasoning", "")
        }
    
    # ========== PREFERENCE LEARNING ==========
    def add_preference(self, preference: str) -> None:
        """Manually add a user preference."""
        if self.preferences:
            self.preferences.add(
                documents=[preference],
                metadatas=[{"timestamp": time.time()}],
                ids=[f"pref_{int(time.time())}"]
            )
            print(f"[Brain V24.5] Added preference: {preference}")
    
    def add_fact(self, fact: str) -> None:
        """Manually add a fact about the user."""
        if self.facts:
            try:
                self.facts.add(
                    documents=[fact],
                    metadatas=[{"timestamp": time.time(), "source": "system"}],
                    ids=[f"fact_{int(time.time())}"]
                )
                emit_sys_log(f"[Brain V24.5] Recorded: {fact}")
            except:
                pass


# Singleton
brain_v245 = JarvisCognitiveBrain()
