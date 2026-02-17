"""
Shared AI Clients - Singleton Pattern
====================================
Prevents redundant model loading and extra RAM usage.
"""

import ollama
import instructor
from langchain_ollama import OllamaLLM
from openai import OpenAI
from config import MODEL_BRAIN

# Single Instructor Client
_instructor_client = None

def get_instructor_client(model=MODEL_BRAIN):
    global _instructor_client
    if _instructor_client is None:
        try:
            # Use OpenAI bridge with MD_JSON (Most robust for 1B models)
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            # MD_JSON allows the model to wrap output in ```json tags
            _instructor_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
            print(f"[Shared] ✅ Instructor MD_JSON bridge initialized for {model}.")
        except Exception as e:
            print(f"[Shared] ❌ Instructor initialization failed: {e}")
            _instructor_client = instructor.from_provider(f"ollama/{model}")
            
    return _instructor_client

# Single LangChain Instance
_langchain_llm = None
_ollama_embeddings = None

def get_langchain_llm(model=MODEL_BRAIN):
    global _langchain_llm
    if _langchain_llm is None:
        from langchain_openai import ChatOpenAI
        print(f"[Shared] Initializing Local-First LangChain Bridge for {model}...")
        _langchain_llm = ChatOpenAI(
            model=model,
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama" 
        )
    return _langchain_llm

def get_ollama_embeddings():
    global _ollama_embeddings
    if _ollama_embeddings is None:
        from langchain_ollama import OllamaEmbeddings
        _ollama_embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://127.0.0.1:11434"
        )
    return _ollama_embeddings

# Legacy Fallback Singleton for server.py
class SharedBrain:
    def chat(self, prompt, system_prompt=None, temperature=0.5):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(
                model=MODEL_BRAIN,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": 4096,
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"[Shared Brain] ❌ Error: {e}")
            return "I'm sorry sir, my reasoning core is currently under heavy load."

shared_brain = SharedBrain()
