"""Memory Worker V24 - Direct ChromaDB (No LlamaIndex)"""

import asyncio
import os
from typing import Dict, Any

class MemoryWorker:
    """Handles memory retrieval using ChromaDB directly (no LlamaIndex)"""
    
    def __init__(self, db_path="./jarvis_memory_v2"):
        self.name = "memory"
        self.db_path = os.path.abspath(db_path)
        self._initialized = False
        self.facts = None
        self.assets = None
        self.contacts = None
    
    def _ensure_initialized(self):
        """Lazy initialize ChromaDB connection"""
        if self._initialized:
            return True
        
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.facts = self.chroma_client.get_or_create_collection("long_term_facts")
            self.assets = self.chroma_client.get_or_create_collection("system_assets")
            self.contacts = self.chroma_client.get_or_create_collection("contacts")
            self._initialized = True
            print("[MemoryWorker] 🧠 ChromaDB connected directly.")
            return True
        except Exception as e:
            print(f"[MemoryWorker] Warning: ChromaDB init failed: {e}")
            return False
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute memory retrieval task"""
        print(f"[MemoryWorker] 🔍 Searching memory for: {query}")
        
        if not self._ensure_initialized():
            return {
                "source": "memory",
                "error": "Memory system offline",
                "summary": "My memory systems are currently unavailable, sir.",
                "status": "error"
            }
        
        try:
            loop = asyncio.get_event_loop()
            
            # Search facts
            facts_results = await loop.run_in_executor(
                None,
                lambda: self.facts.query(query_texts=[query], n_results=5)
            )
            
            # Search assets (images, files)
            assets_results = await loop.run_in_executor(
                None,
                lambda: self.assets.query(query_texts=[query], n_results=3)
            )
            
            # Process facts
            facts_found = []
            if facts_results["documents"] and facts_results["documents"][0]:
                facts_found = facts_results["documents"][0]
            
            # Process assets
            assets_found = []
            if assets_results["metadatas"] and assets_results["metadatas"][0]:
                for i, meta in enumerate(assets_results["metadatas"][0]):
                    doc = assets_results["documents"][0][i] if assets_results["documents"][0] else ""
                    assets_found.append({
                        "path": meta.get("path", ""),
                        "type": meta.get("type", "unknown"),
                        "description": doc
                    })
            
            # Build summary
            summary_parts = []
            if facts_found:
                summary_parts.append("Facts: " + "; ".join(facts_found[:3]))
            if assets_found:
                asset_descs = [a["description"] for a in assets_found if a["description"]]
                if asset_descs:
                    summary_parts.append("Assets: " + "; ".join(asset_descs[:2]))
            
            summary = " | ".join(summary_parts) if summary_parts else "No matching records found."
            
            return {
                "source": "memory",
                "query": query,
                "facts": facts_found,
                "assets": assets_found,
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            print(f"[MemoryWorker] ❌ Recall Error: {e}")
            return {
                "source": "memory",
                "error": str(e),
                "summary": f"Memory recall encountered an issue: {str(e)[:100]}",
                "status": "error"
            }
    
    def open_asset(self, path: str) -> bool:
        """Opens a file using macOS 'open' command"""
        if not path or not os.path.exists(path):
            print(f"[MemoryWorker] Open failed: File does not exist at {path}")
            return False
        
        try:
            import subprocess
            subprocess.run(["open", path])
            return True
        except Exception as e:
            print(f"[MemoryWorker] Open error: {e}")
            return False


# Singleton instance
memory_worker = MemoryWorker()
