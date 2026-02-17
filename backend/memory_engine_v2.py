import chromadb
import uuid
import time
import os
import subprocess
from typing import List, Dict, Optional

class JarvisMemoryV2:
    def __init__(self, db_path="./jarvis_memory_v2"):
        """
        Initializes ChromaDB persistent storage.
        Three collections:
        1. Facts (User preferences, personal info)
        2. Assets (Images, PDFs, screenshots)
        3. Contacts (Names and numbers)
        """
        self.db_path = os.path.abspath(db_path)
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Long-term facts with cosine distance for best semantic match
        self.facts = self.client.get_or_create_collection(
            name="long_term_facts",
            metadata={"hnsw:space": "cosine"}
        )
        
        # System assets (files + descriptions)
        self.assets = self.client.get_or_create_collection(
            name="system_assets",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Contacts (Phone book)
        self.contacts = self.client.get_or_create_collection(
            name="contacts"
        )

    def add_fact(self, text: str, source: str = "conversation"):
        """Stores a new fact about the user."""
        fact_id = str(uuid.uuid4())
        self.facts.add(
            documents=[text],
            metadatas=[{"source": source, "timestamp": time.time()}],
            ids=[fact_id]
        )
        print(f"[Memory V2] 🧠 Fact stored: {text}")

    def recall(self, query: str, limit: int = 5) -> List[str]:
        """Searches for semantic matches in facts."""
        try:
            results = self.facts.query(query_texts=[query], n_results=limit)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"[Memory V2] Recall error: {e}")
            return []

    def get_context(self, query: str = "", limit: int = 5) -> str:
        """
        Retrieves context relevant to the current query.
        Used to ground the LLM's response.
        """
        search_query = query if query else "user preferences and personal facts"
        facts = self.recall(search_query, limit=limit)
        
        if not facts:
            return ""
            
        context_str = "\nRELEVANT MEMORY:\n" + "\n".join([f"- {f}" for f in facts])
        return context_str

    def add_asset(self, type: str, path: str, description: str):
        """Indexes a generated file so it can be 'shown' later."""
        asset_id = str(uuid.uuid4())
        self.assets.add(
            documents=[description],
            metadatas=[{"type": type, "path": path, "timestamp": time.time()}],
            ids=[asset_id]
        )
        print(f"[Memory V2] 📁 Asset indexed: {description} ({type})")

    def find_asset(self, query: str, preferred_type: str = None) -> Optional[Dict]:
        """
        Searches system_assets for a matching description.
        Returns the path and metadata if found.
        """
        try:
            # Query the assets collection
            results = self.assets.query(query_texts=[query], n_results=5)
            
            if results["metadatas"] and results["metadatas"][0]:
                metadatas = results["metadatas"][0]
                documents = results["documents"][0]
                
                # 1. Try to find preferred type match
                if preferred_type:
                    for i, meta in enumerate(metadatas):
                        if meta["type"] == preferred_type:
                            return {
                                "path": meta["path"],
                                "type": meta["type"],
                                "description": documents[i]
                            }
                
                # 2. Return best overall match
                meta = metadatas[0]
                return {
                    "path": meta["path"],
                    "type": meta["type"],
                    "description": documents[0]
                }
            return None
        except Exception as e:
            print(f"[Memory V2] Search error: {e}")
            return None

    def add_contact(self, name: str, number: str) -> bool:
        """Stores a phone contact."""
        self.contacts.add(
            documents=[name],
            metadatas=[{"number": str(number), "timestamp": time.time()}],
            ids=[f"contact_{name.lower().replace(' ', '_')}"]
        )
        print(f"[Memory V2] 📱 Contact saved: {name} -> {number}")
        return True

    def get_contact(self, name: str) -> Optional[str]:
        """Retrieves a contact number by name."""
        try:
            results = self.contacts.query(query_texts=[name], n_results=1)
            if results["metadatas"] and results["metadatas"][0]:
                return results["metadatas"][0][0]["number"]
            return None
        except:
            return None

    def open_asset(self, path: str) -> bool:
        """Opens a file using macOS 'open'."""
        if not path or not os.path.exists(path):
            print(f"[Memory V2] Open failed: File does not exist at {path}")
            return False
            
        try:
            subprocess.run(["open", path])
            return True
        except Exception as e:
            print(f"[Memory V2] Open error: {e}")
            return False

# Singleton instance
memory_v2 = JarvisMemoryV2()
memory_engine = memory_v2
