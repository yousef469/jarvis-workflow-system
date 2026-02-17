"""
Jarvis Document Engine (LlamaIndex Integration)
===============================================
Uses LlamaIndex to ingest, index, and retrieve documents (PDFs, text) 
using ChromaDB as the vector store and Ollama (Gemma 3) as the LLM.
"""
import os
import chromadb
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import MODEL_ROUTER as MODEL_NAME # Use same model as router

# === Configuration ===
PERSIST_DIR = "./jarvis_memory_v2" # Same path as memory_engine_v2
COLLECTION_NAME = "jarvis_documents"

class JarvisDocumentEngine:
    def __init__(self):
        """Initialize LlamaIndex with local components."""
        print("[Document Engine] Initializing LlamaIndex...")
        
        # 1. Setup LLM (Ollama)
        Settings.llm = Ollama(model=MODEL_NAME, request_timeout=60.0)
        
        # 2. Setup Embeddings (Local HuggingFace)
        # Using a small, efficient model for local embedding
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 3. Setup Vector Store (ChromaDB)
        db_path = os.path.abspath(PERSIST_DIR)
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 4. Load or Initialize Index
        try:
            # Try to load existing index from storage
            self.index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=self.storage_context
            )
            print("[Document Engine] ✅ Loaded existing document index.")
        except Exception as e:
            # If empty, create new empty index
            print(f"[Document Engine] Creating new index (Reason: {e})")
            self.index = VectorStoreIndex.from_documents(
                [], storage_context=self.storage_context
            )
            
    def ingest_document(self, file_path: str):
        """Ingest a single document into the index."""
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
            
        try:
            print(f"[Document Engine] Ingesting: {file_path}")
            documents = SimpleDirectoryReader(
                input_files=[file_path]
            ).load_data()
            
            # Update the index with new document
            self.index.insert_nodes(self.index.as_node_parser().get_nodes_from_documents(documents))
            self.index.storage_context.persist() # Ensure save
            
            return f"Successfully processed {os.path.basename(file_path)}"
        except Exception as e:
            print(f"[Document Engine] Ingestion error: {e}")
            return f"Failed to ingest document: {e}"

    def query_documents(self, query_text: str):
        """Query the document index."""
        try:
            query_engine = self.index.as_query_engine(streaming=True)
            response = query_engine.query(query_text)
            return response
        except Exception as e:
            print(f"[Document Engine] Query error: {e}")
            return "I had trouble accessing the document archives."

# Singleton instance
document_engine = JarvisDocumentEngine()
