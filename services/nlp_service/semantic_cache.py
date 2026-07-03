import os
import json
from typing import Optional

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.schema import Document
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

class SemanticCache:
    """
    Algorithmic cache using FAISS to store and retrieve previously answered questions.
    Bypasses the LLM if a semantically similar query was asked before.
    """
    def __init__(self, ollama_model: str = "llama3.2:3b", similarity_threshold: float = 0.5):
        self.embeddings = None
        self.vector_store = None
        # In FAISS L2 distance, lower is better. 0 is exact match.
        self.similarity_threshold = similarity_threshold 
        
        if not CACHE_AVAILABLE:
            print("⚠️ Semantic Cache disabled: missing langchain dependencies.")
            return
            
        try:
            self.embeddings = OllamaEmbeddings(model=ollama_model)
            # Test the connection by attempting a small embedding
            self.embeddings.embed_query("test")
            print("✅ Semantic Cache initialized")
        except Exception as e:
            self.embeddings = None
            print(f"⚠️ Semantic Cache disabled (Ollama not available for embeddings): {e}")
            
    def get(self, query: str) -> Optional[dict]:
        """Search for a semantically similar query in cache."""
        if not self.embeddings or not self.vector_store:
            return None
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=1)
            if results:
                doc, score = results[0]
                if score <= self.similarity_threshold:
                    print(f"⚡ Semantic Cache Hit! (Score: {score:.4f})")
                    return json.loads(doc.metadata["response_data"])
            return None
        except Exception as e:
            print(f"Cache search error: {e}")
            return None
            
    def set(self, query: str, response_data: dict):
        """Save a query and its response to the semantic cache."""
        if not self.embeddings:
            return
            
        try:
            doc = Document(
                page_content=query,
                metadata={"response_data": json.dumps(response_data)}
            )
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            else:
                self.vector_store.add_documents([doc])
        except Exception as e:
            print(f"Cache save error: {e}")
