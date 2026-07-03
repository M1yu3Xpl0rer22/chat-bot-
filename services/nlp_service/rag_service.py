import os
from typing import List, Dict

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from sentence_transformers import CrossEncoder
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG dependencies not installed: {e}")
    RAG_AVAILABLE = False

class RAGService:
    """Handles PDF ingestion, chunking, embedding, and semantic search."""
    
    def __init__(self, ollama_model: str = "llama3.2:3b", upload_dir: str = "data/uploads"):
        self.embeddings = None
        self.upload_dir = upload_dir
        self.retrievers = {} # session_id -> EnsembleRetriever
        self.cross_encoder = None
        
        if not RAG_AVAILABLE:
            return

        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
        
        try:
            self.embeddings = OllamaEmbeddings(model=ollama_model)
            # Test the connection
            self.embeddings.embed_query("test")
            # Load Cross-Encoder model for re-ranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ RAG Service initialized with Hybrid Search & Re-ranking")
        except Exception as e:
            self.embeddings = None
            print(f"⚠️ RAG Service disabled (Ollama not available for embeddings): {e}")

    def process_document(self, file_path: str, session_id: str) -> bool:
        """Extract text from PDF, chunk it, and store in FAISS vector store."""
        if not self.embeddings or not RAG_AVAILABLE:
            return False
            
        try:
            # 1. Load PDF
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # 2. Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return False

            # 3. Create FAISS vector store
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            
            # 4. Create BM25 sparse retriever
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 10
            
            # 5. Combine into Ensemble Retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
            )
            
            # 6. Save to session
            self.retrievers[session_id] = ensemble_retriever
            
            return True
        except Exception as e:
            print(f"Error processing document: {e}")
            return False
            
    def search(self, query: str, session_id: str, k: int = 3) -> str:
        """Search the Ensemble index and re-rank chunks using Cross-Encoder."""
        if not self.embeddings or session_id not in self.retrievers:
            return ""
            
        try:
            ensemble_retriever = self.retrievers[session_id]
            # Retrieve top 10 from hybrid search
            docs = ensemble_retriever.invoke(query)
            
            if not docs:
                return ""
                
            if self.cross_encoder:
                # Re-rank using Cross-Encoder
                pairs = [[query, doc.page_content] for doc in docs]
                scores = self.cross_encoder.predict(pairs)
                
                # Sort documents by score
                scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
                
                # Take top k docs
                top_docs = [doc for score, doc in scored_docs[:k]]
            else:
                top_docs = docs[:k]
            
            # Combine the chunks into a single context string
            context = "\n\n".join([f"Excerpt {i+1}:\n{doc.page_content}" for i, doc in enumerate(top_docs)])
            return context
        except Exception as e:
            print(f"Error searching document: {e}")
            return ""
            
    def has_document(self, session_id: str) -> bool:
        return session_id in self.retrievers

    def clear_session(self, session_id: str):
        if session_id in self.retrievers:
            del self.retrievers[session_id]
