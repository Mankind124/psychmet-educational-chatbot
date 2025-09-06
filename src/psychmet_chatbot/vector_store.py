"""Vector store management using FAISS"""

import logging
from typing import List, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self):
        config.validate()
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.vector_store = None
        self.index_path = config.get_index_file_path()
    
    def create_or_load_store(self) -> FAISS:
        """Create new or load existing FAISS vector store"""
        # Check if FAISS index files exist (index.faiss and index.pkl)
        index_faiss_path = config.FAISS_INDEX_PATH / "index.faiss"
        index_pkl_path = config.FAISS_INDEX_PATH / "index.pkl"
        
        if index_faiss_path.exists() and index_pkl_path.exists():
            # Load existing index
            self.vector_store = FAISS.load_local(
                str(config.FAISS_INDEX_PATH), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS index from {config.FAISS_INDEX_PATH}")
        else:
            # Create an empty vector store with a dummy document to avoid None
            dummy_text = ["This is a placeholder document for initializing the vector store."]
            dummy_metadata = [{"source": "placeholder", "type": "initialization"}]
            self.vector_store = FAISS.from_texts(dummy_text, self.embeddings, metadatas=dummy_metadata)
            logger.info("Created new empty FAISS index with placeholder document")
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS vector store"""
        if documents:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            if self.vector_store is None:
                # Create new index from scratch with real documents
                self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                logger.info(f"Created new FAISS index with {len(documents)} documents")
            else:
                # Check if this is the first real content (replacing placeholder)
                existing_docs = self.vector_store.similarity_search("", k=1)
                if (len(existing_docs) == 1 and 
                    existing_docs[0].metadata.get("type") == "initialization"):
                    # Replace placeholder with real documents
                    self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                    logger.info(f"Replaced placeholder with {len(documents)} real documents")
                else:
                    # Add to existing real index
                    self.vector_store.add_texts(texts, metadatas=metadatas)
                    logger.info(f"Added {len(documents)} documents to existing FAISS index")
            
            # Save the updated index
            self.vector_store.save_local(str(config.FAISS_INDEX_PATH))
            logger.info("FAISS index saved successfully")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vector_store is None:
            self.create_or_load_store()
        
        if self.vector_store is None:
            logger.warning("No vector store available - no documents have been processed yet")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} similar documents for query")
        return results
    
    def clear_store(self) -> None:
        """Clear the FAISS vector store"""
        if self.index_path.exists():
            self.index_path.unlink()
            logger.info("FAISS index file deleted")
        self.vector_store = None