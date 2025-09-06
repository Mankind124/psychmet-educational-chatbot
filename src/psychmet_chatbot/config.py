"""Configuration management for PsychMet Chatbot with FAISS"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class Config:
    """Application configuration for FAISS-based system"""
    
    # API Keys - Handle both Streamlit secrets and environment variables
    @classmethod
    def get_openai_api_key(cls):
        """Get OpenAI API key from Streamlit secrets or environment variables"""
        # Try Streamlit secrets first (for cloud deployment)
        try:
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
        except:
            pass
        
        # Fall back to environment variables (for local development)
        return os.getenv("OPENAI_API_KEY")
    
    OPENAI_API_KEY = get_openai_api_key()
    
    # Model Configuration (updated defaults to match your choices)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Updated default
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")  # Updated default
    
    # FAISS Vector Store Configuration
    FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
    INDEX_NAME = os.getenv("INDEX_NAME", "psychmet_index")
    FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat").lower()
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    DOCUMENT_BATCH_SIZE = int(os.getenv("DOCUMENT_BATCH_SIZE", 100))
    
    # LLM Configuration (updated temperature default)
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))  # Updated to match your .env
    
    # Retrieval Configuration
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", 4))
    
    # Paths
    DATA_DIR = Path("./data")
    DOCS_DIR = Path("./docs")
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        api_key = cls.get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or Streamlit secrets")
        
        # Update the class variable with the retrieved key
        cls.OPENAI_API_KEY = api_key
        
        # Create directories if they don't exist
        cls.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCS_DIR.mkdir(exist_ok=True)
        
        # Log configuration in development
        print(f"Using LLM: {cls.LLM_MODEL}")
        print(f"Using Embeddings: {cls.EMBEDDING_MODEL}")
        print(f"FAISS Index: {cls.FAISS_INDEX_PATH}")
    
    @classmethod
    def get_index_file_path(cls) -> Path:
        """Get the full path to the FAISS index file"""
        return cls.FAISS_INDEX_PATH / f"{cls.INDEX_NAME}.faiss"

config = Config()