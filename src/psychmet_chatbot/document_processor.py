"""Document processing module for handling PDFs"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF loading and text chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF file"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")
            return []
    
    def load_all_pdfs(self, directory: Path = None) -> List[Document]:
        """Load all PDFs from a directory"""
        if directory is None:
            directory = config.DATA_DIR
        
        all_documents = []
        pdf_files = list(directory.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            documents = self.load_pdf(pdf_file)
            all_documents.extend(documents)
        
        logger.info(f"Loaded {len(all_documents)} total pages from {len(pdf_files)} PDFs")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def process_pdfs(self, directory: Path = None) -> List[Document]:
        """Complete pipeline: load and chunk all PDFs"""
        documents = self.load_all_pdfs(directory)
        if documents:
            return self.chunk_documents(documents)
        return []