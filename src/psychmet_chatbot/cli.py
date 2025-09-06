"""Command-line interface for PsychMet Chatbot"""

import argparse
import logging
from pathlib import Path

from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .chatbot import PsychMetChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_documents(args):
    """Process PDF documents"""
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    
    data_dir = Path(args.directory)
    documents = processor.process_pdfs(data_dir)
    
    if documents:
        vector_manager.add_documents(documents)
        print(f"Successfully processed {len(documents)} chunks")
    else:
        print("No documents found to process")

def interactive_chat(args):
    """Start interactive chat session"""
    chatbot = PsychMetChatbot()
    
    print("PsychMet Chatbot - Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        response = chatbot.chat(question)
        print(f"\nBot: {response['answer']}")
        
        if args.show_sources and response['source_documents']:
            print("\nSources:")
            for i, doc in enumerate(response['source_documents'], 1):
                print(f"{i}. {doc.page_content[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="PsychMet Chatbot CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDF documents")
    process_parser.add_argument(
        "--directory", "-d",
        default="./data",
        help="Directory containing PDFs"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--show-sources", "-s",
        action="store_true",
        help="Show source documents"
    )
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_documents(args)
    elif args.command == "chat":
        interactive_chat(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()