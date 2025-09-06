"""Basic usage example for PsychMet Chatbot"""

from psychmet_chatbot import PsychMetChatbot, DocumentProcessor, VectorStoreManager

def main():
    # Initialize components
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    chatbot = PsychMetChatbot()
    
    # Process a PDF (if you have one)
    # documents = processor.process_pdfs("./data")
    # vector_manager.add_documents(documents)
    
    # Example questions
    questions = [
        "What is reliability in psychometrics?",
        "Explain the difference between validity and reliability",
        "What is Item Response Theory?",
        "How do you calculate Cronbach's alpha?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = chatbot.chat(question)
        print(f"A: {response['answer'][:500]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()