"""
Complete RAGAS Implementation Example for PsychMet Chatbot
This script demonstrates how to implement and use RAGAS evaluation
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psychmet_chatbot.ragas_integration import (
    PsychMetRAGASIntegration,
    create_psychometrics_test_dataset
)
from psychmet_chatbot.chatbot import PsychMetChatbot
from psychmet_chatbot.vector_store import VectorStoreManager
from psychmet_chatbot.document_processor import DocumentProcessor


async def main():
    """Main function demonstrating RAGAS implementation"""
    
    print("🎯 RAGAS Implementation for PsychMet Chatbot")
    print("=" * 60)
    print()
    
    print("📋 RAGAS evaluates 7 key metrics:")
    print("1. Faithfulness - How grounded answers are in retrieved contexts")
    print("2. Answer Relevancy - How relevant answers are to questions")
    print("3. Context Precision - Proportion of relevant retrieved contexts")
    print("4. Context Recall - Coverage of ground truth in contexts")
    print("5. Context Relevancy - Overall relevancy of contexts")
    print("6. Answer Similarity - Semantic similarity to ground truth")
    print("7. Answer Correctness - Factual correctness of answers")
    print()
    
    # Step 1: Initialize the chatbot system
    print("🚀 Step 1: Initializing PsychMet Chatbot System")
    print("-" * 50)
    
    try:
        print("📚 Loading document processor...")
        doc_processor = DocumentProcessor()
        
        print("🗃️  Setting up vector store...")
        vector_store_manager = VectorStoreManager(doc_processor)
        vector_store = vector_store_manager.create_or_load_store()
        
        print("🤖 Initializing chatbot...")
        chatbot = PsychMetChatbot(vector_store)
        
        print("✅ Chatbot system ready!")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return
    
    # Step 2: Set up RAGAS evaluation
    print("🔬 Step 2: Setting up RAGAS Evaluation System")
    print("-" * 50)
    
    try:
        print("🧪 Creating RAGAS integration...")
        ragas_integration = PsychMetRAGASIntegration(chatbot)
        
        print("📝 Loading test dataset...")
        questions, ground_truths = create_psychometrics_test_dataset()
        
        print(f"📊 Prepared {len(questions)} evaluation questions")
        print("✅ RAGAS evaluation system ready!")
        print()
        
    except Exception as e:
        print(f"❌ Failed to setup RAGAS: {e}")
        return
    
    # Step 3: Run evaluation on a subset (for demo)
    print("🏃‍♂️ Step 3: Running RAGAS Evaluation (Demo - 2 questions)")
    print("-" * 50)
    
    # Use first 2 questions for quick demo
    demo_questions = questions[:2]
    demo_ground_truths = ground_truths[:2]
    
    print("Demo Questions:")
    for i, q in enumerate(demo_questions, 1):
        print(f"{i}. {q}")
    print()
    
    try:
        print("🔄 Running evaluation...")
        results = await ragas_integration.evaluate_chatbot_performance(
            demo_questions,
            demo_ground_truths,
            output_dir="./evaluation_results"
        )
        
        print("\n" + "🎉 RAGAS EVALUATION COMPLETED!")
        print("=" * 60)
        print(results['report'])
        
        # Display file locations
        print("\n📁 Generated Files:")
        print(f"• Dataset: {results['dataset_path']}")
        print(f"• Results: {results['results_path']}")
        print(f"• Report: {results['report_path']}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Show how to run full evaluation
    print("\n" + "📖 How to Run Full Evaluation:")
    print("-" * 40)
    print("To evaluate all questions, run:")
    print("python scripts/run_ragas_evaluation.py --mode full")
    print()
    print("To create custom questions:")
    print("python scripts/run_ragas_evaluation.py --mode sample")
    print("python scripts/run_ragas_evaluation.py --mode custom --questions-file sample_evaluation_questions.json")
    print()
    
    # Step 5: Interpretation guide
    print("📊 How to Interpret RAGAS Scores:")
    print("-" * 40)
    print("• 0.8-1.0: Excellent performance")
    print("• 0.6-0.8: Good performance")
    print("• 0.4-0.6: Average performance")
    print("• 0.0-0.4: Needs improvement")
    print()
    print("Key Insights:")
    print("• Low Faithfulness → Reduce hallucinations")
    print("• Low Context Precision → Improve retrieval filtering")
    print("• Low Context Recall → Increase retrieval coverage")
    print("• Low Answer Correctness → Improve generation quality")
    print()
    
    print("✨ RAGAS implementation demonstration complete!")


if __name__ == "__main__":
    # Check if OpenAI API key is available
    from psychmet_chatbot.config import get_openai_api_key
    
    try:
        api_key = get_openai_api_key()
        if not api_key:
            print("❌ OpenAI API key not found!")
            print("Please set OPENAI_API_KEY in your .env file or Streamlit secrets")
            sys.exit(1)
        
        # Run the main demonstration
        asyncio.run(main())
        
    except Exception as e:
        print(f"❌ Failed to run RAGAS demonstration: {e}")
        import traceback
        traceback.print_exc()
