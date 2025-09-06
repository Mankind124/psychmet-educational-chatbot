"""
CLI script for running RAGAS evaluation on PsychMet chatbot
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psychmet_chatbot.ragas_integration import (
    PsychMetRAGASIntegration, 
    create_psychometrics_test_dataset,
    run_psychmet_evaluation_example
)
from psychmet_chatbot.chatbot import PsychMetChatbot
from psychmet_chatbot.vector_store import VectorStoreManager
from psychmet_chatbot.document_processor import DocumentProcessor


async def run_full_evaluation():
    """Run comprehensive RAGAS evaluation"""
    print("🚀 Starting PsychMet RAGAS Evaluation")
    print("="*50)
    
    try:
        # Initialize chatbot components
        print("📚 Initializing chatbot components...")
        doc_processor = DocumentProcessor()
        vector_store_manager = VectorStoreManager(doc_processor)
        vector_store = vector_store_manager.create_or_load_store()
        chatbot = PsychMetChatbot(vector_store)
        
        # Create RAGAS integration
        print("🔬 Setting up RAGAS evaluator...")
        ragas_integration = PsychMetRAGASIntegration(chatbot)
        
        # Get test dataset
        print("📝 Loading test dataset...")
        questions, ground_truths = create_psychometrics_test_dataset()
        
        print(f"📊 Evaluating {len(questions)} questions...")
        
        # Run evaluation
        results = await ragas_integration.evaluate_chatbot_performance(
            questions, 
            ground_truths,
            output_dir="./evaluation_results"
        )
        
        print("\n" + "🎉 EVALUATION COMPLETED!")
        print("="*50)
        print(results['report'])
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_quick_demo():
    """Run quick demo evaluation with 3 questions"""
    print("🏃‍♂️ Running Quick Demo Evaluation")
    print("="*40)
    
    return await run_psychmet_evaluation_example()


async def run_custom_evaluation(questions_file: str):
    """Run evaluation with custom questions from file"""
    print(f"📂 Running Custom Evaluation from {questions_file}")
    print("="*50)
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data['questions']
        ground_truths = data['ground_truths']
        
        if len(questions) != len(ground_truths):
            raise ValueError("Questions and ground truths must have same length")
        
        # Initialize components
        doc_processor = DocumentProcessor()
        vector_store_manager = VectorStoreManager(doc_processor)
        vector_store = vector_store_manager.create_or_load_store()
        chatbot = PsychMetChatbot(vector_store)
        
        # Run evaluation
        ragas_integration = PsychMetRAGASIntegration(chatbot)
        results = await ragas_integration.evaluate_chatbot_performance(
            questions, 
            ground_truths,
            output_dir="./evaluation_results"
        )
        
        print("\n" + "🎉 CUSTOM EVALUATION COMPLETED!")
        print("="*50)
        print(results['report'])
        
        return results
        
    except Exception as e:
        print(f"❌ Custom evaluation failed: {e}")
        return None


def create_sample_questions_file():
    """Create a sample questions file for custom evaluation"""
    questions, ground_truths = create_psychometrics_test_dataset()
    
    sample_data = {
        "questions": questions[:5],  # First 5 questions
        "ground_truths": ground_truths[:5]
    }
    
    with open("sample_evaluation_questions.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("📝 Created sample_evaluation_questions.json")
    print("Edit this file to add your own questions and ground truths!")


def main():
    parser = argparse.ArgumentParser(description='RAGAS Evaluation for PsychMet Chatbot')
    parser.add_argument(
        '--mode', 
        choices=['full', 'demo', 'custom', 'sample'],
        default='demo',
        help='Evaluation mode: full (all questions), demo (3 questions), custom (from file), sample (create sample file)'
    )
    parser.add_argument(
        '--questions-file',
        type=str,
        help='JSON file with custom questions and ground truths (for custom mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        create_sample_questions_file()
        return
    
    if args.mode == 'custom' and not args.questions_file:
        print("❌ Custom mode requires --questions-file argument")
        return
    
    # Run appropriate evaluation mode
    if args.mode == 'full':
        asyncio.run(run_full_evaluation())
    elif args.mode == 'demo':
        asyncio.run(run_quick_demo())
    elif args.mode == 'custom':
        asyncio.run(run_custom_evaluation(args.questions_file))


if __name__ == "__main__":
    main()
