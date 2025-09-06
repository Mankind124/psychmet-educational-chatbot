"""
RAGAS Integration for PsychMet Chatbot
Integrates RAGAS evaluation with the existing chatbot system
"""

import asyncio
import json
import os
from typing import List, Dict, Any
from datetime import datetime

from .evaluation import RAGASEvaluator, EvaluationDataset, create_evaluation_dataset_from_json, save_evaluation_results, generate_evaluation_report
from .chatbot import PsychMetChatbot
from .config import get_openai_api_key


class PsychMetRAGASIntegration:
    """
    Integration class for evaluating PsychMet chatbot with RAGAS
    """
    
    def __init__(self, chatbot: PsychMetChatbot):
        """
        Initialize RAGAS integration
        
        Args:
            chatbot: Instance of PsychMetChatbot to evaluate
        """
        self.chatbot = chatbot
        self.api_key = get_openai_api_key()
        self.evaluator = RAGASEvaluator(self.api_key)
    
    async def generate_test_dataset(self, questions: List[str], ground_truths: List[str]) -> List[EvaluationDataset]:
        """
        Generate evaluation dataset by running chatbot on test questions
        
        Args:
            questions: List of test questions
            ground_truths: List of expected answers
            
        Returns:
            List of EvaluationDataset objects
        """
        if len(questions) != len(ground_truths):
            raise ValueError("Questions and ground truths must have the same length")
        
        dataset = []
        
        for question, ground_truth in zip(questions, ground_truths):
            print(f"Generating answer for: {question[:50]}...")
            
            # Get chatbot response with context
            response_data = self.chatbot.get_response_with_context(question)
            
            # Extract answer and contexts
            answer = response_data.get('answer', '')
            contexts = [doc.page_content for doc in response_data.get('source_documents', [])]
            
            # Create evaluation example
            example = EvaluationDataset(
                question=question,
                ground_truth=ground_truth,
                contexts=contexts,
                answer=answer,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'model': 'gpt-4.1',
                    'embedding_model': 'text-embedding-3-small'
                }
            )
            
            dataset.append(example)
        
        return dataset
    
    async def evaluate_chatbot_performance(self, 
                                         test_questions: List[str], 
                                         ground_truths: List[str],
                                         output_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """
        Complete evaluation pipeline for the chatbot
        
        Args:
            test_questions: List of questions to test
            ground_truths: List of expected answers
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate dataset
        print("Generating evaluation dataset...")
        dataset = await self.generate_test_dataset(test_questions, ground_truths)
        
        # Save dataset
        dataset_path = os.path.join(output_dir, f"evaluation_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        dataset_dict = [
            {
                'question': example.question,
                'ground_truth': example.ground_truth,
                'contexts': example.contexts,
                'answer': example.answer,
                'metadata': example.metadata
            }
            for example in dataset
        ]
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to: {dataset_path}")
        
        # Run RAGAS evaluation
        print("Running RAGAS evaluation...")
        results = await self.evaluator.evaluate_dataset(dataset)
        
        # Save results
        results_path = os.path.join(output_dir, f"ragas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_evaluation_results(results, results_path)
        
        # Generate and save report
        report = generate_evaluation_report(results)
        report_path = os.path.join(output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Evaluation completed!")
        print(f"Results: {results_path}")
        print(f"Report: {report_path}")
        
        return {
            'results': results,
            'report': report,
            'dataset_path': dataset_path,
            'results_path': results_path,
            'report_path': report_path
        }


def create_psychometrics_test_dataset() -> tuple[List[str], List[str]]:
    """
    Create a sample test dataset specific to psychometrics education
    
    Returns:
        Tuple of (questions, ground_truths)
    """
    
    questions = [
        "What is reliability in psychological testing?",
        "Explain the difference between criterion and construct validity.",
        "What are the different types of reliability coefficients?",
        "How do you interpret a correlation coefficient of 0.85 in test validation?",
        "What is the standard error of measurement and how is it calculated?",
        "Explain the concept of test fairness and bias in psychological assessment.",
        "What is factor analysis and how is it used in test development?",
        "Describe the process of test standardization.",
        "What is the difference between norm-referenced and criterion-referenced testing?",
        "How do you establish convergent and discriminant validity?"
    ]
    
    ground_truths = [
        "Reliability refers to the consistency and stability of test scores across different conditions, times, or forms. It indicates the degree to which a test produces consistent results when measuring the same construct. High reliability means that if the same person takes the test multiple times under similar conditions, they would get similar scores.",
        
        "Criterion validity refers to how well a test predicts or correlates with an external criterion or outcome (like job performance or academic success). Construct validity refers to how well a test measures the theoretical construct or trait it claims to measure (like intelligence or personality). Criterion validity focuses on prediction, while construct validity focuses on theoretical accuracy.",
        
        "The main types of reliability coefficients include: 1) Test-retest reliability (stability over time), 2) Internal consistency reliability (Cronbach's alpha, split-half), 3) Inter-rater reliability (agreement between scorers), and 4) Parallel forms reliability (consistency between equivalent test versions). Each type addresses different sources of measurement error.",
        
        "A correlation coefficient of 0.85 in test validation indicates a strong positive relationship between the test and the criterion being validated against. This suggests good validity evidence, as 85% of the variance (r² = 0.72) in one measure is associated with the other. Values above 0.70 are generally considered strong evidence for validity.",
        
        "The standard error of measurement (SEM) estimates the precision of individual test scores. It's calculated as SEM = SD × √(1 - rₓₓ), where SD is the standard deviation of test scores and rₓₓ is the reliability coefficient. The SEM indicates the range within which a person's true score likely falls, helping interpret score precision.",
        
        "Test fairness means that a test provides equally valid inferences for all groups taking it, regardless of gender, ethnicity, or other characteristics. Test bias occurs when systematic differences in test performance between groups are not related to the construct being measured but to irrelevant factors like cultural background or language differences.",
        
        "Factor analysis is a statistical technique used to identify underlying factors or dimensions that explain correlations among test items. In test development, it helps determine which items measure the same construct, establish the test's structure, and provide evidence for construct validity by showing that items cluster as theoretically expected.",
        
        "Test standardization involves establishing uniform procedures for test administration, scoring, and interpretation. This includes developing standard instructions, time limits, materials, and environmental conditions. It also involves creating norms by testing a representative sample of the target population to establish typical performance levels.",
        
        "Norm-referenced testing compares an individual's performance to that of a reference group (norm group), providing relative standing. Criterion-referenced testing compares performance to a predetermined standard or criterion, indicating whether specific skills or knowledge have been mastered regardless of how others perform.",
        
        "Convergent validity is established by showing that measures of the same construct correlate highly with each other. Discriminant validity is shown when measures of different constructs have low correlations. Together, they provide evidence that a test measures what it claims to measure (convergent) while not measuring unrelated constructs (discriminant)."
    ]
    
    return questions, ground_truths


async def run_psychmet_evaluation_example():
    """
    Example function showing how to run RAGAS evaluation on PsychMet chatbot
    """
    from .chatbot import PsychMetChatbot
    from .vector_store import VectorStoreManager
    
    print("Initializing PsychMet chatbot for evaluation...")
    
    # Initialize components
    vector_store_manager = VectorStoreManager()
    
    # Load or create vector store
    vector_store = vector_store_manager.create_or_load_store()
    
    # Initialize chatbot
    chatbot = PsychMetChatbot(vector_store)
    
    # Create RAGAS integration
    ragas_integration = PsychMetRAGASIntegration(chatbot)
    
    # Get test dataset
    questions, ground_truths = create_psychometrics_test_dataset()
    
    # Run evaluation (use subset for demo)
    demo_questions = questions[:3]  # First 3 questions for demo
    demo_ground_truths = ground_truths[:3]
    
    print(f"Running evaluation on {len(demo_questions)} questions...")
    
    try:
        results = await ragas_integration.evaluate_chatbot_performance(
            demo_questions, 
            demo_ground_truths,
            output_dir="./evaluation_results"
        )
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(results['report'])
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


if __name__ == "__main__":
    # Run the evaluation example
    asyncio.run(run_psychmet_evaluation_example())
