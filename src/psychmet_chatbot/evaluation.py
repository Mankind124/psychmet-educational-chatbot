"""
RAGAS (Retrieval-Augmented Generation Assessment) Implementation
Evaluates RAG system performance across multiple dimensions
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationDataset:
    """Represents a single evaluation example"""
    question: str
    ground_truth: str
    contexts: List[str]
    answer: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RAGASMetrics:
    """Container for RAGAS evaluation metrics"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float
    answer_similarity: float
    answer_correctness: float
    overall_score: float

class RAGASEvaluator:
    """
    Comprehensive RAGAS evaluation system for RAG applications
    Implements all core RAGAS metrics for thorough assessment
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        """
        Initialize RAGAS evaluator
        
        Args:
            openai_api_key: OpenAI API key for LLM-based evaluations
            model: OpenAI model to use for evaluations
        """
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.vectorizer = TfidfVectorizer()
        
    async def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate how faithful the answer is to the retrieved contexts
        
        Args:
            answer: Generated answer
            contexts: List of retrieved context documents
            
        Returns:
            Faithfulness score (0-1, higher is better)
        """
        try:
            # Break answer into claims
            claims_prompt = f"""
            Break down the following answer into individual factual claims:
            Answer: {answer}
            
            Return only the claims as a numbered list.
            """
            
            claims_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": claims_prompt}],
                temperature=0.1
            )
            
            claims_text = claims_response.choices[0].message.content
            claims = [claim.strip() for claim in claims_text.split('\n') if claim.strip() and claim[0].isdigit()]
            
            if not claims:
                return 0.0
            
            # Verify each claim against contexts
            supported_claims = 0
            context_text = "\n".join(contexts)
            
            for claim in claims:
                verification_prompt = f"""
                Context: {context_text}
                
                Claim: {claim}
                
                Is this claim supported by the given context? Answer only 'YES' or 'NO'.
                """
                
                verification_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": verification_prompt}],
                    temperature=0.1
                )
                
                if "YES" in verification_response.choices[0].message.content.upper():
                    supported_claims += 1
            
            return supported_claims / len(claims)
            
        except Exception as e:
            logger.error(f"Error in faithfulness evaluation: {e}")
            return 0.0
    
    async def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Answer relevancy score (0-1, higher is better)
        """
        try:
            # Generate multiple questions from the answer
            question_gen_prompt = f"""
            Given this answer, generate 3 different questions that this answer would appropriately respond to:
            Answer: {answer}
            
            Return only the questions, one per line.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question_gen_prompt}],
                temperature=0.3
            )
            
            generated_questions = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
            
            if not generated_questions:
                return 0.0
            
            # Calculate semantic similarity between original and generated questions
            all_questions = [question] + generated_questions
            
            # Use TF-IDF for similarity calculation
            tfidf_matrix = self.vectorizer.fit_transform(all_questions)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Error in answer relevancy evaluation: {e}")
            return 0.0
    
    async def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate precision of retrieved contexts (relevant contexts / total contexts)
        
        Args:
            question: Original question
            contexts: List of retrieved contexts
            
        Returns:
            Context precision score (0-1, higher is better)
        """
        try:
            relevant_contexts = 0
            
            for context in contexts:
                relevance_prompt = f"""
                Question: {question}
                Context: {context}
                
                Is this context relevant for answering the question? Answer only 'YES' or 'NO'.
                """
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": relevance_prompt}],
                    temperature=0.1
                )
                
                if "YES" in response.choices[0].message.content.upper():
                    relevant_contexts += 1
            
            return relevant_contexts / len(contexts) if contexts else 0.0
            
        except Exception as e:
            logger.error(f"Error in context precision evaluation: {e}")
            return 0.0
    
    async def evaluate_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Evaluate recall of retrieved contexts (how much of ground truth is covered)
        
        Args:
            ground_truth: Expected correct answer
            contexts: List of retrieved contexts
            
        Returns:
            Context recall score (0-1, higher is better)
        """
        try:
            # Extract key information from ground truth
            info_extraction_prompt = f"""
            Extract the key factual information from this ground truth answer:
            {ground_truth}
            
            Return the key facts as a numbered list.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": info_extraction_prompt}],
                temperature=0.1
            )
            
            key_facts = [fact.strip() for fact in response.choices[0].message.content.split('\n') 
                        if fact.strip() and fact[0].isdigit()]
            
            if not key_facts:
                return 0.0
            
            # Check how many key facts are supported by contexts
            context_text = "\n".join(contexts)
            supported_facts = 0
            
            for fact in key_facts:
                support_prompt = f"""
                Context: {context_text}
                Fact: {fact}
                
                Is this fact supported by the given context? Answer only 'YES' or 'NO'.
                """
                
                support_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": support_prompt}],
                    temperature=0.1
                )
                
                if "YES" in support_response.choices[0].message.content.upper():
                    supported_facts += 1
            
            return supported_facts / len(key_facts)
            
        except Exception as e:
            logger.error(f"Error in context recall evaluation: {e}")
            return 0.0
    
    async def evaluate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate overall relevancy of contexts to the question
        
        Args:
            question: Original question
            contexts: List of retrieved contexts
            
        Returns:
            Context relevancy score (0-1, higher is better)
        """
        try:
            relevancy_scores = []
            
            for context in contexts:
                relevancy_prompt = f"""
                Question: {question}
                Context: {context}
                
                Rate the relevancy of this context to the question on a scale of 1-10:
                1 = Completely irrelevant
                10 = Perfectly relevant
                
                Return only the number.
                """
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": relevancy_prompt}],
                    temperature=0.1
                )
                
                try:
                    score = float(response.choices[0].message.content.strip()) / 10.0
                    relevancy_scores.append(score)
                except ValueError:
                    relevancy_scores.append(0.0)
            
            return float(np.mean(relevancy_scores)) if relevancy_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in context relevancy evaluation: {e}")
            return 0.0
    
    def evaluate_answer_similarity(self, ground_truth: str, answer: str) -> float:
        """
        Evaluate semantic similarity between ground truth and generated answer
        
        Args:
            ground_truth: Expected correct answer
            answer: Generated answer
            
        Returns:
            Answer similarity score (0-1, higher is better)
        """
        try:
            # Use TF-IDF for semantic similarity
            texts = [ground_truth, answer]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error in answer similarity evaluation: {e}")
            return 0.0
    
    async def evaluate_answer_correctness(self, question: str, ground_truth: str, answer: str) -> float:
        """
        Evaluate factual correctness of the answer
        
        Args:
            question: Original question
            ground_truth: Expected correct answer
            answer: Generated answer
            
        Returns:
            Answer correctness score (0-1, higher is better)
        """
        try:
            correctness_prompt = f"""
            Question: {question}
            Ground Truth Answer: {ground_truth}
            Generated Answer: {answer}
            
            Rate the factual correctness of the generated answer compared to the ground truth on a scale of 1-10:
            1 = Completely incorrect
            10 = Perfectly correct
            
            Consider:
            - Factual accuracy
            - Completeness
            - Absence of hallucinations
            
            Return only the number.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": correctness_prompt}],
                temperature=0.1
            )
            
            try:
                score = float(response.choices[0].message.content.strip()) / 10.0
                return score
            except ValueError:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in answer correctness evaluation: {e}")
            return 0.0
    
    async def evaluate_single_example(self, example: EvaluationDataset) -> RAGASMetrics:
        """
        Evaluate a single example across all RAGAS metrics
        
        Args:
            example: Evaluation dataset example
            
        Returns:
            RAGASMetrics containing all computed scores
        """
        logger.info(f"Evaluating example: {example.question[:50]}...")
        
        # Run all evaluations concurrently for efficiency
        faithfulness_task = self.evaluate_faithfulness(example.answer, example.contexts)
        answer_relevancy_task = self.evaluate_answer_relevancy(example.question, example.answer)
        context_precision_task = self.evaluate_context_precision(example.question, example.contexts)
        context_recall_task = self.evaluate_context_recall(example.ground_truth, example.contexts)
        context_relevancy_task = self.evaluate_context_relevancy(example.question, example.contexts)
        answer_correctness_task = self.evaluate_answer_correctness(
            example.question, example.ground_truth, example.answer
        )
        
        # Await all async evaluations
        faithfulness = await faithfulness_task
        answer_relevancy = await answer_relevancy_task
        context_precision = await context_precision_task
        context_recall = await context_recall_task
        context_relevancy = await context_relevancy_task
        answer_correctness = await answer_correctness_task
        
        # Compute answer similarity (synchronous)
        answer_similarity = self.evaluate_answer_similarity(example.ground_truth, example.answer)
        
        # Calculate overall score (weighted average)
        overall_score = (
            faithfulness * 0.2 +
            answer_relevancy * 0.15 +
            context_precision * 0.15 +
            context_recall * 0.15 +
            context_relevancy * 0.1 +
            answer_similarity * 0.1 +
            answer_correctness * 0.15
        )
        
        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            context_relevancy=context_relevancy,
            answer_similarity=answer_similarity,
            answer_correctness=answer_correctness,
            overall_score=overall_score
        )
    
    async def evaluate_dataset(self, dataset: List[EvaluationDataset]) -> Dict[str, Any]:
        """
        Evaluate entire dataset and compute aggregate statistics
        
        Args:
            dataset: List of evaluation examples
            
        Returns:
            Dictionary containing detailed evaluation results
        """
        logger.info(f"Starting evaluation of {len(dataset)} examples...")
        
        results = []
        for i, example in enumerate(dataset):
            logger.info(f"Processing example {i+1}/{len(dataset)}")
            metrics = await self.evaluate_single_example(example)
            results.append({
                'question': example.question,
                'ground_truth': example.ground_truth,
                'answer': example.answer,
                'contexts_count': len(example.contexts),
                'faithfulness': metrics.faithfulness,
                'answer_relevancy': metrics.answer_relevancy,
                'context_precision': metrics.context_precision,
                'context_recall': metrics.context_recall,
                'context_relevancy': metrics.context_relevancy,
                'answer_similarity': metrics.answer_similarity,
                'answer_correctness': metrics.answer_correctness,
                'overall_score': metrics.overall_score,
                'metadata': example.metadata
            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Compute aggregate statistics
        aggregate_stats = {
            'total_examples': len(dataset),
            'evaluation_date': datetime.now().isoformat(),
            'mean_scores': {
                'faithfulness': df['faithfulness'].mean(),
                'answer_relevancy': df['answer_relevancy'].mean(),
                'context_precision': df['context_precision'].mean(),
                'context_recall': df['context_recall'].mean(),
                'context_relevancy': df['context_relevancy'].mean(),
                'answer_similarity': df['answer_similarity'].mean(),
                'answer_correctness': df['answer_correctness'].mean(),
                'overall_score': df['overall_score'].mean()
            },
            'std_scores': {
                'faithfulness': df['faithfulness'].std(),
                'answer_relevancy': df['answer_relevancy'].std(),
                'context_precision': df['context_precision'].std(),
                'context_recall': df['context_recall'].std(),
                'context_relevancy': df['context_relevancy'].std(),
                'answer_similarity': df['answer_similarity'].std(),
                'answer_correctness': df['answer_correctness'].std(),
                'overall_score': df['overall_score'].std()
            },
            'detailed_results': results
        }
        
        logger.info("Evaluation completed successfully!")
        return aggregate_stats

# Utility functions for dataset creation and management
def create_evaluation_dataset_from_json(file_path: str) -> List[EvaluationDataset]:
    """Load evaluation dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [
        EvaluationDataset(
            question=item['question'],
            ground_truth=item['ground_truth'],
            contexts=item['contexts'],
            answer=item['answer'],
            metadata=item.get('metadata')
        )
        for item in data
    ]

def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")

def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable evaluation report"""
    mean_scores = results['mean_scores']
    
    report = f"""
RAGAS Evaluation Report
Generated: {results['evaluation_date']}
Total Examples: {results['total_examples']}

=== PERFORMANCE METRICS ===
Overall Score: {mean_scores['overall_score']:.3f}

Faithfulness: {mean_scores['faithfulness']:.3f}
  ↳ How well the answer is grounded in retrieved contexts

Answer Relevancy: {mean_scores['answer_relevancy']:.3f}
  ↳ How relevant the answer is to the question

Context Precision: {mean_scores['context_precision']:.3f}
  ↳ Proportion of relevant contexts in retrieval

Context Recall: {mean_scores['context_recall']:.3f}
  ↳ How well contexts cover the ground truth

Context Relevancy: {mean_scores['context_relevancy']:.3f}
  ↳ Overall relevancy of retrieved contexts

Answer Similarity: {mean_scores['answer_similarity']:.3f}
  ↳ Semantic similarity to ground truth

Answer Correctness: {mean_scores['answer_correctness']:.3f}
  ↳ Factual correctness compared to ground truth

=== PERFORMANCE ANALYSIS ===
"""
    
    # Add performance analysis
    if mean_scores['overall_score'] >= 0.8:
        report += "✅ EXCELLENT: System performing very well across all metrics\n"
    elif mean_scores['overall_score'] >= 0.6:
        report += "⚠️  GOOD: System performing well with room for improvement\n"
    else:
        report += "❌ NEEDS IMPROVEMENT: System requires significant optimization\n"
    
    # Identify areas for improvement
    report += "\n=== RECOMMENDATIONS ===\n"
    if mean_scores['faithfulness'] < 0.7:
        report += "• Improve answer grounding - reduce hallucinations\n"
    if mean_scores['context_precision'] < 0.7:
        report += "• Enhance retrieval precision - filter irrelevant documents\n"
    if mean_scores['context_recall'] < 0.7:
        report += "• Increase retrieval coverage - retrieve more relevant documents\n"
    if mean_scores['answer_correctness'] < 0.7:
        report += "• Improve answer generation - focus on factual accuracy\n"
    
    return report
