# RAGAS Implementation for PsychMet Chatbot

## Overview

This implementation provides comprehensive evaluation capabilities for the PsychMet Educational Chatbot using RAGAS (Retrieval-Augmented Generation Assessment). RAGAS is a framework specifically designed to evaluate RAG systems across multiple dimensions.

## What is RAGAS?

RAGAS evaluates RAG systems using 7 key metrics:

### 1. **Faithfulness** 
- **Purpose**: Measures how grounded the generated answer is in the retrieved contexts
- **Method**: Breaks down answers into claims and verifies each against retrieved documents
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate hallucinations or unsupported claims

### 2. **Answer Relevancy**
- **Purpose**: Evaluates how relevant the answer is to the original question
- **Method**: Generates questions from the answer and measures semantic similarity to original
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate off-topic or irrelevant responses

### 3. **Context Precision**
- **Purpose**: Measures the proportion of relevant contexts in retrieval
- **Method**: Evaluates each retrieved context for relevance to the question
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate poor retrieval filtering

### 4. **Context Recall**
- **Purpose**: Measures how well retrieved contexts cover the ground truth
- **Method**: Extracts key facts from ground truth and checks coverage in contexts
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate incomplete retrieval coverage

### 5. **Context Relevancy**
- **Purpose**: Evaluates overall relevancy of contexts to the question
- **Method**: Rates each context's relevancy on a 1-10 scale
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate poor context quality

### 6. **Answer Similarity**
- **Purpose**: Measures semantic similarity between generated and ground truth answers
- **Method**: Uses TF-IDF vectorization and cosine similarity
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate semantic divergence from expected answers

### 7. **Answer Correctness**
- **Purpose**: Evaluates factual correctness of the generated answer
- **Method**: LLM-based assessment comparing generated answer to ground truth
- **Range**: 0-1 (higher is better)
- **Interpretation**: Low scores indicate factual errors or inaccuracies

## Implementation Structure

```
src/psychmet_chatbot/
├── evaluation.py          # Core RAGAS implementation
├── ragas_integration.py   # Integration with PsychMet chatbot
└── chatbot.py            # Updated chatbot with evaluation support

scripts/
└── run_ragas_evaluation.py  # CLI tool for running evaluations

examples/
└── ragas_demo.py         # Complete demonstration example
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Run Demo Evaluation

```bash
python examples/ragas_demo.py
```

### 3. Run Full Evaluation

```bash
python scripts/run_ragas_evaluation.py --mode full
```

## Usage Examples

### Basic Evaluation

```python
import asyncio
from psychmet_chatbot.ragas_integration import PsychMetRAGASIntegration
from psychmet_chatbot.chatbot import PsychMetChatbot
from psychmet_chatbot.vector_store import VectorStoreManager
from psychmet_chatbot.document_processor import DocumentProcessor

async def run_evaluation():
    # Initialize chatbot
    doc_processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager(doc_processor)
    vector_store = vector_store_manager.create_or_load_store()
    chatbot = PsychMetChatbot(vector_store)
    
    # Create RAGAS integration
    ragas_integration = PsychMetRAGASIntegration(chatbot)
    
    # Define test data
    questions = ["What is reliability in psychological testing?"]
    ground_truths = ["Reliability refers to the consistency..."]
    
    # Run evaluation
    results = await ragas_integration.evaluate_chatbot_performance(
        questions, ground_truths
    )
    
    print(results['report'])

# Run evaluation
asyncio.run(run_evaluation())
```

### Custom Test Dataset

```python
# Create custom questions file
import json

custom_data = {
    "questions": [
        "What is construct validity?",
        "How do you calculate Cronbach's alpha?"
    ],
    "ground_truths": [
        "Construct validity refers to how well a test measures...",
        "Cronbach's alpha is calculated using the formula..."
    ]
}

with open("custom_questions.json", "w") as f:
    json.dump(custom_data, f, indent=2)

# Run evaluation with custom data
python scripts/run_ragas_evaluation.py --mode custom --questions-file custom_questions.json
```

## CLI Usage

The `run_ragas_evaluation.py` script provides several modes:

### Demo Mode (Default)
```bash
python scripts/run_ragas_evaluation.py --mode demo
```
- Evaluates 3 sample questions
- Quick demonstration of RAGAS capabilities

### Full Mode
```bash
python scripts/run_ragas_evaluation.py --mode full
```
- Evaluates all 10 built-in psychometrics questions
- Comprehensive assessment

### Custom Mode
```bash
python scripts/run_ragas_evaluation.py --mode custom --questions-file my_questions.json
```
- Evaluates custom questions from JSON file
- Flexible for domain-specific testing

### Sample Generation
```bash
python scripts/run_ragas_evaluation.py --mode sample
```
- Creates `sample_evaluation_questions.json`
- Template for custom question creation

## Output Files

RAGAS evaluation generates three types of files:

### 1. Dataset File (`evaluation_dataset_*.json`)
```json
{
  "question": "What is reliability?",
  "ground_truth": "Reliability refers to...",
  "contexts": ["Context 1", "Context 2"],
  "answer": "Generated answer",
  "metadata": {"timestamp": "..."}
}
```

### 2. Results File (`ragas_results_*.json`)
```json
{
  "total_examples": 10,
  "evaluation_date": "2025-09-06T...",
  "mean_scores": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.82,
    "context_recall": 0.71,
    "context_relevancy": 0.79,
    "answer_similarity": 0.73,
    "answer_correctness": 0.80,
    "overall_score": 0.78
  },
  "detailed_results": [...]
}
```

### 3. Report File (`evaluation_report_*.txt`)
```
RAGAS Evaluation Report
Generated: 2025-09-06T...
Total Examples: 10

=== PERFORMANCE METRICS ===
Overall Score: 0.780

Faithfulness: 0.850
Answer Relevancy: 0.780
Context Precision: 0.820
...

=== RECOMMENDATIONS ===
• Improve answer grounding - reduce hallucinations
• Enhance retrieval precision - filter irrelevant documents
```

## Score Interpretation

| Score Range | Performance Level | Action Required |
|-------------|-------------------|-----------------|
| 0.8 - 1.0   | Excellent        | Maintain quality |
| 0.6 - 0.8   | Good             | Minor optimizations |
| 0.4 - 0.6   | Average          | Moderate improvements |
| 0.0 - 0.4   | Poor             | Major overhaul needed |

## Troubleshooting Common Issues

### Low Faithfulness Scores
- **Cause**: Answer contains unsupported claims
- **Solution**: Improve prompt engineering, add context verification

### Low Context Precision
- **Cause**: Retrieval returns irrelevant documents
- **Solution**: Improve embedding quality, adjust similarity thresholds

### Low Context Recall
- **Cause**: Missing relevant information in retrieval
- **Solution**: Increase retrieval count, improve document preprocessing

### Low Answer Correctness
- **Cause**: Factual errors in generated answers
- **Solution**: Better model prompting, fact verification steps

## Advanced Configuration

### Custom RAGAS Metrics
```python
from psychmet_chatbot.evaluation import RAGASEvaluator

# Initialize with custom model
evaluator = RAGASEvaluator(
    openai_api_key="your-key",
    model="gpt-4-turbo"  # Use different model
)

# Custom metric weights
overall_score = (
    faithfulness * 0.3 +      # Increase faithfulness weight
    answer_relevancy * 0.2 +
    context_precision * 0.1 + # Decrease precision weight
    context_recall * 0.2 +
    context_relevancy * 0.1 +
    answer_similarity * 0.05 +
    answer_correctness * 0.05
)
```

### Async Evaluation
```python
import asyncio

async def evaluate_multiple_systems():
    # Evaluate multiple chatbot configurations
    configurations = [
        {"temperature": 0.1, "retrieval_k": 3},
        {"temperature": 0.3, "retrieval_k": 5},
        {"temperature": 0.5, "retrieval_k": 7}
    ]
    
    results = []
    for config in configurations:
        # Initialize chatbot with config
        chatbot = PsychMetChatbot(config)
        ragas = PsychMetRAGASIntegration(chatbot)
        
        # Run evaluation
        result = await ragas.evaluate_chatbot_performance(questions, ground_truths)
        results.append((config, result))
    
    return results
```

## Integration with CI/CD

```yaml
# .github/workflows/ragas-evaluation.yml
name: RAGAS Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: pip install -e .
    - name: Run RAGAS evaluation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: python scripts/run_ragas_evaluation.py --mode full
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: ragas-results
        path: evaluation_results/
```

## Best Practices

1. **Regular Evaluation**: Run RAGAS evaluation after significant changes
2. **Baseline Tracking**: Maintain baseline scores for comparison
3. **Metric Focus**: Prioritize metrics most relevant to your use case
4. **Dataset Quality**: Ensure high-quality ground truth answers
5. **Iterative Improvement**: Use results to guide system improvements

## Contributing

To contribute to the RAGAS implementation:

1. Fork the repository
2. Create a feature branch
3. Add tests for new metrics
4. Ensure all evaluations pass
5. Submit a pull request

## License

This RAGAS implementation is part of the PsychMet Chatbot project and is licensed under the MIT License.
