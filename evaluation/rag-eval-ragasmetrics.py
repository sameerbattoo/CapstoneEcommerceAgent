"""
RAG Evaluation Script using RAGAS Framework
Evaluates Knowledge Base retrieval and generation quality using multiple metrics
"""

import json
import boto3
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_aws import AmazonKnowledgeBasesRetriever
import pandas as pd
from datetime import datetime

# Configuration
KNOWLEDGE_BASE_ID = "H36GPUOAYX"
#LLM_FOR_TEXT_GENERATION = "anthropic.claude-sonnet-4-5-20250929-v1:0"
LLM_FOR_TEXT_GENERATION_ARN = "arn:aws:bedrock:us-west-2:175918693907:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
LLM_FOR_EVALUATION = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
BEDROCK_EMBEDDINGS = "amazon.titan-embed-text-v2:0"
QUESTIONS_FILE = "evaluation/kb_eval_questions_groundtruths.jsonl"
REGION = "us-west-2"

# Initialize Bedrock clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


def load_questions_and_ground_truths(file_path):
    """Load questions and ground truths from JSONL file"""
    questions = []
    ground_truths = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append(data['Question'])
                ground_truths.append(data['GroundTruth'])
    
    print(f"Loaded {len(questions)} questions from {file_path}")
    return questions, ground_truths


def retrieve_and_generate(question, knowledge_base_id, model_id):
    """
    Retrieve context from Knowledge Base and generate answer
    Returns: answer, contexts (list of retrieved text chunks)
    """
    try:
        # Use foundation model ARN format
        model_arn = model_id
        
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': question},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_arn,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 5
                        }
                    }
                }
            }
        )
        
        # Extract answer
        answer = response['output']['text']
        
        # Extract contexts from citations
        contexts = []
        if 'citations' in response:
            for citation in response['citations']:
                if 'retrievedReferences' in citation:
                    for ref in citation['retrievedReferences']:
                        if 'content' in ref and 'text' in ref['content']:
                            contexts.append(ref['content']['text'])
        
        # If no contexts from citations, try to get from retrievalResults
        if not contexts and 'retrievalResults' in response:
            for result in response['retrievalResults']:
                if 'content' in result and 'text' in result['content']:
                    contexts.append(result['content']['text'])
        
        return answer, contexts
    
    except Exception as e:
        print(f"Error in retrieve_and_generate for question '{question}': {str(e)}")
        return "", []


def prepare_evaluation_dataset(questions, ground_truths, knowledge_base_id, model_id):
    """
    Prepare dataset for RAGAS evaluation
    Format: question, answer, contexts, ground_truth
    """
    data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }
    
    print("\nGenerating answers and retrieving contexts...")
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        print(f"Processing question {i}/{len(questions)}: {question[:60]}...")
        
        answer, contexts = retrieve_and_generate(question, knowledge_base_id, model_id)
        
        data['question'].append(question)
        data['answer'].append(answer)
        data['contexts'].append(contexts if contexts else ["No context retrieved"])
        data['ground_truth'].append(ground_truth)
    
    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset, llm_for_eval, embeddings):
    """
    Run RAGAS evaluation with multiple metrics
    """
    print("\nRunning RAGAS evaluation...")
    print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    
    # Initialize LangChain models for evaluation
    eval_llm = ChatBedrock(
        model_id=llm_for_eval,
        client=bedrock_runtime,
        model_kwargs={"temperature": 0.0}
    )
    
    eval_embeddings = BedrockEmbeddings(
        model_id=embeddings,
        client=bedrock_runtime
    )
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    
    return result


def save_results(result, dataset):
    """Save evaluation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to DataFrame for better visualization
    df = result.to_pandas()
    
    # Save detailed results
    detailed_file = f"evaluation/ragas_eval_detailed_{timestamp}.csv"
    df.to_csv(detailed_file, index=False)
    print(f"\nDetailed results saved to: {detailed_file}")
    
    # Calculate average metrics from the DataFrame
    metrics = {}
    for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if col in df.columns:
            # Handle NaN values by filtering them out
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                metrics[col] = float(valid_values.mean())
            else:
                metrics[col] = 0.0
    
    # Save summary metrics
    summary = {
        "timestamp": timestamp,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "llm_generation": LLM_FOR_TEXT_GENERATION_ARN,
        "llm_evaluation": LLM_FOR_EVALUATION,
        "embeddings": BEDROCK_EMBEDDINGS,
        "num_questions": len(dataset),
        "metrics": metrics
    }
    
    summary_file = f"evaluation/ragas_eval_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    return df, summary


def print_summary(summary, df):
    """Print evaluation summary to console"""
    print("\n" + "="*70)
    print("RAG EVALUATION SUMMARY")
    print("="*70)
    print(f"Knowledge Base ID: {summary['knowledge_base_id']}")
    print(f"LLM for Generation: {summary['llm_generation']}")
    print(f"LLM for Evaluation: {summary['llm_evaluation']}")
    print(f"Embeddings Model: {summary['embeddings']}")
    print(f"Number of Questions: {summary['num_questions']}")
    print("\n" + "-"*70)
    print("METRICS (0.0 - 1.0, higher is better)")
    print("-"*70)
    for metric, score in summary['metrics'].items():
        print(f"{metric.replace('_', ' ').title():.<50} {score:.4f}")
    print("="*70)
    
    # Print per-question breakdown if columns exist
    print("\nPER-QUESTION SCORES:")
    print("-"*70)
    
    # Get available metric columns
    metric_cols = [col for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'] if col in df.columns]
    
    if metric_cols:
        for idx, row in df.iterrows():
            print(f"\nQuestion {idx+1}:")
            for metric in metric_cols:
                if pd.notna(row[metric]):
                    print(f"  {metric.replace('_', ' ').title()}: {row[metric]:.4f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: N/A")
    else:
        print("No metric columns found in results.")
    
    print("\n" + "="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("RAG EVALUATION USING RAGAS FRAMEWORK")
    print("="*70)
    
    # Load questions and ground truths
    questions, ground_truths = load_questions_and_ground_truths(QUESTIONS_FILE)
    
    # Prepare evaluation dataset
    dataset = prepare_evaluation_dataset(
        questions,
        ground_truths,
        KNOWLEDGE_BASE_ID,
        LLM_FOR_TEXT_GENERATION_ARN
    )
    
    # Run RAGAS evaluation
    result = run_ragas_evaluation(
        dataset,
        LLM_FOR_EVALUATION,
        BEDROCK_EMBEDDINGS
    )
    
    # Save and display results
    df, summary = save_results(result, dataset)
    print_summary(summary, df)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
