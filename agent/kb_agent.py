"""AWS Strands Knowledge Base Agent for retrieving information from Bedrock Knowledge Bases."""

import json
import re
import os
from typing import Any, Dict, List, Optional
import boto3
import logging
from datetime import datetime

# Import Strands SDK
from strands import Agent
from strands.models.bedrock import BedrockModel

# Define constants
MAX_ROWS_FOR_RETRIVAL = 5

def log_info(logger: logging.Logger, function: str, content: str) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {function}: {content[:200]}..." if len(content) > 200 else f"[{timestamp}] {function}: {content}")


class KnowledgeBaseRetriever:
    """Retrieves information from AWS Bedrock Knowledge Base."""
    
    def __init__(self, kb_id: str, region: str, model_arn: str):
        self.kb_id = kb_id
        self.model_arn = model_arn
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)
    
    def retrieve_and_generate(self, query: str, max_results: int = MAX_ROWS_FOR_RETRIVAL) -> Dict[str, Any]:
        """Retrieve documents and generate response using Bedrock without streaming.."""
        
        try:
            response = self.client.retrieve_and_generate(
                input={
                    'text': query
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.kb_id,
                        'modelArn': self.model_arn,
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': max_results
                            }
                        }
                    }
                }
            )
            
            # Extract generated response
            output = response.get('output', {}).get('text', '')
            
            # Extract citations
            citations = []
            for citation in response.get('citations', []):
                for reference in citation.get('retrievedReferences', []):
                    citation_data = {
                        'content': reference.get('content', {}).get('text', ''),
                        'location': reference.get('location', {}),
                        'metadata': reference.get('metadata', {})
                    }
                    
                    # Extract source information
                    location = reference.get('location', {})
                    if location.get('type') == 'S3':
                        s3_location = location.get('s3Location', {})
                        citation_data['source'] = {
                            'type': 'S3',
                            'uri': s3_location.get('uri', ''),
                            'bucket': s3_location.get('uri', '').split('/')[2] if s3_location.get('uri') else None,
                            'key': '/'.join(s3_location.get('uri', '').split('/')[3:]) if s3_location.get('uri') else None
                        }
                    elif location.get('type') == 'WEB':
                        citation_data['source'] = {
                            'type': 'WEB',
                            'url': location.get('webLocation', {}).get('url', '')
                        }
                    else:
                        citation_data['source'] = {
                            'type': 'UNKNOWN',
                            'raw': location
                        }
                    
                    citations.append(citation_data)
            
            return {
                "success": True,
                "response": output,
                "citations": citations,
                "citation_count": len(citations),
                "session_id": response.get('sessionId')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "citations": [],
                "citation_count": 0
            }


class KnowledgeBaseAgent:
    """Main Knowledge Base Agent using retrieve_and_generate function with Bedrock."""
    
    def __init__(self, logger: logging.Logger, kb_id: str, aws_region: str, model_arn: str):
        
        # Get AWS region & other config value from environment or parameter
        self.kb_id = kb_id or os.getenv("KB_ID")
        self.region = aws_region or os.getenv("AWS_REGION")
        self.model_arn = model_arn or os.getenv("BEDROCK_MODEL_ARN")
        self.logger = logger

        log_info(logger, "KnowledgeBaseAgent.Init", f"Starting function, Kb Id:{kb_id}, Region:{aws_region}, Model ARN:{model_arn}")
        
        # Initialize Knowledge Base retriever
        self.retriever = KnowledgeBaseRetriever(self.kb_id, self.region, self.model_arn)
        log_info(logger, "KnowledgeBaseAgent.Init", f"Ending function, KnowledgeBaseRetriever initialized, Kb Id:{kb_id}")
            
    def process_query(self, user_query: str, max_results: int = MAX_ROWS_FOR_RETRIVAL) -> Dict[str, Any]:
        """
        Query the knowledge base and return results with sources.
        
        Args:
            user_query: User's question
            max_results: Maximum number of documents to retrieve
        
        Returns:
            Dictionary with response, sources, and metadata
        """

        log_info(self.logger, "KnowledgeBaseAgent.process_query", f"Starting function, User Query:{user_query}")
        
        # Use Bedrock's built-in RAG (Retrieve and Generate)
        result = self.retriever.retrieve_and_generate(user_query, max_results)

        if not result["success"]:
            log_info(self.logger, "KnowledgeBaseAgent.process_query", f"After calling retrieve_and_generate, No results, Error Info:{result.get("error")}")
            raise Exception(f"Failed to query knowledge base for user query:{user_query}. Error Info: {result.get("error")}")
        
        # Format sources
        sources = []
        for citation in result["citations"]:
            source = {
                "content_preview": citation["content"][:1000] + "..." if len(citation["content"]) > 1000 else citation["content"],
                "source_type": citation["source"]["type"],
                "metadata": citation.get("metadata", {})
            }
            
            if citation["source"]["type"] == "S3":
                source["uri"] = citation["source"]["uri"]
                source["bucket"] = citation["source"]["bucket"]
                source["key"] = citation["source"]["key"]
                source["link"] = f"s3://{citation['source']['bucket']}/{citation['source']['key']}"
            elif citation["source"]["type"] == "WEB":
                source["url"] = citation["source"]["url"]
                source["link"] = citation["source"]["url"]
            else:
                source["link"] = "Unknown source"
            
            sources.append(source)
        
        log_info(self.logger, "KnowledgeBaseAgent.process_query", f"Ending function, Source Count:{len(sources)}, Response:{result["response"]}")

        return {
            "type": "unstructured",
            "status": "success",
            "response": result["response"],
            "sources": sources,
            "source_count": len(sources),
        }
                
    def get_kb_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base."""
        try:
            client = boto3.client('bedrock-agent', region_name=self.region)
            response = client.get_knowledge_base(knowledgeBaseId=self.kb_id)
            
            kb = response.get('knowledgeBase', {})
            return {
                "success": True,
                "kb_id": kb.get('knowledgeBaseId'),
                "name": kb.get('name'),
                "description": kb.get('description'),
                "status": kb.get('status'),
                "created_at": str(kb.get('createdAt')),
                "updated_at": str(kb.get('updatedAt'))
            }
        except Exception as e:
            raise Exception(f"Failed to query knowledge base. Error Info: {str(e)}")

    def get_sample_questions(self) -> List[str]:
        """Get a list of sample questions that can be answered from the configured knowledgebase."""

        log_info(self.logger, "KnowledgeBaseAgent.get_sample_questions", f"Starting function")

        # Use agent to generate the sample questions
        product_list = "1) Running shoes 2) Smartphone x 3) Denim Jeans 4) Winter Jacket 5) Cotton T-Shirt 6) Bluetooth Headphones"
        prompt = f"""
User Query: Give me a list of questions that the can be answered by the knowledgebase for the following products - {product_list}. Make sure to list 2 questions per product.

Format your response as a JSON array of stringregion = aws_region os (no extra text) like:
```json
[]
```
"""

        # Use Bedrock's built-in RAG (Retrieve and Generate)
        result = self.retriever.retrieve_and_generate(prompt)
        
        if not result["success"]:
            log_info(self.logger, "KnowledgeBaseAgent.get_sample_questions", f"After calling retrieve_and_generate, No results, Failed to query knowledge base. Error Info:{result.get("error")}")
            raise Exception(f"Failed to query knowledge base for sample questions. Error Info: {result.get("error")}")

        # Extract list of possible questions from response
        response_text = result["response"]
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        json_list = json_match.group(1).strip()
        question_list = json.loads(json_list)

        log_info(self.logger, "KnowledgeBaseAgent.get_sample_questions", f"Ending function, Question List:{str(question_list)}")

        return question_list
