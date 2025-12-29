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
    logger.info(f"[{timestamp}] {function}: {content[:500]}..." if len(content) > 500 else f"[{timestamp}] {function}: {content}")


class KnowledgeBaseRetriever:
    """Retrieves information from AWS Bedrock Knowledge Base."""
    
    def __init__(self, kb_id: str, region: str, model_arn: str, logger: logging.Logger):
        self.kb_id = kb_id
        self.model_arn = model_arn
        self.logger = logger
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)
    
    def retrieve_and_generate(self, query: str, max_results: int = MAX_ROWS_FOR_RETRIVAL, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve documents and generate response using Bedrock without streaming.
        
        Args:
            query: The user's query
            max_results: Maximum number of results to retrieve
            tenant_id: Optional tenant ID for filtering results by tenant_access metadata
        """
        
        try:
            # Build retrieval configuration
            retrieval_config = {
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
            
            # Add filter for tenant_id
            tenant_id_filter = tenant_id or "N/A"
            retrieval_config['vectorSearchConfiguration']['filter'] = {
                'equals': {
                    'key': 'tenant_access',
                    'value': tenant_id_filter
                }
            }
                
            response = self.client.retrieve_and_generate(
                input={
                    'text': query
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.kb_id,
                        'modelArn': self.model_arn,
                        'retrievalConfiguration': retrieval_config
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
            error_msg = f"Failed to retrieve and generate from knowledge base. Error: {str(e)}"
            self.logger.error(f"[KnowledgeBaseRetriever] {error_msg}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "citations": [],
                "citation_count": 0
            }


class KnowledgeBaseAgent:
    """Main Knowledge Base Agent using retrieve_and_generate function with Bedrock."""
    
    def __init__(self, logger: logging.Logger, kb_id: str, aws_region: str, model_arn: str, tenant_id: str):
        
        # Get AWS region & other config value from environment or parameter
        self.kb_id = kb_id 
        self.region = aws_region 
        self.model_arn = model_arn
        self.tenant_id = tenant_id
        self.logger = logger

        log_info(logger, "KnowledgeBaseAgent.Init", f"Starting function, Kb Id:{kb_id}, Region:{aws_region}, Model ARN:{model_arn}, Tenant ID:{tenant_id}")
        
        # Initialize Knowledge Base retriever
        self.retriever = KnowledgeBaseRetriever(self.kb_id, self.region, self.model_arn, logger)
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

        log_info(self.logger, "KnowledgeBaseAgent.process_query", f"Starting function, User Query:{user_query}, max_results: {max_results}, Tenant ID:{self.tenant_id}")
        
        # Use Bedrock's built-in RAG (Retrieve and Generate)
        result = self.retriever.retrieve_and_generate(user_query, max_results, self.tenant_id)

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

        log_info(self.logger, "KnowledgeBaseAgent.get_sample_questions", f"Starting function, Tenant ID:{self.tenant_id}")

        # Use agent to generate the sample questions
        product_list = "1) Running shoes 2) Smartphone x 3) Denim Jeans 4) Winter Jacket 5) Cotton T-Shirt 6) Bluetooth Headphones"
        prompt = f"""
        <task_description>
        Generate a list of questions that can be answered by the knowledge base for each product in the provided product list.
        </task_description>

        <input>
        <product_list>{product_list}</product_list>
        </input>

        <instructions>
        1. For each product in the provided product list, create 2 relevant questions that could be answered by consulting a knowledge base about that product.
        2. Ensure questions are specific to each product's features, use cases, or common customer inquiries.
        3. Make questions clear, concise, and directly related to information typically found in product knowledge bases.
        4. Generate exactly 2 questions per product, no more and no less.
        </instructions>

        <response_format>
        Return your response as a JSON array of strings containing all the generated questions, with no additional text or explanations.

        Example format:
        ```json
        [
        "What are the technical specifications of Product X?",
        "How do I troubleshoot connectivity issues with Product X?",
        ...
        ]
        ```
        </response_format>

        Provide your JSON array of questions immediately without any preamble or additional explanations.
        """

        # Use Bedrock's built-in RAG (Retrieve and Generate)
        result = self.retriever.retrieve_and_generate(prompt, MAX_ROWS_FOR_RETRIVAL, tenant_id=self.tenant_id)
        
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
