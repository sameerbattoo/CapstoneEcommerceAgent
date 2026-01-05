"""AWS Strands Knowledge Base Agent for retrieving information from Bedrock Knowledge Bases."""

import json
import re
from typing import Any, Dict, List, Optional
import boto3
import logging
from datetime import datetime
import time

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
    
    def __init__(self, kb_id: str, region: str, model_id: str, logger: logging.Logger):
        self.kb_id = kb_id
        self.region = region
        self.model_id = model_id
        self.logger = logger
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)
    
    def retrieve_and_generate(self, query: str, max_results: int = MAX_ROWS_FOR_RETRIVAL, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve documents and generate response using separate retrieve + converse calls.
        
        This approach uses the lower-level APIs to get accurate token counts:
        1. retrieve() - Get relevant documents from KB
        2. converse() - Generate response with token usage metrics
        
        Args:
            query: The user's query
            max_results: Maximum number of results to retrieve
            tenant_id: Optional tenant ID for filtering results by tenant_access metadata
        """
        
        try:
            start_time = time.time()
            total_input_tokens = 0
            total_output_tokens = 0

            # Step 1: Retrieve relevant documents from Knowledge Base
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

            # Perform the retrive operation first
            retrieve_response = self.client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={'text': query},
                retrievalConfiguration=retrieval_config
            )
            
            # Extract retrieved documents
            retrieved_docs = retrieve_response.get('retrievalResults', [])
            
            if not retrieved_docs:
                log_info(self.logger, "KnowledgeBaseRetriever.retrieve_and_generate", 
                        f"No documents retrieved for query: {query}")
                return {
                    "success": True,
                    "response": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "citations": [],
                    "citation_count": 0,
                    "session_id": None,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            
            # Step 2: Build context from retrieved documents
            context_parts = []
            citations = []
            
            for idx, doc in enumerate(retrieved_docs, 1):
                content = doc.get('content', {}).get('text', '')
                location = doc.get('location', {})
                metadata = doc.get('metadata', {})
                
                # Add to context
                context_parts.append(f"[Document {idx}]\n{content}\n")
                
                # Build citation data
                citation_data = {
                    'content': content,
                    'location': location,
                    'metadata': metadata
                }
                
                # Extract page number from metadata if available
                page_number = metadata.get('x-amz-bedrock-kb-document-page-number')
                if page_number is not None:
                    citation_data['page_number'] = int(page_number) if isinstance(page_number, (int, float)) else page_number
                
                # Extract source information
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
            
            context = "\n".join(context_parts)
            
            # Step 3: Generate response using converse API (which provides token counts)
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)
            
            # Build the prompt with retrieved context
            system_prompt = """You are a helpful assistant that answers questions based on the provided context documents. 
            Use only the information from the documents to answer the question. If the documents don't contain enough information, say so.
            Cite the document numbers,  when referencing specific information."""
            
            user_message = f"""Context from knowledge base:

            {context}

            Question: {query}

            Please answer the question based on the context provided above."""
                       
            converse_response = bedrock_runtime.converse(
                modelId=self.model_id,
                messages=[
                    {
                        'role': 'user',
                        'content': [{'text': user_message}]
                    }
                ],
                system=[{'text': system_prompt}],
                inferenceConfig={
                    'maxTokens': 2048,
                    'temperature': 0.7
                }
            )
            
            # Extract response text
            output = converse_response.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
            
            # Extract token usage from converse response
            usage = converse_response.get('usage', {})
            total_input_tokens = usage.get('inputTokens', 0)
            total_output_tokens = usage.get('outputTokens', 0)
            
            end_time = time.time()
            processing_duration_in_secs = abs(end_time - start_time)

            log_info(self.logger, "KnowledgeBaseRetriever.retrieve_and_generate", 
                    f"Duration_In_Sec: {processing_duration_in_secs:.2f}, Input_Token_Count: {total_input_tokens}, Output_Token_Count: {total_output_tokens}, tenant_id: {tenant_id}, Retrieved_Docs: {len(retrieved_docs)}")

            return {
                "success": True,
                "response": output,
                "citations": citations,
                "citation_count": len(citations),
                "session_id": None,  # No session ID with this approach
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }
            
        except Exception as e:
            error_msg = f"Failed to retrieve and generate from knowledge base. Error: {str(e)}"
            self.logger.error(f"[KnowledgeBaseRetriever] {error_msg}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "citations": [],
                "citation_count": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }


class KnowledgeBaseAgent:
    """Main Knowledge Base Agent using retrieve_and_generate function with Bedrock."""
    
    def __init__(self, logger: logging.Logger, kb_id: str, aws_region: str, model_id: str, tenant_id: str, session_id: str = "unknown", token_callback: Optional[callable] = None):
        
        # Get AWS region & other config value from environment or parameter
        self.kb_id = kb_id 
        self.region = aws_region 
        self.model_id = model_id
        self.tenant_id = tenant_id
        self.session_id = session_id  # Store session_id for metrics
        self.token_callback = token_callback  # Store callback for token accumulation
        self.logger = logger

        log_info(logger, "KnowledgeBaseAgent.Init", f"Starting function, Kb Id:{kb_id}, Region:{aws_region}, Model ID:{model_id}, Tenant ID:{tenant_id}, Session ID:{session_id}")
        
        # Initialize Knowledge Base retriever
        self.retriever = KnowledgeBaseRetriever(self.kb_id, self.region, self.model_id, logger)
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
        
        import time
        start_time = time.time()
        
        try:
            # Use Bedrock's built-in RAG (Retrieve and Generate)
            result = self.retriever.retrieve_and_generate(user_query, max_results, self.tenant_id)

            if not result["success"]:
                log_info(self.logger, "KnowledgeBaseAgent.process_query", f"After calling retrieve_and_generate, No results, Error Info:{result.get('error')}")
                raise Exception(f"Failed to query knowledge base for user query:{user_query}. Error Info: {result.get('error')}")
            
            # Format sources
            sources = []
            for citation in result["citations"]:
                source = {
                    "content_preview": citation["content"][:100] + "..." if len(citation["content"]) > 100 else citation["content"],
                    "source_type": citation["source"]["type"],
                    "metadata": citation.get("metadata", {})
                }
                
                # Add page number if available
                if "page_number" in citation:
                    source["page_number"] = citation["page_number"]
                
                if citation["source"]["type"] == "S3":
                    source["uri"] = citation["source"]["uri"]
                    source["bucket"] = citation["source"]["bucket"]
                    source["key"] = citation["source"]["key"]
                    source["link"] = f"s3://{citation['source']['bucket']}/{citation['source']['key']}"
                    # Add page number to link if available
                    if "page_number" in citation:
                        source["link"] += f" (Page {citation['page_number']})"
                elif citation["source"]["type"] == "WEB":
                    source["url"] = citation["source"]["url"]
                    source["link"] = citation["source"]["url"]
                else:
                    source["link"] = "Unknown source"
                
                sources.append(source)
            
            end_time = time.time()
            
            log_info(self.logger, "KnowledgeBaseAgent.process_query", f"Ending function, Source Count:{len(sources)}, Response:{result['response']}")

            # Emit step metrics for successful retrieval
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="kb_agent_execution",
                start_time=start_time,
                end_time=end_time,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status="success",
                additional_data={
                    "retrieved_docs": len(sources),
                    "max_results": max_results,
                    "kb_id": self.kb_id
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(
                    result.get("input_tokens", 0), 
                    result.get("output_tokens", 0), 
                    "kb_agent_execution"
                )

            return {
                "type": "unstructured",
                "status": "success",
                "response": result["response"],
                "sources": sources,
                "source_count": len(sources),
            }
        
        except Exception as e:
            end_time = time.time()
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="kb_agent_execution",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "error": str(e),
                    "max_results": max_results,
                    "kb_id": self.kb_id
                }
            )
            
            raise
                
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

        start_time = time.time()

        try:
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
                log_info(self.logger, "KnowledgeBaseAgent.get_sample_questions", f"After calling retrieve_and_generate, No results, Failed to query knowledge base. Error Info:{result.get('error')}")
                raise Exception(f"Failed to query knowledge base for sample questions. Error Info: {result.get('error')}")

            # Extract list of possible questions from response
            response_text = result["response"]
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            json_list = json_match.group(1).strip()
            question_list = json.loads(json_list)

            end_time = time.time()

            # Emit step metrics for successful question generation
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="kb_sample_questions",
                start_time=start_time,
                end_time=end_time,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                status="success",
                additional_data={
                    "questions_generated": len(question_list),
                    "kb_id": self.kb_id
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(
                    result.get("input_tokens", 0), 
                    result.get("output_tokens", 0), 
                    "kb_sample_questions"
                )

            log_info(self.logger, "KnowledgeBaseAgent.get_sample_questions", f"Ending function, Question List:{str(question_list)}")

            return question_list

        except Exception as e:
            end_time = time.time()
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="kb_sample_questions",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "error": str(e),
                    "kb_id": self.kb_id
                }
            )
            
            raise
