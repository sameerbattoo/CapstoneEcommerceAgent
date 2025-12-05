# Standard library imports
import logging
from typing import Dict, Any, List, Optional
import time
import json
from datetime import datetime
import os
import base64

import pandas as pd
import uuid
import jwt  # PyJWT

from strands import Agent, tool
from strands_tools import current_time
from strands.models import BedrockModel

# Import the 2 sub agents
from agent import sql_agent
from agent import kb_agent

# Import dotenv for loading environment variables
from dotenv import load_dotenv
import asyncio

# ADDED: BEDROCK_AGENTCORE IMPORT
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager
)

from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client, StreamableHTTPTransport
from botocore.session import Session
from botocore.credentials import Credentials
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import httpx
from typing import Generator
from datetime import timedelta

AWS_SECRET_NAME = "capstone-ecommerce-agent-config" # AWS Secret Manager 

#=====================================================================================
# Configure logging
# For AWS Lambda/AgentCore: StreamHandler writes to stdout, which Lambda captures and sends to CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Lambda automatically sends this to CloudWatch Logs
    ]
)
logger = logging.getLogger("eCommerceAgent")


#=====================================================================================
# Helper class for - HTTPX Auth class that signs requests with AWS SigV4.
class SigV4HTTPXAuth(httpx.Auth):
    """HTTPX Auth class that signs requests with AWS SigV4."""

    def __init__(
        self,
        credentials: Credentials,
        service: str,
        region: str,
    ):
        self.credentials = credentials
        self.service = service
        self.region = region
        self.signer = SigV4Auth(credentials, service, region)

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Signs the request with SigV4 and adds the signature to the request headers."""

        headers = dict(request.headers)
        headers.pop("connection", None)

        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=headers,
        )

        self.signer.add_auth(aws_request)

        request.headers.update(dict(aws_request.headers))

        yield request


#=====================================================================================
# Function to get the list of all tools from a MCP Server
def get_full_tools_list(client):
    """Get all tools with pagination support"""
    more_tools = True
    tools = []
    pagination_token = None
    while more_tools:
        tmp_tools = client.list_tools_sync(pagination_token=pagination_token)
        tools.extend(tmp_tools)
        if tmp_tools.pagination_token is None:
            more_tools = False
        else:
            more_tools = True
            pagination_token = tmp_tools.pagination_token

    return tools

#=====================================================================================
# Generic Function to validate environment variables and return the value
def validate_and_set_env_variable(env_variable: str) -> str:
    env_variable_val = os.getenv(env_variable)
    
    if env_variable_val is None:
        logger.error(f"{env_variable} not found in environment variables.")
        raise ValueError(f"{env_variable} environment variable is required")

    return env_variable_val

#=====================================================================================
# Helper function to Load configuration from AWS Secrets Manager
def load_secrets_from_aws() -> bool:
    """Load configuration from AWS Secrets Manager"""
    import boto3
    from botocore.exceptions import ClientError
    
    load_dotenv() # Get the AWS_SECRET_NAME & AWS_REGION from env, override the rest from Secrets Manager
    secret_name = os.getenv("AWS_SECRET_NAME", AWS_SECRET_NAME)
    region_name = os.getenv("AWS_REGION", "us-west-2")  # Fallback to env or default
    
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(get_secret_value_response['SecretString'])
        
        # Set environment variables from secret
        for key, value in secret.items():
            os.environ[key] = str(value)
        
        logger.info(f"Successfully loaded {len(secret)} configuration values from Secrets Manager")
        return True
        
    except ClientError as e:
        logger.error(f"Error loading secrets from AWS Secrets Manager: {str(e)}")
        logger.info("Falling back to .env file")
        return False

#=====================================================================================
def log_conversation(role: str, content: str, tool_calls: Optional[List] = None) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {role}: {content[:100]}..." if len(content) > 100 else f"[{timestamp}] {role}: {content}")
    
    if tool_calls:
        for call in tool_calls:
            logger.info(f"  Tool used: {call['name']} with args: {json.dumps(call['args'])}")

#=====================================================================================
# Initialize the agent
#=====================================================================================
# ADDED: BEDROCK_AGENTCORE APP CREATION
app = BedrockAgentCoreApp()

# Load environment variables from AWS Secrets Manager (with .env fallback)
load_secrets_from_aws()

# Set - AWS Region, Model ID and ARN, KB ID from environment variables
AWS_REGION = validate_and_set_env_variable("AWS_REGION")
BEDROCK_MODEL_ID = validate_and_set_env_variable("BEDROCK_MODEL_ID")
BEDROCK_MODEL_ARN = validate_and_set_env_variable("BEDROCK_MODEL_ARN")
KB_ID = validate_and_set_env_variable("KB_ID")
DB_CONFIG = {
            "host": validate_and_set_env_variable("DB_HOST"),
            "port": validate_and_set_env_variable("DB_PORT"),
            "database": validate_and_set_env_variable("DB_NAME"),
            "user": validate_and_set_env_variable("DB_USER"),
            "password": validate_and_set_env_variable("DB_PASSWORD")
        }

# ADDED: BEDROCK_AGENTCORE Memory
AGENTCORE_MEMORY_ID = validate_and_set_env_variable("AGENTCORE_MEMORY_ID")

# (AgentCore) Gateway URL (this is the gateway which provides acess product reviews via the lambda function)
GATEWAY_URL = os.getenv("GATEWAY_URL")

# Define constants
MAX_ROWS_FOR_SQL_RESULT_DISPLAY = 20
MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY = 5

# Global variables for lazy initialization
_sql_agent_client = None    # for SQL Agent
_kb_agent_client = None     # for KB Agent
_dynamodb_mcp_client = None # for MCP client
_agent_client = None        # for the orchestrator Agent

#=====================================================================================
# Helper functions to enable lazy loading of various tools
def get_sql_agent_client():
    """Lazy initialization of SQL agent client"""
    global _sql_agent_client

    if _sql_agent_client is None:
        logger.info("Initializing SQL agent client...")
        _sql_agent_client = sql_agent.SQLAgent(logger, DB_CONFIG, AWS_REGION, BEDROCK_MODEL_ID)
    
    return _sql_agent_client

def get_kb_agent_client():
    """Lazy initialization of KB agent client"""
    global _kb_agent_client

    if _kb_agent_client is None:
        logger.info("Initializing KB agent client...")
        _kb_agent_client = kb_agent.KnowledgeBaseAgent(logger, KB_ID, AWS_REGION, BEDROCK_MODEL_ARN)
    
    return _kb_agent_client

def initialize_mcp_client():
    """Lazy initialization of MCP client"""
    global _dynamodb_mcp_client

    if _dynamodb_mcp_client is None:
        logger.info("Initializing MCP client...")

        credentials = Session().get_credentials()
        auth = SigV4HTTPXAuth(credentials, "bedrock-agentcore", AWS_REGION)
        transport_factory = lambda: streamablehttp_client(url=GATEWAY_URL, auth=auth)
        _dynamodb_mcp_client = MCPClient(transport_factory)
#=====================================================================================

#=====================================================================================
# Worker Agent A: Data retrieval agent for Structured data sored in RDS
@tool
def get_answers_for_structured_data(user_query: str) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products and orders transactional data stored in the RDS Postgres database.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """

    log_conversation("User", f"user_query: {user_query}")

    try:
        start_time = time.time()
        # Call the agent and return its response
        response = get_sql_agent_client().process_query(user_query)
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the structured assistant")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the structured assistant: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the structured assistant: {str(e)}. Please try again later."}}
#=====================================================================================

#=====================================================================================
# Worker Agent B: Data retrieval agent for Un-Structured data sored in Bedrock KB
@tool
def get_answers_for_unstructured_data(user_query: str) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products specifications based on the unstructured data from the Bedrock knowledge-base.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """

    log_conversation("User", user_query)

    try:
        start_time = time.time()
        # Call the agent and return its response
        response = get_kb_agent_client().process_query(user_query)
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the unstructured assistant")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the unstructured assistant: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the unstructured assistant: {str(e)}. Please try again later."}}
#=====================================================================================

#=====================================================================================
@tool
def get_default_questions(user_query: str) -> Dict[str, Any]:
    """
    Lists sample of questions that the user can ask related to products and orders. These questions can be based on the un-strutured data from the Bedrock knowledge-base or based on the structured data from the RDS Postgres database.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """

    log_conversation("User", user_query)
    
    try:
        
        start_time = time.time()

        response = (
            "Questions related to Order and Products transactional data in natural language \n"
            "Examples:\n"
        )

        # Call the agent and return its response
        for question in get_sql_agent_client().get_sample_questions():
            response += f"  • {question}\n"

        response += (
            "\n\n, Questions related to Products based on the product specifications \n"
            "Examples:\n"
        )
        for question in get_kb_agent_client().get_sample_questions():
            response += f"  • {question}\n"
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the structured assistant")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the assistant: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the assistant: {str(e)}. Please try again later."}}
#=====================================================================================


#=====================================================================================
def initialize_agent_client(memory_config: AgentCoreMemoryConfig):
    """Lazy initialization of orch agent client"""
    global _agent_client
    global _dynamodb_mcp_client

    # System prompt for the eCommerce orchestrator agent
    AGENT_SYSTEM_PROMPT = f"""
    You are an intelligent E-commerce Assistant Agent designed to help users with queries related to products, orders and reviews. 

    You are a orchestrator agent which has access to the following tools:
    - get_default_questions: for listing the sample questions that the user can ask related to products and orders.
    - get_answers_for_structured_data: for any queries related to Transactional Data, this tool already has the capability to generate SQL based on user query and execute and get the formatted results.
    - get_answers_for_unstructured_data: for any query related to Product Knowledge Base, this toll already has the capability to query the knowledgebase based on user query and get the formatted results.
    - ProductReviewLambda___get_product_reviews: for any queries related to product reviews and sentiments.
    - current_time: for current time infomation.

    Your responsibilities:
    - Categorize the user query and find out which of the above listed tools can answer the question.
    - If the user question is unrelated then gracefully let the user with reasons of which the question can be answered.
    - Else, call the specific tool 
    - Remember the tools produce formatted results, please convert it to markdown so that it can be displayed on the frontend.
    - When the tool - get_answers_for_structured_data is called,
        1) Show the Tabular data first, in a nicely formated tabular grid. Merge the column values while displaying if the data is grouped by a column. Show a max of {MAX_ROWS_FOR_SQL_RESULT_DISPLAY} rows in the table and show the row_count.
        2) Based on the tabular data, if possible, show a graphical representation of the data. 
        3) Then Show the Key Insights, and please make it brief and concise
        4) Then if there is a SQL statement, show the generate SQL at the end within <details> tag.
    - When the tool - get_answers_for_unstructured_data is called, show the sources and the preview at the end within <details> tag.
    - When the tool - ProductReviewLambda___get_product_reviews is called, 
        1) pass the related product_id(s), customer_id(s), product_name(s), customer_name(s), top_rows and review_date range based on the user query. 
        2) Show a max of {MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY} rows in the table and show the total row count.
        3) Please collect all these tool calls and display them at the end within <details> tag.

    Remember previous context from the conversation when responding.

    IMPORTANT: When thinking about how to answer the users question or using tools, please:
    1. Provide text output to the user, this helps users understand your reasoning process while waiting.
    2. Then provide your final answer
    3. When providing the finaly answer keep the summary / insights crisp and short.
    """

    if _agent_client is None:
        logger.info("Initializing Orch agent client...")

        #Load the product review gateway tool if possible
        gateway_tools = []
        initialize_mcp_client()
        
        # Start MCP client session - keep it running for the agent lifecycle
        _dynamodb_mcp_client.__enter__()
        
        try:
            gateway_tools = get_full_tools_list(_dynamodb_mcp_client)
            logger.info(f"Successfully loaded {len(gateway_tools)} tools from Gateway.")
        except Exception as e:
            logger.error(f"Error loading tools from Gateway: {str(e)}")
            pass # Proceed without gateway tools if there's an error

        # Create an Strands agent with our defined tools
        _all_tools = [get_default_questions, get_answers_for_structured_data, get_answers_for_unstructured_data, current_time]
        _all_tools += gateway_tools # Add the gateway tools

        # Now define the Bedrock Model (we can include the reasoning flag here)
        bedrock_model = BedrockModel(
            model_id=BEDROCK_MODEL_ID,
            additional_request_fields={
                "thinking": {          # Bedrock rationales / reasoning config
                    "type": "enabled", # or model-specific enum
                    "budget_tokens": 4096,  # pick a value within model limits
                }
            },
        )

        # Now define the Strands Agent with the model, tools and system prompt from above
        _agent_client = Agent(
            model=bedrock_model,
            tools=_all_tools,
            session_manager=AgentCoreMemorySessionManager(memory_config, AWS_REGION), # Reintroduced session_manager
            system_prompt=AGENT_SYSTEM_PROMPT
        )


#=====================================================================================
async def process_user_input(user_query: str, user_email: str) -> str:
    """
    Creates a Strands agent that answers questions about order and products for a eCommerce solution
    using the 2 tools, one for Structured data and the second one for Unstructed data
    There is another additional tool, which help users by showing the sample questions
    
    Args:
        question: The customer's question or request
        
    Returns:
        The agent's response
    """

    global _agent_client
    log_conversation("User", f"user_query: {user_query}, user_email: {user_email}")

    try:

        start_time = time.time()

        # If the user_email has been passed, then add that filter context to the prompt
        if user_email:
            user_query += f". Also if the above query results in a call to the get_answers_for_structured_data tool and the query mentions my orders or my products or my shipments, if possible, add a WHERE clause to filter the data by customers.email='{user_email}'"

        # Process the question and stream the response back
        async for event in _agent_client.stream_async(user_query):
            if "data" in event:
                yield event["data"]

            #Yield tool use events for UI display
            elif "current_tool_use" in event:
                tool_use = event["current_tool_use"]
                if tool_use.get("name"): # Only yield when tool name is aailable
                    tool_info = f"Tool: {tool_use['name']}\nInput: {tool_use.get('input',{})}"
                    yield f"\n[TOOL USE]{tool_info}\n"

            #. Yield reasining/Thinking events for UI
            elif "reasoning" in event and "reasoningText" in event:
                yield f"\n[THINKING]{event['reasoningText']}\n"

        end_time = time.time()
        
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Yield a graceful error response
        yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."

#=====================================================================================

# ADDED: BEDROCK_AGENTCORE - APP ENTRYPOINT DECLARATION
@app.entrypoint
async def main(payload, context):

    logger.info("Starting eCommerce Agent")
    logger.info(f"Received payload: {payload}")
    logger.info(f"Is payload string? {isinstance(payload, str)}")
    
    try:
        # Extract the user input from the payload
        logger.info(f"Input Payload: {payload}")
        user_input = payload.get("user_input")
        
        # Add explicit check
        if "user_input" not in payload or not user_input:
            logger.error("No 'user_input' key found in payload, using default")
            yield "No user_input provided in payload."
            return
              
        headers = context.request_headers
                
        # Get the session Id and Actor Id
        actor_id = headers.get('X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id', 'user') if headers else 'user'
        session_id = None
        if headers and 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id' in headers:
            session_id = headers.get('X-Amzn-Bedrock-AgentCore-Runtime-Session-Id')
        elif payload and 'session_id' in payload:
            session_id = payload.get('session_id')
        else:
            session_id = str(uuid.uuid4())

        # Get the calling user's email address
        # 1. Get the bearer token from headers
        auth_header = None
        if headers and 'Authorization' in headers:
            auth_header = headers.get("Authorization")

        if not auth_header:
            # treat as anonymous or raise
            user_email = None
        else:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else auth_header
            # 2. Decode JWT without verifying (or with verification if you load JWKS)
            claims = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
            # 3. Extract email or a stable user id depending on your IdP
            user_email = claims.get("username") + "@email.com"  # may be None if you chose not to put PII in token
            # 4. Set the Actor Id to the user EMail address, remember the actor must satisfy regular expression pattern: [a-zA-Z0-9][a-zA-Z0-9-_/]*(?::[a-zA-Z0-9-_/]+)*[a-zA-Z0-9-_/]*
            actor_id = user_email.replace(".","_").replace("@","-")

        logger.info(f"Extracted user_input: {user_input}, session_id: {session_id}, actor_id: {actor_id}, user_email: {user_email}")
        logger.info("\n🚀 Processing request...")

        # Configure the Agentcore MemoryConfig
        memory_config = AgentCoreMemoryConfig(
            memory_id=AGENTCORE_MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                f"/users/{actor_id}/facts": RetrievalConfig(top_k=5, relevance_score=0.5),
                f"/users/{actor_id}/preferences": RetrievalConfig(top_k=5, relevance_score=0.5)
            }
        )

        # Use the Agent. Process the request and stream the response
        initialize_agent_client(memory_config)
        async for chunk in process_user_input(user_input, user_email):
            logger.info(f"Streaming chunk: {chunk[:50]}..." if len(str(chunk)) > 50 else f"Streaming chunk: {chunk}")
            yield chunk

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        logger.info(f"\n❌ {error_msg}")
        
        # Yield error response for AgentCore
        yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
        
    finally:
        logger.info("eCommerce Agent request processed")


if __name__ == "__main__":
    app.run()
