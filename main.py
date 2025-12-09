# Standard library imports
import logging
from typing import Dict, Any, List, Optional, Tuple
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
# HELPER CLASSES
#=====================================================================================

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
# MAIN CLASSES
#=====================================================================================

class AgentManager:
    """Manages all agent instances with lazy loading."""
    
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        """
        Initialize the AgentManager.
        
        Args:
            logger: Logger instance
            config: Configuration dictionary with AWS settings, DB config, etc.
        """
        self.logger = logger
        self.config = config
        
        # Agent instances (lazy loaded)
        self._sql_agent = None
        self._kb_agent = None
        self._mcp_client = None
        self._orchestrator_agent = None
        self._mcp_session_active = False
    
    @property
    def sql_agent(self):
        """Lazy initialization of SQL agent."""
        if self._sql_agent is None:
            self.logger.info("Initializing SQL agent client...")
            self._sql_agent = sql_agent.SQLAgent(
                self.logger,
                self.config['db_config'],
                self.config['aws_region'],
                self.config['bedrock_model_id']
            )
        return self._sql_agent
    
    @property
    def kb_agent(self):
        """Lazy initialization of KB agent."""
        if self._kb_agent is None:
            self.logger.info("Initializing KB agent client...")
            self._kb_agent = kb_agent.KnowledgeBaseAgent(
                self.logger,
                self.config['kb_id'],
                self.config['aws_region'],
                self.config['bedrock_model_arn']
            )
        return self._kb_agent
    
    @property
    def mcp_client(self):
        """Lazy initialization of MCP client."""
        if self._mcp_client is None:
            self._initialize_mcp_client()
        return self._mcp_client
    
    def _initialize_mcp_client(self) -> None:
        """Initialize MCP client with SigV4 authentication."""
        if self.config.get('gateway_url') is None:
            self.logger.warning("Gateway URL not configured, MCP client will not be initialized")
            return
        
        self.logger.info("Initializing MCP client...")
        credentials = Session().get_credentials()
        auth = SigV4HTTPXAuth(credentials, "bedrock-agentcore", self.config['aws_region'])
        transport_factory = lambda: streamablehttp_client(url=self.config['gateway_url'], auth=auth)
        self._mcp_client = MCPClient(transport_factory)
    
    def _get_mcp_tools(self) -> List:
        """Get all tools from MCP server with pagination support."""
        if self._mcp_client is None:
            return []
        
        try:
            tools = []
            pagination_token = None
            more_tools = True
            
            while more_tools:
                tmp_tools = self._mcp_client.list_tools_sync(pagination_token=pagination_token)
                tools.extend(tmp_tools)
                
                if tmp_tools.pagination_token is None:
                    more_tools = False
                else:
                    pagination_token = tmp_tools.pagination_token
            
            self.logger.info(f"Successfully loaded {len(tools)} tools from Gateway")
            return tools
        except Exception as e:
            self.logger.error(f"Error loading tools from Gateway: {str(e)}")
            return []
    
    def _build_system_prompt(self, max_sql_rows: int, max_dynamo_rows: int) -> str:
        """Build the system prompt for the orchestrator agent."""
        return f"""
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
    1) Show the Tabular data first, in a nicely formated tabular grid. Merge the column values while displaying if the data is grouped by a column. Show a max of {max_sql_rows} rows in the table and show the row_count.
    2) Based on the tabular data, if possible, show a graphical representation of the data. 
    3) Then Show the Key Insights, and please make it brief and concise
    4) Then if there is a SQL statement, show the generate SQL at the end within <details> tag.
- When the tool - get_answers_for_unstructured_data is called, show the sources and the preview at the end within <details> tag.
- When the tool - ProductReviewLambda___get_product_reviews is called, 
    1) pass the related product_id(s), customer_id(s), product_name(s), customer_name(s), top_rows and review_date range based on the user query. 
    2) Show a max of {max_dynamo_rows} rows in the table and show the total row count.
    3) Please collect all these tool calls and display them at the end within <details> tag.

Remember previous context from the conversation when responding.

IMPORTANT: When thinking about how to answer the users question or using tools, please:
1. Provide text output to the user, this helps users understand your reasoning process while waiting.
2. Then provide your final answer
3. When providing the finaly answer keep the summary / insights crisp and short.
"""
    
    def initialize_orchestrator(
        self,
        memory_config: AgentCoreMemoryConfig,
        max_sql_rows: int,
        max_dynamo_rows: int
    ) -> None:
        """
        Initialize the orchestrator agent with tools and configuration.
        
        Args:
            memory_config: AgentCore memory configuration
            max_sql_rows: Maximum rows to display for SQL results
            max_dynamo_rows: Maximum rows to display for DynamoDB results
        """
        if self._orchestrator_agent is not None:
            self.logger.info("Orchestrator agent already initialized")
            return
        
        self.logger.info("Initializing orchestrator agent...")
        
        # Load gateway tools if MCP client is available
        gateway_tools = []
        if self.config.get('gateway_url'):
            # Initialize and start MCP client session
            _ = self.mcp_client  # Trigger lazy loading
            if self._mcp_client:
                self._mcp_client.__enter__()
                self._mcp_session_active = True
                gateway_tools = self._get_mcp_tools()
        
        # Collect all tools
        all_tools = [
            get_default_questions,
            get_answers_for_structured_data,
            get_answers_for_unstructured_data,
            current_time
        ]
        all_tools += gateway_tools
        
        # Build system prompt
        system_prompt = self._build_system_prompt(max_sql_rows, max_dynamo_rows)
        
        # Create Bedrock model with reasoning
        bedrock_model = BedrockModel(
            model_id=self.config['bedrock_model_id'],
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 8000,
                }
            },
        )
        
        # Create the orchestrator agent
        self._orchestrator_agent = Agent(
            model=bedrock_model,
            tools=all_tools,
            session_manager=AgentCoreMemorySessionManager(memory_config, self.config['aws_region']),
            system_prompt=system_prompt
        )
        
        self.logger.info("Orchestrator agent initialized successfully")
    
    @property
    def orchestrator(self) -> Optional[Agent]:
        """Get the orchestrator agent."""
        return self._orchestrator_agent
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._mcp_client and self._mcp_session_active:
            try:
                self._mcp_client.__exit__(None, None, None)
                self._mcp_session_active = False
                self.logger.info("MCP client session closed")
            except Exception as e:
                self.logger.error(f"Error closing MCP client session: {str(e)}")


class ECommerceAgentApplication:
    """Main application orchestrator that handles configuration and request processing."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the ECommerceAgentApplication.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.config = {}
        self.agent_manager = None
        
        # Constants
        self.MAX_ROWS_FOR_SQL_RESULT_DISPLAY = 20
        self.MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY = 5
    
    def load_configuration(self) -> None:
        """Load configuration from AWS Secrets Manager or .env file."""
        self.logger.info("Loading configuration...")
        
        # Try loading from AWS Secrets Manager
        if not self._load_secrets_from_aws():
            self.logger.info("Loading from .env file")
            load_dotenv()
        
        # Validate and set required environment variables
        self.config = {
            'aws_region': self._validate_env_var("AWS_REGION"),
            'bedrock_model_id': self._validate_env_var("BEDROCK_MODEL_ID"),
            'bedrock_model_arn': self._validate_env_var("BEDROCK_MODEL_ARN"),
            'kb_id': self._validate_env_var("KB_ID"),
            'agentcore_memory_id': self._validate_env_var("AGENTCORE_MEMORY_ID"),
            'db_config': {
                "host": self._validate_env_var("DB_HOST"),
                "port": self._validate_env_var("DB_PORT"),
                "database": self._validate_env_var("DB_NAME"),
                "user": self._validate_env_var("DB_USER"),
                "password": self._validate_env_var("DB_PASSWORD")
            },
            'gateway_url': os.getenv("GATEWAY_URL")
        }
        
        # Initialize agent manager
        self.agent_manager = AgentManager(self.logger, self.config)
        
        self.logger.info("Configuration loaded successfully")
    
    def _load_secrets_from_aws(self) -> bool:
        """Load configuration from AWS Secrets Manager."""
        import boto3
        from botocore.exceptions import ClientError
        
        load_dotenv()
        secret_name = os.getenv("AWS_SECRET_NAME", AWS_SECRET_NAME)
        region_name = os.getenv("AWS_REGION", "us-west-2")
        
        try:
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=region_name
            )
            
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)
            secret = json.loads(get_secret_value_response['SecretString'])
            
            for key, value in secret.items():
                os.environ[key] = str(value)
            
            self.logger.info(f"Successfully loaded {len(secret)} configuration values from Secrets Manager")
            return True
            
        except ClientError as e:
            self.logger.error(f"Error loading secrets from AWS Secrets Manager: {str(e)}")
            self.logger.info("Falling back to .env file")
            return False
    
    def _validate_env_var(self, env_variable: str) -> str:
        """Validate and return environment variable value."""
        env_variable_val = os.getenv(env_variable)
        
        if env_variable_val is None:
            self.logger.error(f"{env_variable} not found in environment variables.")
            raise ValueError(f"{env_variable} environment variable is required")
        
        return env_variable_val
    
    def _extract_session_id(self, headers: Optional[Dict], payload: Dict) -> str:
        """Extract or generate session ID."""
        if headers and 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id' in headers:
            return headers.get('X-Amzn-Bedrock-AgentCore-Runtime-Session-Id')
        elif payload and 'session_id' in payload:
            return payload.get('session_id')
        else:
            return str(uuid.uuid4())
    
    def _extract_user_context(self, headers: Optional[Dict]) -> Tuple[Optional[str], str]:
        """
        Extract user email and actor_id from request headers.
        
        Args:
            headers: Request headers containing Authorization token
            
        Returns:
            Tuple of (user_email, actor_id)
        """
        user_email = None
        actor_id = 'user'  # Default actor
        
        if not headers:
            return user_email, actor_id
        
        # Check for custom actor ID header
        if 'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id' in headers:
            actor_id = headers.get('X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id')
        
        # Try to extract from JWT token
        auth_header = headers.get("Authorization")
        if auth_header:
            try:
                token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else auth_header
                claims = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
                
                username = claims.get("username")
                if username:
                    user_email = f"{username}@email.com"
                    actor_id = user_email.replace(".", "_").replace("@", "-")
                    self.logger.info(f"Extracted user context: email={user_email}, actor_id={actor_id}")
            except Exception as e:
                self.logger.warning(f"Failed to parse JWT token: {str(e)}")
        
        return user_email, actor_id
    
    def _build_memory_config(self, session_id: str, actor_id: str) -> AgentCoreMemoryConfig:
        """Build AgentCore memory configuration."""
        return AgentCoreMemoryConfig(
            memory_id=self.config['agentcore_memory_id'],
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                f"/users/{actor_id}/facts": RetrievalConfig(top_k=5, relevance_score=0.5),
                f"/users/{actor_id}/preferences": RetrievalConfig(top_k=5, relevance_score=0.5)
            }
        )
    
    def _add_user_context_filter(self, user_query: str, user_email: Optional[str]) -> str:
        """Add user context filtering to the query."""
        if not user_email:
            return user_query
        
        context_filter = f""". 
Also,
    If the above query results in a call to the get_answers_for_structured_data tool and the query mentions my orders or my products or my shipments, 
    - then, if possible, add a WHERE clause to filter the data by customers.email='{user_email}'.
    If the above query results in a call to the ProductReviewLambda___get_product_reviews tool and the query mentions my reviews,
    - then, filter product review data by passing the current user: {user_email} as customer_id parameter.
"""
        return user_query + context_filter
    
    async def process_and_stream(self, user_query: str, user_email: Optional[str]):
        """
        Process user input and stream responses.
        
        Args:
            user_query: The user's question or request
            user_email: The email address of the current logged-in user
            
        Yields:
            Streaming response chunks, tool use events, and thinking events
        """
        log_conversation("User", f"user_query: {user_query}, user_email: {user_email}")
        
        if not self.agent_manager.orchestrator:
            yield "Agent not initialized. Please try again."
            return
        
        try:
            start_time = time.time()
            
            # Add user context filtering
            enhanced_query = self._add_user_context_filter(user_query, user_email)
            
            # Stream responses from the agent
            async for event in self.agent_manager.orchestrator.stream_async(enhanced_query):
                if "data" in event:
                    yield event["data"]
                
                # Yield tool use events for UI display
                elif "current_tool_use" in event:
                    tool_use = event["current_tool_use"]
                    if tool_use.get("name"):
                        tool_info = f"Tool: {tool_use['name']}\nInput: {tool_use.get('input', {})}"
                        yield f"\n[TOOL USE]{tool_info}\n"
                
                # Yield reasoning/thinking events for UI
                elif "reasoning" in event and "reasoningText" in event:
                    yield f"\n[THINKING]{event['reasoningText']}\n"
            
            end_time = time.time()
            self.logger.info(f"Request processed in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.logger.error(error_msg)
            yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
    
    async def handle_request(self, payload: Dict[str, Any], context: Any):
        """
        Main entry point for handling requests.
        
        Args:
            payload: Request payload containing user_input
            context: Request context with headers
            
        Yields:
            Streaming response chunks
        """
        self.logger.info(f"Received payload: {payload}")
        
        try:
            # Extract user input
            user_input = payload.get("user_input")
            if not user_input:
                self.logger.error("No 'user_input' key found in payload")
                yield "No user_input provided in payload."
                return
            
            # Extract headers
            headers = context.request_headers if hasattr(context, 'request_headers') else None
            
            # Get session and user context
            session_id = self._extract_session_id(headers, payload)
            user_email, actor_id = self._extract_user_context(headers)
            
            self.logger.info(f"Processing - user_input: {user_input}, session_id: {session_id}, actor_id: {actor_id}, user_email: {user_email}")
            self.logger.info("\n🚀 Processing request...")
            
            # Build memory configuration
            memory_config = self._build_memory_config(session_id, actor_id)
            
            # Initialize orchestrator with memory config
            self.agent_manager.initialize_orchestrator(
                memory_config,
                self.MAX_ROWS_FOR_SQL_RESULT_DISPLAY,
                self.MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY
            )
            
            # Process the request and stream responses
            async for chunk in self.process_and_stream(user_input, user_email):
                self.logger.info(f"Streaming chunk: {chunk[:50]}..." if len(str(chunk)) > 50 else f"Streaming chunk: {chunk}")
                yield chunk
        
        except Exception as e:
            error_msg = f"Error in handle_request: {str(e)}"
            self.logger.error(error_msg)
            self.logger.info(f"\n❌ {error_msg}")
            yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
    
    def cleanup(self) -> None:
        """Clean up application resources."""
        self.logger.info("Cleaning up application resources...")
        if self.agent_manager:
            self.agent_manager.cleanup()


#=====================================================================================
# UTILITY FUNCTIONS
#=====================================================================================

def log_conversation(role: str, content: str, tool_calls: Optional[List] = None) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {role}: {content[:100]}..." if len(content) > 100 else f"[{timestamp}] {role}: {content}")
    
    if tool_calls:
        for call in tool_calls:
            logger.info(f"  Tool used: {call['name']} with args: {json.dumps(call['args'])}")


#=====================================================================================
# GLOBAL APPLICATION INSTANCE
#=====================================================================================

# ADDED: BEDROCK_AGENTCORE APP CREATION
app = BedrockAgentCoreApp()

# Global application instance (initialized on first request)
_app_instance: Optional[ECommerceAgentApplication] = None


def get_app_instance() -> ECommerceAgentApplication:
    """Get or create the application instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = ECommerceAgentApplication(logger)
        _app_instance.load_configuration()
    return _app_instance


#=====================================================================================
# TOOL DEFINITIONS
#=====================================================================================

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
        app_instance = get_app_instance()
        response = app_instance.agent_manager.sql_agent.process_query(user_query)
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the structured assistant")
        return response

    except Exception as e:
        logger.error(f"Error processing request in the structured assistant: {str(e)}")
        return {"message": {"content": f"I'm sorry, I encountered an error in the structured assistant: {str(e)}. Please try again later."}}


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
        app_instance = get_app_instance()
        response = app_instance.agent_manager.kb_agent.process_query(user_query)
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the unstructured assistant")
        return response

    except Exception as e:
        logger.error(f"Error processing request in the unstructured assistant: {str(e)}")
        return {"message": {"content": f"I'm sorry, I encountered an error in the unstructured assistant: {str(e)}. Please try again later."}}


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
        app_instance = get_app_instance()

        response = (
            "Questions related to Order and Products transactional data in natural language \n"
            "Examples:\n"
        )

        for question in app_instance.agent_manager.sql_agent.get_sample_questions():
            response += f"  • {question}\n"

        response += (
            "\n\n, Questions related to Products based on the product specifications \n"
            "Examples:\n"
        )
        for question in app_instance.agent_manager.kb_agent.get_sample_questions():
            response += f"  • {question}\n"
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the assistant: {str(e)}")
        return {"message": {"content": f"I'm sorry, I encountered an error in the assistant: {str(e)}. Please try again later."}}


#=====================================================================================
# ENTRY POINT
#=====================================================================================


@app.entrypoint
async def main(payload, context):
    """
    Main entry point for the eCommerce Agent application.
    
    Args:
        payload: Request payload containing user_input
        context: Request context with headers and metadata
        
    Yields:
        Streaming response chunks
    """
    logger.info("Starting eCommerce Agent")
    logger.info(f"Received payload: {payload}")
    logger.info(f"Is payload string? {isinstance(payload, str)}")
    
    try:
        # Get or create application instance
        app_instance = get_app_instance()
        
        # Process the request and stream responses
        async for chunk in app_instance.handle_request(payload, context):
            yield chunk
    
    except Exception as e:
        error_msg = f"Error in main entry point: {str(e)}"
        logger.error(error_msg)
        logger.info(f"\n❌ {error_msg}")
        yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
    
    finally:
        logger.info("eCommerce Agent request processed")


if __name__ == "__main__":
    app.run()
