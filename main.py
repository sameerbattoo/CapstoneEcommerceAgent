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
from bedrock_agentcore.memory import MemoryClient
from strands.hooks.events import MessageAddedEvent, AgentInitializedEvent
from strands.hooks.registry import HookProvider
from strands.agent.conversation_manager import SlidingWindowConversationManager

from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client, StreamableHTTPTransport
from botocore.session import Session
from botocore.credentials import Credentials
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import httpx
from typing import Generator
from datetime import timedelta

from strands.types.content import SystemContentBlock

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

class ECommerceMemoryHook(HookProvider):
    """Memory hook for persisting conversations to AgentCore Memory."""
    
    def __init__(
        self,
        memory_client: MemoryClient,
        memory_id: str,
        actor_id: str,
        session_id: str,
        logger: logging.Logger
    ):
        """
        Initialize the memory hook.
        
        Args:
            memory_client: AgentCore memory client
            memory_id: AgentCore memory ID
            actor_id: Actor/user ID
            session_id: Session ID
            logger: Logger instance
        """
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        self.logger = logger
    
    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load recent conversation history when agent starts."""
        try:
            # Load the last 10 conversation turns from memory
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=10,
            )
            
            if recent_turns:
                # Format conversation history for context
                context_messages = []
                for turn in recent_turns:
                    for message in turn:
                        role = "assistant" if message["role"] == "ASSISTANT" else "user"
                        content = message["content"]["text"]
                        context_messages.append(
                            {"role": role, "content": [{"text": content}]}
                        )
                
                # Add context to agent's messages
                event.agent.messages = context_messages
                self.logger.info(f"Loaded {len(context_messages)} messages from memory for session {self.session_id}")
            
            # Retrieve and add user preferences and facts as context
            self._add_context_to_system_prompt(event)
            
        except Exception as e:
            self.logger.error(f"Memory load error: {e}")
    
    def _add_context_to_system_prompt(self, event: AgentInitializedEvent):
        """Add user preferences and facts to system prompt."""
        try:
            # Get user preferences, top 10
            preferences = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=f"/users/{self.actor_id}/preferences",
                query="What does the user prefer? What are their settings and product choices, preferred products?",
                top_k=10
            )
            
            # Get user facts, top 10
            facts = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=f"/users/{self.actor_id}/facts",
                query="What information do we know about the user? User email, location, past purchases, product reviews.",
                top_k=10
            )
            
            # Build context string
            context_parts = []
            
            if preferences:
                prefs_text = "\n".join([p["content"]["text"] for p in preferences])
                context_parts.append(f"User Preferences:\n{prefs_text}")
            
            if facts:
                facts_text = "\n".join([f["content"]["text"] for f in facts])
                context_parts.append(f"User Facts:\n{facts_text}")
            
            if context_parts:
                context = "\n\n".join(context_parts)
                event.agent.system_prompt += f"\n\n<user_context>\n{context}\n</user_context>\n\nNote: Use this context to personalize responses, but do not explicitly mention these preferences or facts unless directly relevant to the user's query."
                self.logger.info(f"Added user context to system prompt for actor {self.actor_id}")
        
        except Exception as e:
            self.logger.warning(f"Could not retrieve user context: {e}")
    
    def on_message_added(self, event: MessageAddedEvent):
        """Store messages in AgentCore Memory."""
        try:
            messages = event.agent.messages
            if not messages:
                return
            
            last_message = messages[-1]
            
            # Only save user and assistant messages
            if last_message["role"] not in ["user", "assistant"]:
                return
            
            # Check if message has text content
            if "content" not in last_message or not last_message["content"]:
                return
            
            if "text" not in last_message["content"][0]:
                return
            
            content = last_message["content"][0]["text"]
            role = last_message["role"].upper() if last_message["role"] == "assistant" else last_message["role"]
            
            # Save conversation turn to AgentCore Memory
            self.memory_client.save_conversation(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(content, role)]
            )
            
            self.logger.info(f"Saved {role} message to memory for session: {self.session_id}, actor_id: {self.actor_id}")
        
        except Exception as e:
            self.logger.error(f"Memory save error: {e}")
    
    def register_hooks(self, registry):
        """Register hook callbacks."""
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)


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
        self._orchestrator_agents = {}  # session_id -> Agent instance
        self._mcp_session_active = False
        
        # Initialize memory client for AgentCore Memory
        self._memory_client = MemoryClient(region_name=config['aws_region'])
    
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
    
    def _build_system_prompt(self, user_email: str, max_sql_rows: int, max_dynamo_rows: int) -> str:
        """Build the system prompt for the orchestrator agent."""
        return f"""
            # E-commerce Assistant Agent

            <role>
            You are an intelligent E-commerce Assistant Agent designed to help users with queries related to products, orders, and reviews. Your primary responsibility is to analyze user queries and utilize the appropriate tools to provide accurate and helpful responses.
            </role>

            <available_tools>
            1. **get_default_questions**
            - Purpose: Lists sample questions users can ask about products and orders.
            - Use when: User is unsure what to ask or needs examples of possible queries.

            2. **get_answers_for_structured_data**
            - Purpose: Handles queries related to Transactional Data.
            - Use when: User asks about orders, shipments, inventory, sales data, or other structured information.
            - Action: Generates SQL, executes it, and returns formatted results.

            3. **get_answers_for_unstructured_data**
            - Purpose: Processes queries related to Product Knowledge Base.
            - Use when: User asks about product specifications, features, compatibility, or usage instructions.
            - Action: Searches the knowledgebase and returns formatted results.

            4. **ProductReviewLambda___get_product_reviews**
            - Purpose: Retrieves information about product reviews and sentiments.
            - Use when: User asks about product ratings, customer feedback, or review analysis.

            5. **current_time**
            - Purpose: Provides current time information.
            - Use when: User needs to know the current time or for time-sensitive operations.
            </available_tools>

            <core_responsibilities>
            1. Analyze each user query to determine which tool is most appropriate.
            2. If the query is outside your scope, politely explain why with specific reasons.
            3. Call the appropriate tool to retrieve the necessary information.
            4. Convert all tool results to markdown format for frontend display.
            5. Maintain context from previous conversation when responding to follow-up queries.
            </core_responsibilities>

            <tool_instructions>
            When using **get_answers_for_structured_data**:
            - For queries about "my orders," "my products," or "my shipments," add a WHERE clause: `customers.email='{user_email}'`
            - Present results in this order:
            1. Display tabular data in a well-formatted markdown grid
            2. Merge column values when data is grouped by a column
            3. Show maximum {max_sql_rows} rows and include the total row count
            4. Include a graphical representation of the data when applicable
            5. Provide brief, concise key insights about the data
            6. Include the generated SQL statement within `<details>` tags at the end

            When using **get_answers_for_unstructured_data**:
            - Display the retrieved information in a clear, structured format
            - Include sources and preview within `<details>` tags at the end

            When using **ProductReviewLambda___get_product_reviews**:
            - Pass relevant parameters based on the query:
            - product_id(s)
            - customer_id(s)
            - product_name(s)
            - customer_name(s)
            - top_rows
            - review_date range
            - For queries about "my reviews," filter by passing {user_email} as customer_id parameter
            - Show maximum {max_dynamo_rows} rows and include the total row count
            - Include all tool calls within `<details>` tags at the end
            </tool_instructions>

            <response_format>
            1. First, provide your reasoning process to show how you analyzed the query and selected the appropriate tool.
            2. Present the requested information in a clear, structured format using markdown.
            3. Include any relevant summaries or insights in a concise manner.
            4. Place technical details (SQL queries, tool calls) in collapsible `<details>` sections.
            </response_format>

            <workflow>
            1. Analyze the user query to understand their intent
            2. Determine which tool is most appropriate for addressing the query
            3. Call the selected tool with the necessary parameters
            4. Format the results in markdown for clear presentation
            5. Add concise insights or summaries if applicable
            6. Include technical details in collapsible sections
            </workflow>

            When responding to a user query, provide your answer immediately without any preamble, focusing only on addressing the user's specific request.
            """
    
    def initialize_orchestrator(
        self,
        session_id: str,
        actor_id: str,
        user_email: str,
        max_sql_rows: int,
        max_dynamo_rows: int
    ) -> None:
        """
        Initialize the orchestrator agent with tools and configuration for a specific session.
        
        Args:
            session_id: Session ID for this conversation
            actor_id: Actor/user ID
            user_email: logged in user email
            max_sql_rows: Maximum rows to display for SQL results
            max_dynamo_rows: Maximum rows to display for DynamoDB results
        """
        # Check if agent already exists for this session
        if session_id in self._orchestrator_agents:
            self.logger.info(f"Orchestrator agent already exists for session: {session_id}")
            return
        
        self.logger.info(f"Initializing orchestrator agent for session: {session_id}")
        
        # Load gateway tools if MCP client is available
        gateway_tools = []
        if self.config.get('gateway_url'):
            # Initialize and start MCP client session
            _ = self.mcp_client  # Trigger lazy loading
            if self._mcp_client:
                if not self._mcp_session_active:
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
        system_prompt = self._build_system_prompt(user_email, max_sql_rows, max_dynamo_rows)
        
        # Create Bedrock model with reasoning
        bedrock_model = BedrockModel(
            model_id=self.config['bedrock_model_id'],
            cache_tools="default", # Using tool caching with BedrockModel
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 8000,
                }
            },
        )
        
        # Create memory hook for this session
        memory_hook = ECommerceMemoryHook(
            memory_client=self._memory_client,
            memory_id=self.config['agentcore_memory_id'],
            actor_id=actor_id,
            session_id=session_id,
            logger=self.logger
        )

        # Configure conversation management for production
        conversation_manager = SlidingWindowConversationManager(
            window_size=10,  # Limit history size
        )

        # Define system prompt with cache points
        system_content = [
            SystemContentBlock(
                text=system_prompt
            ),
            SystemContentBlock(cachePoint={"type": "default"})
        ]
        
        # Create the orchestrator agent for this session with memory hook
        agent = Agent(
            model=bedrock_model,
            tools=all_tools,
            hooks=[memory_hook],
            system_prompt=system_content, # System prompt with cache points
            conversation_manager=conversation_manager
        )
        
        # Store agent by session_id
        self._orchestrator_agents[session_id] = agent
        
        self.logger.info(f"Orchestrator agent initialized successfully for session: {session_id}")
    
    def get_orchestrator(self, session_id: str) -> Optional[Agent]:
        """Get the orchestrator agent for a specific session."""
        return self._orchestrator_agents.get(session_id)
    
    @property
    def orchestrator(self) -> Optional[Agent]:
        """Get the orchestrator agent. Note: Use get_orchestrator(session_id) for session-specific agents."""
        # Return the first agent if any exists (for backward compatibility)
        if self._orchestrator_agents:
            return next(iter(self._orchestrator_agents.values()))
        return None
    
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
    
    async def process_and_stream(self, user_query: str, user_email: Optional[str], session_id: str):
        """
        Process user input and stream responses.
        
        Args:
            user_query: The user's question or request
            user_email: The email address of the current logged-in user
            session_id: The session ID for this conversation
            
        Yields:
            Streaming response chunks, tool use events, and thinking events
        """
        log_conversation("User", f"user_query: {user_query}, user_email: {user_email}, session_id: {session_id}")
        
        # Get the agent for this specific session
        agent = self.agent_manager.get_orchestrator(session_id)
        if not agent:
            yield "Agent not initialized. Please try again."
            return
        
        try:
            start_time = time.time()
                      
            # Stream responses from the session-specific agent
            async for event in agent.stream_async(user_query):
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
            context: Request context with headers, metadata, and session_id
            
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
            
            # Get session_id from AgentCore context (set by X-Amzn-Bedrock-AgentCore-Runtime-Session-Id header)
            session_id = context.session_id if hasattr(context, 'session_id') else None
            if not session_id:
                # Fallback to generating one if not provided
                session_id = str(uuid.uuid4())
                self.logger.warning(f"No session_id in context, generated: {session_id}")
            
            # Extract headers for user context
            headers = context.request_headers if hasattr(context, 'request_headers') else None
            user_email, actor_id = self._extract_user_context(headers)
            
            self.logger.info(f"Processing - user_input: {user_input}, session_id: {session_id}, actor_id: {actor_id}, user_email: {user_email}")
            self.logger.info("\n🚀 Processing request...")
            
            # Initialize orchestrator with memory hook for this session
            self.agent_manager.initialize_orchestrator(
                session_id,
                actor_id,
                user_email,
                self.MAX_ROWS_FOR_SQL_RESULT_DISPLAY,
                self.MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY
            )
            
            # Process the request and stream responses with session-specific agent
            async for chunk in self.process_and_stream(user_input, user_email, session_id):
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
