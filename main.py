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
from psycopg2 import pool

from strands import Agent, tool, ToolContext
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
                        # AgentCore Memory returns uppercase role strings like "ASSISTANT" or "USER"
                        # Convert to lowercase for Strands agent messages
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
                top_k=5
            )
            
            # Get user facts, top 10
            facts = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=f"/users/{self.actor_id}/facts",
                query="What information do we know about the user? User email, location, past purchases, product reviews.",
                top_k=5
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
            # Convert Strands MessageRole enum to uppercase string for AgentCore Memory API
            # Strands uses MessageRole enum, but AgentCore expects string "USER" or "ASSISTANT"
            role_str = str(last_message["role"]).split('.')[-1].upper()
           
            # Save conversation turn to AgentCore Memory using create_event
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(content, role_str)]
            )
            
            self.logger.info(f"Saved {role_str} message to memory for session: {self.session_id}, actor_id: {self.actor_id}")
        
        except Exception as e:
            self.logger.error(f"Memory save error: {e}")
    
    def register_hooks(self, registry):
        """Register hook callbacks."""
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)


class AgentManager:
    """Manages all agent instances with lazy loading for a specific session."""
    
    def __init__(
        self, 
        logger: logging.Logger, 
        config: Dict[str, Any],
        session_id: str,
        user_email: str,
        tenant_id: str,
        actor_id: str,
        access_token: Optional[str] = None
    ):
        """
        Initialize the AgentManager.
        
        Args:
            logger: Logger instance
            config: Configuration dictionary with AWS settings, DB config, etc.
            session_id: Session ID for this conversation
            user_email: User's email address
            tenant_id: Tenant ID for multi-tenancy
            actor_id: Actor/user ID for memory namespacing
            access_token: Cognito access token for Bearer authentication (optional)
        """
        self.logger = logger
        self.config = config
        
        # User context (immutable for the session) - stored as private attributes
        self._session_id = session_id
        self._user_email = user_email
        self._tenant_id = tenant_id
        self._actor_id = actor_id
        self._access_token = access_token
        
        # Agent instances (lazy loaded)
        self._sql_agent = None
        self._kb_agent = None
        self._mcp_client = None
        self._orchestrator_agent = None  # Single orchestrator agent for this session
        self._mcp_session_active = False
        
        # Initialize memory client for AgentCore Memory
        self._memory_client = MemoryClient(region_name=config['aws_region'])

    @property
    def session_id(self) -> str:
        """Session ID for this conversation (read-only)."""
        return self._session_id
    
    @property
    def user_email(self) -> str:
        """User's email address (read-only)."""
        return self._user_email
    
    @property
    def tenant_id(self) -> str:
        """Tenant ID for multi-tenancy (read-only)."""
        return self._tenant_id
    
    @property
    def actor_id(self) -> str:
        """Actor/user ID for memory namespacing (read-only)."""
        return self._actor_id
    
    @property
    def access_token(self) -> Optional[str]:
        """Access token for authentication (read-only)."""
        return self._access_token

    @property
    def sql_agent(self):
        """Lazy initialization of SQL agent."""
        if self._sql_agent is None:
            self.logger.info(f"Initializing SQL agent client for schema: {self._tenant_id} ...")
            
            # Get reference to application-level schema cache
            schema_cache = self.config['schema_cache']
            cached_schema = schema_cache.get(self._tenant_id)
            
            if cached_schema:
                self.logger.info(f"Using cached schema for tenant: {self._tenant_id}")
            else:
                self.logger.info(f"No cached schema found for tenant: {self._tenant_id}, will extract from database")
            
            # Create SQL agent with optional cached schema
            self._sql_agent = sql_agent.SQLAgent(
                self.logger,
                self.config['db_pool'],
                self.config['aws_region'],
                self.config['bedrock_model_id'],
                self._tenant_id,
                self.config['chart_s3_bucket'],
                cached_schema=cached_schema,  # Pass cached schema if available
                cloudfront_domain=self.config.get('cloudfront_domain'),  # Pass CloudFront domain if configured
                valkey_config=self.config.get('valkey_config')  # Pass Valkey configuration
            )
            
            # If schema was extracted (not cached), store it in cache for future use
            if not cached_schema:
                schema_cache[self._tenant_id] = self._sql_agent.create_statements
                self.logger.info(f"Cached schema for tenant: {self._tenant_id} ({len(self._sql_agent.create_statements)} tables)")
        
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
                self.config['bedrock_model_arn'],
                self._tenant_id
            )
        return self._kb_agent
    
    @property
    def mcp_client(self) -> Optional[MCPClient]:
        """Lazy initialization of MCP client."""
        if self._mcp_client is None and self._access_token:
            self._initialize_mcp_client()
        return self._mcp_client
    
    def _initialize_mcp_client(self) -> None:
        """Initialize MCP client with Bearer token authentication."""
        if self.config.get('gateway_url') is None:
            self.logger.warning("Gateway URL not configured, MCP client will not be initialized")
            return
        
        if not self._access_token:
            self.logger.warning("No access token available for MCP client")
            return
        
        self.logger.info("Initializing MCP client with Bearer token...")
        
        # Create a simple Bearer token auth class for httpx
        class BearerAuth(httpx.Auth):
            def __init__(self, token: str):
                self.token = token
            
            def auth_flow(self, request: httpx.Request):
                request.headers["Authorization"] = f"Bearer {self.token}"
                yield request
        
        auth = BearerAuth(self._access_token)
        try:

            transport_factory = lambda: streamablehttp_client(
                url=self.config['gateway_url']
                , auth=auth
                , headers={
                    "x-amzn-bedrock-agentcore-runtime-custom-tenantid": self._tenant_id,
                }
            )
            self._mcp_client = MCPClient(transport_factory)
        except Exception as e:
            self.logger.error(f"Error creating MCP Client: {str(e)}")

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
            # E-commerce Assistant Agent

            <role>
            You are an intelligent E-commerce Assistant Agent designed to help users with queries related to products, orders, and reviews. Your primary responsibility is to analyze user queries and utilize the appropriate tools to provide accurate and helpful responses.
            You can also be asked about some internal financial info by certain admin users, you can use the Knowledge Base tool for that.
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
            - For queries about "my orders," "my products," or "my shipments," add a WHERE clause: `customers.email='{self._user_email}'`
            - Present results in this order:
            1. Always, display the results tabular data in a well-formatted markdown grid
            2. Merge column values when data is grouped by a column
            3. Show maximum {max_sql_rows} rows and include the total row count
            4. Include a graphical representation of the data when applicable
            5. Provide brief, concise key insights about the data
            6. If chart is returned, display it using this EXACT HTML format:
                - If chart.chart_url is available (preferred): <img src="{{chart_url}}" alt="Data Visualization" style="max-width: 100%;" />
                Replace {{chart_url}} with the actual URL from chart.chart_url
            7. Include the generated SQL statement within `<details>` tags at the end

            When using **get_answers_for_unstructured_data**:
            - Display the retrieved information in a clear, structured format
            - Include sources and preview within `<details>` tags at the end

            When using **ProductReviewLambda___get_product_reviews**:
            - Pass relevant parameters based on the query:
            - product_id(s) - can be comma separated
            - customer_id(s) - can be comma separated
            - product_name(s) - can be comma separated
            - customer_name(s) - can be comma separated
            - rating (s) - can be comma separated
            - top_rows
            - review_date range
            - For queries about "my reviews," filter by passing {self._user_email} as customer_email parameter
            - Show maximum {max_dynamo_rows} rows and include the total row count
            - Include all tool calls within `<details>` tags at the end
            </tool_instructions>

            <workflow>
            1. Analyze the user query to understand their intent
            2. Only answer the most recent user question unless explicitly asked to reference previous questions
            3. Determine which tool is most appropriate for addressing the query
            4. Call the selected tool with the necessary parameters
            5. Format the results in markdown for clear presentation
            6. Add concise insights or summaries if applicable
            7. Include technical details in collapsible sections
            </workflow>

            <response_format>
            1. First, provide your reasoning process to show how you analyzed the query and selected the appropriate tool.
            2. Present the requested information in a clear, structured format using markdown.
            3. First show any Tabular data in formatted table with header and merge repeating data.
            4. Then, if available, show the chat as "Data Visualization"
            5. Then, include any relevant summaries or insights in a concise manner.
            6. Always show SQL query and Tool Call info in complete details when present but at the end in collapsable `<details>` sections.
            </response_format>

            When responding to a user query, provide your answer immediately without any preamble, focusing only on addressing the user's specific request.
            """
    
    def initialize_orchestrator(
        self,
        max_sql_rows: int,
        max_dynamo_rows: int
    ) -> None:
        """
        Initialize the orchestrator agent with tools and configuration.
        
        Args:
            max_sql_rows: Maximum rows to display for SQL results
            max_dynamo_rows: Maximum rows to display for DynamoDB results
        """
        # Check if agent already exists
        if self._orchestrator_agent is not None:
            self.logger.info(f"Orchestrator agent already initialized for this session")
            return
        
        self.logger.info(f"Initializing orchestrator agent for actor: {self._actor_id}")
        
        # Load gateway tools if MCP client is available
        gateway_tools = []
        if self.config.get('gateway_url') and self._access_token:
            # Access mcp_client property (lazy initialization)
            if self.mcp_client:
                if not self._mcp_session_active:
                    self.mcp_client.__enter__()
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
        
        # Build system prompt using self.user_email
        system_prompt = self._build_system_prompt(max_sql_rows, max_dynamo_rows)
        
        # Create Bedrock model with reasoning
        bedrock_model = BedrockModel(
            model_id=self.config['bedrock_model_id'],
            #========== For Claude model ===============
            cache_tools="default", # Using tool caching with BedrockModel
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 8000,
                }
            },
            # #======== For Nova Models ===============
            # additionalModelRequestFields={
            #     "reasoningConfig": {
            #         "type": "enabled",
            #         "maxReasoningEffort": "medium"  # or "low" / "high"
            #     }
            # },
        )
        
        # Create memory hook for this session using self.session_id
        memory_hook = ECommerceMemoryHook(
            memory_client=self._memory_client,
            memory_id=self.config['agentcore_memory_id'],
            actor_id=self._actor_id,
            session_id=self._session_id,  # Using actual session_id for conversation tracking
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
        
        # Create the orchestrator agent with memory hook
        agent = Agent(
            model=bedrock_model,
            tools=all_tools,
            hooks=[memory_hook],
            system_prompt=system_content, # System prompt with cache points
            conversation_manager=conversation_manager,
            # Store only session_id in agent state for AgentManager lookup in tools
            state={"session_id": self._session_id}
        )
        
        # Store the orchestrator agent
        self._orchestrator_agent = agent
        
        self.logger.info(f"Orchestrator agent initialized successfully for actor: {self._actor_id}")
    
    @property
    def orchestrator(self) -> Optional[Agent]:
        """Get the orchestrator agent for this session."""
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
        self.agent_managers = {}  # session_id -> AgentManager
        self.session_timestamps = {}  # session_id -> last_access_time
        self.db_pool = None  # Database connection pool
        self.schema_cache = {}  # tenant_id -> Dict[str, str] (table_name -> CREATE statement)
        
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
            'chart_s3_bucket': self._validate_env_var("CHART_S3_BUCKET"),
            'cloudfront_domain': os.getenv("CLOUDFRONT_DOMAIN"),  # Optional CloudFront domain
            'db_config': {
                "host": self._validate_env_var("DB_HOST"),
                "port": self._validate_env_var("DB_PORT"),
                "database": self._validate_env_var("DB_NAME"),
                "user": self._validate_env_var("DB_USER"),
                "password": self._validate_env_var("DB_PASSWORD")
            },
            'gateway_url': os.getenv("GATEWAY_URL"),
            # Valkey cache configuration
            'valkey_config': {
                'endpoint': os.getenv("VALKEY_ENDPOINT"),
                'port': int(os.getenv("VALKEY_PORT", "6379")),
                'password': os.getenv("VALKEY_PASSWORD", ""),
                'use_tls': os.getenv("VALKEY_USE_TLS", "true").lower() == "true",
                'cache_ttl_seconds': int(os.getenv("CACHE_TTL_SECONDS", "3600")),
                'similarity_threshold': float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90")),
                'similarity_threshold_min': float(os.getenv("CACHE_SIMILARITY_THRESHOLD_MIN", "0.75")),
                'embed_model': os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
            }
        }
        
        # Initialize database connection pool
        self._initialize_db_pool()
        
        # Store pool reference in config for AgentManager
        self.config['db_pool'] = self.db_pool
        
        # Store schema cache reference in config for AgentManager
        self.config['schema_cache'] = self.schema_cache
        
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
    
    def _initialize_db_pool(self) -> None:
        """Initialize database connection pool."""
        try:
            db_config = self.config['db_config']
            
            # Create threaded connection pool
            # minconn: minimum connections to maintain
            # maxconn: maximum connections allowed
            self.db_pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password']
            )
            
            self.logger.info(f"Database connection pool initialized (min=2, max=10)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection pool: {str(e)}")
            raise ValueError(f"Database connection pool initialization failed: {str(e)}")
    
    def _get_or_create_agent_manager(
        self, 
        session_id: str,
        user_email: str,
        tenant_id: str,
        actor_id: str,
        access_token: Optional[str] = None
    ) -> AgentManager:
        """
        Get existing AgentManager for a session or create a new one if it doesn't exist.
        
        Args:
            session_id: Session ID for this conversation
            user_email: User's email address
            tenant_id: Tenant ID for multi-tenancy
            actor_id: Actor/user ID for memory namespacing
            access_token: Cognito access token for Bearer authentication (optional)
            
        Returns:
            AgentManager instance for this session (existing or new)
        """
        # Check if AgentManager already exists for this session
        if session_id not in self.agent_managers:
            # Create new AgentManager if it doesn't exist
            self.logger.info(f"Creating new AgentManager for session: {session_id}")
            agent_manager = AgentManager(
                self.logger,
                self.config,
                session_id,
                user_email,
                tenant_id,
                actor_id,
                access_token
            )
            # Store in dictionary
            self.agent_managers[session_id] = agent_manager
        else:
            self.logger.info(f"Reusing existing AgentManager for session: {session_id}")
        
        # Update last access timestamp
        self.session_timestamps[session_id] = time.time()
        
        return self.agent_managers[session_id]
      
    def _extract_user_context(self, headers: Optional[Dict]) -> Tuple[Optional[str], str]:
        """
        Extract user email, actor_id and tenant_id from request headers.
        
        Args:
            headers: Request headers containing Authorization token
            
        Returns:
            Tuple of (user_email, actor_id, tenant_id)
        """
        user_email = None
        actor_id = 'user'  # Default actor
        tenant_id = 'N/A'
        
        if not headers:
            return user_email, actor_id, tenant_id
        
        # Check for custom actor ID header
        if 'x-amzn-bedrock-agentcore-runtime-custom-actorid' in headers:
            actor_id = headers.get('x-amzn-bedrock-agentcore-runtime-custom-actorid')

        # Check for custom tenant ID header
        if 'x-amzn-bedrock-agentcore-runtime-custom-tenantid' in headers:
            tenant_id = headers.get('x-amzn-bedrock-agentcore-runtime-custom-tenantid')
        
        # Try to extract from JWT token
        auth_header = headers.get("Authorization")
        if auth_header:
            try:
                token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else auth_header
                claims = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
                
                username = claims.get("username")
                if username:
                    user_email = f"{username}@email.com"
                    # Infer actor id from user_email if it not already passed through the custom headers
                    if actor_id == 'user':
                        actor_id = user_email.replace(".", "_").replace("@", "-")

                # Get the tenant_id from the token claim if it not already passed through the custom headers
                if tenant_id == 'N/A' and claims.get("tenantId"):
                    tenant_id = claims.get("tenantId")

                self.logger.info(f"Extracted user context: email={user_email}, actor_id={actor_id}, tenant_id={tenant_id}")
            except Exception as e:
                self.logger.warning(f"Failed to parse JWT token: {str(e)}")
        
        return user_email, actor_id, tenant_id
    
    async def process_and_stream(self, user_query: str, agent_manager: AgentManager):
        """
        Process user input and stream responses.
        
        Args:
            user_query: The user's question or request
            agent_manager: The AgentManager instance for this session
            
        Yields:
            Streaming response chunks, tool use events, and thinking events
        """
        log_conversation("User", f"user_query: {user_query}, user_email: {agent_manager.user_email}, session_id: {agent_manager.session_id}")
        
        # Get the agent from the session-specific AgentManager
        agent = agent_manager.orchestrator
        if not agent:
            yield "Agent not initialized. Please try again."
            return
        
        try:
            start_time = time.time()
                      
            # Stream responses from the agent
            async for event in agent.stream_async(user_query):
                if "data" in event:
                    yield event["data"]
                
                # Yield tool use events for UI display
                elif "current_tool_use" in event:
                    tool_use = event["current_tool_use"]
                    if tool_use.get("name"):
                        tool_info = f"Tool: {tool_use['name']}\nInput: {tool_use.get('input', {})}"
                        yield f"\n[TOOL USE]{tool_info}\n"
                
                # Yield reasoning/thinking events for UI (Claude model)
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
            # Clean up old sessions periodically (1 hour timeout)
            self.cleanup_old_sessions(max_age_seconds=3600)
            
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
            user_email, actor_id, tenant_id = self._extract_user_context(headers)
            
            # Extract access token from Authorization header
            access_token = None
            if headers and "Authorization" in headers:
                auth_header = headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    access_token = auth_header.replace("Bearer ", "")
                    self.logger.info("Access token extracted from Authorization header")
            
            self.logger.info(f"Processing - user_input: {user_input}, session_id: {session_id}, actor_id: {actor_id}, user_email: {user_email}")
            self.logger.info("\n🚀 Processing request...")
            
            # Get or create session-specific AgentManager with user context
            agent_manager = self._get_or_create_agent_manager(
                session_id, user_email, tenant_id, actor_id, access_token
            )
            
            # Initialize orchestrator (reads user context from agent_manager)
            agent_manager.initialize_orchestrator(
                self.MAX_ROWS_FOR_SQL_RESULT_DISPLAY,
                self.MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY
            )
            
            # Process the request and stream responses (reads user context from agent_manager)
            async for chunk in self.process_and_stream(user_input, agent_manager):
                self.logger.info(f"Streaming chunk: {chunk[:100]}..." if len(str(chunk)) > 100 else f"Streaming chunk: {chunk}")
                yield chunk
        
        except Exception as e:
            error_msg = f"Error in handle_request: {str(e)}"
            self.logger.error(error_msg)
            self.logger.info(f"\n❌ {error_msg}")
            yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
    
    def cleanup(self) -> None:
        """Clean up application resources."""
        self.logger.info("Cleaning up application resources...")
        
        # Clean up all agent managers
        for session_id, agent_manager in self.agent_managers.items():
            try:
                agent_manager.cleanup()
                self.logger.info(f"Cleaned up AgentManager for session: {session_id}")
            except Exception as e:
                self.logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        
        self.agent_managers.clear()
        self.session_timestamps.clear()
        
        # Close database connection pool
        if self.db_pool:
            try:
                self.db_pool.closeall()
                self.logger.info("Database connection pool closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection pool: {str(e)}")
    
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up resources for a specific session.
        
        Args:
            session_id: Session ID to clean up
        """
        if session_id in self.agent_managers:
            try:
                self.agent_managers[session_id].cleanup()
                del self.agent_managers[session_id]
                if session_id in self.session_timestamps:
                    del self.session_timestamps[session_id]
                self.logger.info(f"Cleaned up AgentManager for session: {session_id}")
            except Exception as e:
                self.logger.error(f"Error cleaning up session {session_id}: {str(e)}")
    
    def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up sessions older than max_age_seconds.
        
        Args:
            max_age_seconds: Maximum age in seconds before a session is considered stale (default: 1 hour)
        """
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, last_access in self.session_timestamps.items():
            if current_time - last_access > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        if sessions_to_remove:
            self.logger.info(f"Cleaning up {len(sessions_to_remove)} old sessions")
            for session_id in sessions_to_remove:
                self.cleanup_session(session_id)


#=====================================================================================
# UTILITY FUNCTIONS
#=====================================================================================

def log_conversation(role: str, content: str, tool_calls: Optional[List] = None) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {role}: {content[:500]}..." if len(content) > 500 else f"[{timestamp}] {role}: {content}")
    
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

@tool(context=True)
def get_answers_for_structured_data(user_query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products and orders transactional data stored in the RDS Postgres database.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """
    # Get the session_id from tool context
    session_id = (tool_context.agent.state.get("session_id")
                 if tool_context.agent and tool_context.agent.state
                 else None)
    
    if not session_id:
        logger.error("session_id not found in tool context")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}
    
    # Look up the AgentManager from the global app instance
    app_instance = get_app_instance()
    agent_manager = app_instance.agent_managers.get(session_id)
    
    if not agent_manager:
        logger.error(f"AgentManager not found for session_id: {session_id}")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}

    log_conversation("User", f"user_query= {user_query}, for the user_email={agent_manager.user_email} for tenant_id={agent_manager.tenant_id}")

    try:
        start_time = time.time()
        response = agent_manager.sql_agent.process_query(user_query)
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the structured assistant")
        return response

    except Exception as e:
        logger.error(f"Error processing request in the structured assistant: {str(e)}")
        return {"message": {"content": f"I'm sorry, I encountered an error in the structured assistant: {str(e)}. Please try again later."}}


@tool(context=True)
def get_answers_for_unstructured_data(user_query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products specifications based on the unstructured data from the Bedrock knowledge-base.

    Args:
        user_query: the user query based on product or orders.
        tool_context: Used for logged in user info passed as agent state

    Returns:
        Detailed answers which is formatted to the user question
    """

    # Get the session_id from tool context
    session_id = (tool_context.agent.state.get("session_id")
                 if tool_context.agent and tool_context.agent.state
                 else None)
    
    if not session_id:
        logger.error("session_id not found in tool context")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}
    
    # Look up the AgentManager from the global app instance
    app_instance = get_app_instance()
    agent_manager = app_instance.agent_managers.get(session_id)
    
    if not agent_manager:
        logger.error(f"AgentManager not found for session_id: {session_id}")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}

    log_conversation("User", f"user_query= {user_query}, for the user_email={agent_manager.user_email} for tenant_id={agent_manager.tenant_id}")

    try:
        start_time = time.time()
        response = agent_manager.kb_agent.process_query(user_query=user_query)
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the unstructured assistant")
        return response

    except Exception as e:
        logger.error(f"Error processing request in the unstructured assistant: {str(e)}")
        return {"message": {"content": f"I'm sorry, I encountered an error in the unstructured assistant: {str(e)}. Please try again later."}}


@tool(context=True)
def get_default_questions(user_query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Lists sample of questions that the user can ask related to products and orders. These questions can be based on the un-strutured data from the Bedrock knowledge-base or based on the structured data from the RDS Postgres database.

    Args:
        user_query: the user query based on product or orders.
        tool_context: Used for logged in user info passed as agent state

    Returns:
        Detailed answers which is formatted to the user question
    """

    # Get the session_id from tool context
    session_id = (tool_context.agent.state.get("session_id")
                 if tool_context.agent and tool_context.agent.state
                 else None)
    
    if not session_id:
        logger.error("session_id not found in tool context")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}
    
    # Look up the AgentManager from the global app instance
    app_instance = get_app_instance()
    agent_manager = app_instance.agent_managers.get(session_id)
    
    if not agent_manager:
        logger.error(f"AgentManager not found for session_id: {session_id}")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}

    log_conversation("User", f"user_query= {user_query}, for the user_email={agent_manager.user_email} for tenant_id={agent_manager.tenant_id}")
    
    try:
        start_time = time.time()

        response = (
            "Questions related to Order and Products transactional data in natural language \n"
            "Examples:\n"
        )

        for question in agent_manager.sql_agent.get_sample_questions():
            response += f"  • {question}\n"

        response += (
            "\n\n, Questions related to Products based on the product specifications \n"
            "Examples:\n"
        )
        for question in agent_manager.kb_agent.get_sample_questions():
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
