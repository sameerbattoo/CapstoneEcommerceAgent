# Standard library imports
import logging
import sys
import os

# Minimal imports needed before configuration
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from datetime import datetime

import uuid
import jwt  # PyJWT
from psycopg2 import pool # for DB pool

# Import orchestrator classes
from agent.orch_agent import AgentManager

# Import dotenv for loading environment variables
from dotenv import load_dotenv

# ADDED: BEDROCK_AGENTCORE IMPORT
from bedrock_agentcore.runtime import BedrockAgentCoreApp

AWS_SECRET_NAME = "capstone-ecommerce-agent-config" # AWS Secret Manager 

#=====================================================================================
# LOGGING CONFIGURATION FUNCTION
# Called AFTER Secrets Manager loads LOG_LEVEL
#=====================================================================================

def configure_logging(log_level: str = "INFO"):
    """
    Configure logging with the specified log level.
    Should be called AFTER loading configuration from Secrets Manager.
    
    Args:
        log_level: Log level string (INFO, WARNING, ERROR, etc.)
    """
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add StreamHandler to root logger
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level_value)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(stream_handler)
    
    # Set log level for common noisy libraries
    logging.getLogger("strands").setLevel(log_level_value)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("bedrock_agentcore").setLevel(log_level_value)


# Create application logger (will be configured later)
logger = logging.getLogger("eCommerceAgent")

#=====================================================================================
# MAIN CLASSES
#=====================================================================================

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
        self.background_cache_task = None  # Track background caching task
        
        # Constants
        self.MAX_ROWS_FOR_SQL_RESULT_DISPLAY = 20
        self.MAX_ROWS_FOR_DYNAMODB_RESULT_DISPLAY = 5
    
    def load_configuration(self) -> None:
        """Load configuration from AWS Secrets Manager or .env file."""
        # Try loading from AWS Secrets Manager
        if not self._load_secrets_from_aws():
            load_dotenv()
        
        # NOW configure logging with LOG_LEVEL from Secrets Manager
        log_level = os.getenv("LOG_LEVEL", "INFO")
        configure_logging(log_level)
        
        # Initialize metrics logging after logging is configured
        import metrics_logger
        namespace = os.getenv("METRICS_NAMESPACE", "CapstoneECommerceAgent/Metrics")
        enable_metrics = os.getenv("ENABLE_METRICS_LOGGING", "true").lower() == "true"
        metrics_logger.initialize_metrics_logging(namespace=namespace, enable_metrics=enable_metrics)
        
        # Validate and set required environment variables
        self.config = {
            'aws_region': self._validate_env_var("AWS_REGION"),
            'bedrock_model_id': self._validate_env_var("BEDROCK_MODEL_ID"),
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
                'similarity_threshold': float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.99")),
                'similarity_threshold_min': float(os.getenv("CACHE_SIMILARITY_THRESHOLD_MIN", "0.70")),
                'embed_model': os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
            }
        }
        
        # Initialize database connection pool
        self._initialize_db_pool()
        
        # Store pool reference in config for AgentManager
        self.config['db_pool'] = self.db_pool
        
        # Store schema cache reference in config for AgentManager
        self.config['schema_cache'] = self.schema_cache
        
        # Start background schema caching (non-blocking)
        self.start_background_schema_caching()
    
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
            
            return True
            
        except ClientError as e:
            self.logger.error(f"Error loading secrets from AWS Secrets Manager: {str(e)}")
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
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection pool: {str(e)}")
            raise ValueError(f"Database connection pool initialization failed: {str(e)}")
    
    def _discover_tenants(self) -> List[str]:
        """
        Discover all tenant schemas in the database.
        
        Returns:
            List of tenant schema names (e.g., ['tenanta', 'tenantb'])
        """
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            # Query for all schemas excluding system schemas
            cursor.execute("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'public')
                ORDER BY schema_name
            """)
            
            tenants = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            self.logger.info(f"Discovered {len(tenants)} tenant schemas: {tenants}")
            return tenants
            
        except Exception as e:
            self.logger.error(f"Failed to discover tenants: {str(e)}")
            return []
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _cache_tenant_schema(self, tenant_id: str) -> bool:
        """
        Extract and cache schema for a single tenant.
        
        Args:
            tenant_id: Tenant ID (schema name)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting schema extraction for tenant: {tenant_id}")
            
            # Import SchemaExtractor here to avoid circular imports
            from agent.sql_agent import SchemaExtractor
            
            # Extract schema
            schema_extractor = SchemaExtractor(self.db_pool, tenant_id, self.logger)
            create_statements = schema_extractor.extract_schema()
            
            # Store in cache using lowercase key for consistent lookups
            cache_key = tenant_id.lower()
            self.schema_cache[cache_key] = create_statements
            
            self.logger.info(f"Successfully cached schema for tenant '{tenant_id}' with cache key '{cache_key}' ({len(create_statements)} tables)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache schema for tenant '{tenant_id}': {str(e)}")
            return False
    
    def _background_schema_cache_worker(self) -> None:
        """
        Background worker that caches schemas for all discovered tenants.
        Runs in a separate thread via ThreadPoolExecutor.
        """
        try:
            self.logger.info("Background schema caching started")
            start_time = time.time()
            
            # Discover all tenants
            tenants = self._discover_tenants()
            
            if not tenants:
                self.logger.warning("No tenants discovered for schema caching")
                return
            
            # Cache each tenant's schema
            success_count = 0
            failed_count = 0
            
            for tenant_id in tenants:
                # Check if already cached (use lowercase key for lookup)
                cache_key = tenant_id.lower()
                if cache_key in self.schema_cache:
                    self.logger.info(f"Schema for tenant '{tenant_id}' already cached, skipping")
                    success_count += 1
                    continue
                
                # Cache the schema
                if self._cache_tenant_schema(tenant_id):
                    success_count += 1
                else:
                    failed_count += 1
            
            # Log completion summary
            duration = time.time() - start_time
            self.logger.info(
                f"Background schema caching completed in {duration:.2f}s: "
                f"{success_count}/{len(tenants)} tenants successful, {failed_count} failed"
            )
            
        except Exception as e:
            self.logger.error(f"Background schema caching encountered an error: {str(e)}")
    
    def start_background_schema_caching(self) -> None:
        """
        Start background schema caching task asynchronously.
        This method returns immediately without blocking.
        """
        try:
            import asyncio
            import concurrent.futures
            
            self.logger.info("Initiating background schema caching...")
            
            # Create a ThreadPoolExecutor with a single worker
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="schema-cache")
            
            # Submit the background task
            future = executor.submit(self._background_schema_cache_worker)
            
            # Store reference for potential cleanup
            self.background_cache_task = {
                'executor': executor,
                'future': future
            }
            
            self.logger.info("Background schema caching task submitted (non-blocking)")
            
        except Exception as e:
            self.logger.error(f"Failed to start background schema caching: {str(e)}")
            self.logger.warning("Application will continue with on-demand schema extraction")
    
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
            # Set the app instance reference for tool functions to access
            from agent.orch_agent import set_app_instance
            set_app_instance(self)
            
            # Create new AgentManager if it doesn't exist
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}]: user_query: {user_query}, user_email: {agent_manager.user_email}, session_id: {agent_manager.session_id}, tenant_id: {agent_manager.tenant_id}")
        
        # Get the agent from the session-specific AgentManager
        orch_agent = agent_manager.orchestrator_agent
        if not orch_agent:
            yield "Agent not initialized. Please try again."
            return
        
        # Reset session metrics for this query
        agent_manager.reset_session_metrics()
        
        start_time = time.time()
        orchestrator_input_tokens = 0
        orchestrator_output_tokens = 0
        orchestrator_cache_read_tokens = 0
        orchestrator_cache_write_tokens = 0
        
        # Track tool executions for metrics
        tool_start_times = {}  # tool_name -> start_time
        current_tools_in_use = set()  # Track which tools are currently executing
        
        try:
            # Stream responses from the orch agent
            async for event in orch_agent.stream_async(user_query):
                if "data" in event:
                    yield event["data"]
                
                # Yield tool use events for UI display AND track metrics
                elif "current_tool_use" in event:
                    tool_use = event["current_tool_use"]
                    tool_name = tool_use.get("name")
                    
                    if tool_name:
                        # Record start time for this tool
                        tool_start_times[tool_name] = time.time()
                        current_tools_in_use.add(tool_name)
                        
                        tool_info = f"Tool: {tool_name}\nInput: {tool_use.get('input', {})}"
                        yield f"\n[TOOL USE]{tool_info}\n"
                
                # Check for message event which indicates tool completion
                elif "message" in event:
                    # When a message event fires, any tools that were in use have completed
                    # Only emit metrics for MCP tools here (native tools are tracked elsewhere)
                    for tool_name in list(current_tools_in_use):
                        # Only process MCP tools (they have ___ in their names)
                        if "___" in tool_name and tool_name in tool_start_times:
                            tool_end_time = time.time()
                            tool_start = tool_start_times[tool_name]
                            
                            # Emit metrics for MCP tool execution
                            self.logger.emit_step_metrics(
                                session_id=agent_manager.session_id,
                                tenant_id=agent_manager.tenant_id,
                                step_name=f"mcp_tool_{tool_name}",
                                start_time=tool_start,
                                end_time=tool_end_time,
                                input_tokens=0,
                                output_tokens=0,
                                cache_read_tokens=0,
                                cache_write_tokens=0,
                                status="success",
                                additional_data={
                                    "tool_name": tool_name,
                                    "tool_type": "mcp"
                                }
                            )
                            
                            # Remove from tracking
                            del tool_start_times[tool_name]
                            current_tools_in_use.remove(tool_name)
                
                # Yield reasoning/thinking events for UI (Claude model)
                elif "reasoning" in event and "reasoningText" in event:
                    yield f"\n[THINKING]{event['reasoningText']}\n"

                # To capture the token count for the orch agent
                elif "result" in event:
                    result = event["result"]
                    metrics_summary = result.metrics.get_summary()
                    if metrics_summary and "accumulated_usage" in metrics_summary:
                        accumulated_usage_metrics_summary = metrics_summary["accumulated_usage"]
                        orchestrator_input_tokens = accumulated_usage_metrics_summary.get("inputTokens", 0)
                        orchestrator_output_tokens = accumulated_usage_metrics_summary.get("outputTokens", 0)
                        orchestrator_cache_read_tokens = accumulated_usage_metrics_summary.get("cacheReadInputTokens", 0)
                        orchestrator_cache_write_tokens = accumulated_usage_metrics_summary.get("cacheWriteInputTokens", 0)
            
            end_time = time.time()
            
            # Add orchestrator tokens to session total
            agent_manager._token_accumulator_callback(
                orchestrator_input_tokens, 
                orchestrator_output_tokens, 
                "orchestration_agent",
                orchestrator_cache_read_tokens,
                orchestrator_cache_write_tokens
            )
            
            # Get accumulated session metrics
            session_metrics = agent_manager.get_session_metrics()

            # Yield metrics event for UI display
            duration_seconds = end_time - session_metrics["session_start_time"]
            model_id = self.config.get("bedrock_model_id", "unknown")
            
            # Calculate cost using metrics_logger
            import metrics_logger
            cost_breakdown = metrics_logger.calculate_cost(
                session_metrics["total_input_tokens"],
                session_metrics["total_output_tokens"],
                model_id,
                session_metrics.get("total_cache_read_tokens", 0),
                session_metrics.get("total_cache_write_tokens", 0)
            )
            
            metrics_data = {
                "duration_seconds": round(duration_seconds, 2),
                "input_tokens": session_metrics["total_input_tokens"],
                "output_tokens": session_metrics["total_output_tokens"],
                "cache_read_tokens": session_metrics.get("total_cache_read_tokens", 0),
                "cache_write_tokens": session_metrics.get("total_cache_write_tokens", 0),
                "step_count": session_metrics["step_count"],
                "model_id": model_id,
                "cost_usd": cost_breakdown["total_cost_usd"],
                "cache_savings_usd": cost_breakdown["cache_savings_usd"],
                "semantic_cache_hits": session_metrics.get("semantic_cache_hits", [])
            }
            yield f"\n[METRICS]{json.dumps(metrics_data)}\n"

            
            # Emit step metrics for orchestration (existing behavior)
            self.logger.emit_step_metrics(
                session_id=agent_manager.session_id,
                tenant_id=agent_manager.tenant_id,
                step_name="orchestration_agent_execution",
                start_time=start_time,
                end_time=end_time,
                input_tokens=orchestrator_input_tokens,
                output_tokens=orchestrator_output_tokens,
                cache_read_tokens=orchestrator_cache_read_tokens,
                cache_write_tokens=orchestrator_cache_write_tokens,
                status="success",
                additional_data={
                    "user_email": agent_manager.user_email,
                    "actor_id": agent_manager.actor_id
                }
            )
            
            # NEW: Emit end-to-end metrics with accumulated tokens from all steps
            self.logger.emit_step_metrics(
                session_id=agent_manager.session_id,
                tenant_id=agent_manager.tenant_id,
                step_name="capstone_ecommerce_E2E_agent_execution",
                start_time=session_metrics["session_start_time"],
                end_time=end_time,
                input_tokens=session_metrics["total_input_tokens"],
                output_tokens=session_metrics["total_output_tokens"],
                cache_read_tokens=session_metrics.get("total_cache_read_tokens", 0),
                cache_write_tokens=session_metrics.get("total_cache_write_tokens", 0),
                status="success",
                additional_data={
                    "user_email": agent_manager.user_email,
                    "actor_id": agent_manager.actor_id,
                    "step_count": session_metrics["step_count"],
                    "query_preview": user_query[:100]
                }
            )
            
        except Exception as e:
            end_time = time.time()
            error_msg = f"Error processing request: {str(e)}"
            self.logger.error(error_msg)
            
            # Add orchestrator tokens even on error
            agent_manager._token_accumulator_callback(
                orchestrator_input_tokens, 
                orchestrator_output_tokens, 
                "orchestration_agent_error",
                orchestrator_cache_read_tokens,
                orchestrator_cache_write_tokens
            )
            
            # Get accumulated session metrics
            session_metrics = agent_manager.get_session_metrics()
            
            # Emit metrics for orchestration error
            self.logger.emit_step_metrics(
                session_id=agent_manager.session_id,
                tenant_id=agent_manager.tenant_id,
                step_name="orchestration_agent_execution",
                start_time=start_time,
                end_time=end_time,
                input_tokens=orchestrator_input_tokens,
                output_tokens=orchestrator_output_tokens,
                cache_read_tokens=orchestrator_cache_read_tokens,
                cache_write_tokens=orchestrator_cache_write_tokens,
                status="error",
                additional_data={
                    "error": str(e),
                    "user_email": agent_manager.user_email,
                    "actor_id": agent_manager.actor_id
                }
            )
            
            # NEW: Emit end-to-end metrics for error case
            self.logger.emit_step_metrics(
                session_id=agent_manager.session_id,
                tenant_id=agent_manager.tenant_id,
                step_name="capstone_ecommerce_E2E_agent_execution",
                start_time=session_metrics["session_start_time"] if session_metrics["session_start_time"] else start_time,
                end_time=end_time,
                input_tokens=session_metrics["total_input_tokens"],
                output_tokens=session_metrics["total_output_tokens"],
                cache_read_tokens=session_metrics.get("total_cache_read_tokens", 0),
                cache_write_tokens=session_metrics.get("total_cache_write_tokens", 0),
                status="error",
                additional_data={
                    "error": str(e),
                    "user_email": agent_manager.user_email,
                    "actor_id": agent_manager.actor_id,
                    "step_count": session_metrics["step_count"],
                    "query_preview": user_query[:100]
                }
            )
            
            # Yield metrics event even on error
            duration_seconds = end_time - (session_metrics["session_start_time"] if session_metrics["session_start_time"] else start_time)
            model_id = self.config.get("bedrock_model_id", "unknown")
            
            # Calculate cost using metrics_logger
            import metrics_logger
            cost_breakdown = metrics_logger.calculate_cost(
                session_metrics["total_input_tokens"],
                session_metrics["total_output_tokens"],
                model_id,
                session_metrics.get("total_cache_read_tokens", 0),
                session_metrics.get("total_cache_write_tokens", 0)
            )
            
            metrics_data = {
                "duration_seconds": round(duration_seconds, 2),
                "input_tokens": session_metrics["total_input_tokens"],
                "output_tokens": session_metrics["total_output_tokens"],
                "cache_read_tokens": session_metrics.get("total_cache_read_tokens", 0),
                "cache_write_tokens": session_metrics.get("total_cache_write_tokens", 0),
                "step_count": session_metrics["step_count"],
                "model_id": model_id,
                "cost_usd": cost_breakdown["total_cost_usd"],
                "cache_savings_usd": cost_breakdown["cache_savings_usd"],
                "semantic_cache_hits": session_metrics.get("semantic_cache_hits", []),
                "status": "error"
            }
            yield f"\n[METRICS]{json.dumps(metrics_data)}\n"
            
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
                yield chunk
        
        except Exception as e:
            error_msg = f"Error in handle_request: {str(e)}"
            self.logger.error(error_msg)
            yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
    
    def cleanup(self) -> None:
        """Clean up application resources."""
        self.logger.info("Cleaning up application resources...")
        
        # Clean up background caching task
        if self.background_cache_task:
            try:
                executor = self.background_cache_task.get('executor')
                future = self.background_cache_task.get('future')
                
                if future and not future.done():
                    self.logger.info("Waiting for background schema caching to complete...")
                    # Wait up to 10 seconds for background task to finish
                    try:
                        future.result(timeout=10)
                    except Exception as e:
                        self.logger.warning(f"Background caching task did not complete cleanly: {str(e)}")
                
                if executor:
                    executor.shutdown(wait=False)
                    self.logger.info("Background schema caching executor shut down")
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up background caching task: {str(e)}")
        
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
    try:
        # Get or create application instance
        app_instance = get_app_instance()
        
        # Process the request and stream responses
        async for chunk in app_instance.handle_request(payload, context):
            yield chunk
    
    except Exception as e:
        error_msg = f"Error in main entry point: {str(e)}"
        logger.error(error_msg)
        yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."


if __name__ == "__main__":
    app.run()
