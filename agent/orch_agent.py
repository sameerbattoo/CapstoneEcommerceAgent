"""Orchestrator Agent and related classes for managing agent instances."""

import logging
import time
import httpx
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime

# Strands imports
from strands import Agent
from strands.models import BedrockModel
from strands.hooks.events import MessageAddedEvent, AgentInitializedEvent
from strands.hooks.registry import HookProvider
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.tools.mcp.mcp_client import MCPClient
from strands.types.content import SystemContentBlock

# AWS imports
from botocore.credentials import Credentials
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from bedrock_agentcore.memory import MemoryClient
from mcp.client.streamable_http import streamablehttp_client

# Import sub-agents
from agent import sql_agent, kb_agent, chart_agent

# Module-level logger - shared by all functions in this module
logger = logging.getLogger("eCommerceAgent")

# Module-level app instance reference - set by initialize_orchestrator
_app_instance_ref = None



#=====================================================================================
# UTILITY FUNCTIONS
#=====================================================================================

def log_conversation(role: str, content: str, tool_calls: Optional[List] = None) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    import json
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {role}: {content[:500]}..." if len(content) > 500 else f"[{timestamp}] {role}: {content}")
    
    if tool_calls:
        for call in tool_calls:
            logger.info(f"  Tool used: {call['name']} with args: {json.dumps(call['args'])}")

MAX_LAST_CONVERSATIONS = 3
MAX_FACTS = 5
MAX_PREFERENCES = 5

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
            # Load the last 3 conversation turns from memory
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=MAX_LAST_CONVERSATIONS,
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
            # Get user preferences, top 5
            preferences = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=f"/users/{self.actor_id}/preferences",
                query="What does the user prefer? What are their settings and product choices, preferred products?",
                top_k=MAX_PREFERENCES
            )
            
            # Get user facts, top 5
            facts = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=f"/users/{self.actor_id}/facts",
                query="What information do we know about the user? User email, location, past purchases, product reviews.",
                top_k=MAX_FACTS
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
        self._chart_agent = None
        self._mcp_client = None
        self._orchestrator_agent = None  # Single orchestrator agent for this session
        self._mcp_session_active = False
        
        # Initialize memory client for AgentCore Memory
        self._memory_client = MemoryClient(region_name=config['aws_region'])
        
        # Session-level metrics accumulator for end-to-end tracking
        self._session_metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "step_count": 0,
            "session_start_time": None,
            "semantic_cache_hits": []  # Track semantic cache hits for UI display
        }
    
    def _token_accumulator_callback(self, input_tokens: int, output_tokens: int, step_name: str = "unknown", cache_read_tokens: int = 0, cache_write_tokens: int = 0):
        """
        Callback function for sub-agents to report token usage.
        This accumulates tokens across all steps for end-to-end metrics.
        
        Args:
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
            step_name: Name of the step reporting tokens (for debugging)
            cache_read_tokens: Number of tokens read from cache
            cache_write_tokens: Number of tokens written to cache
        """
        self._session_metrics["total_input_tokens"] += input_tokens
        self._session_metrics["total_output_tokens"] += output_tokens
        self._session_metrics["total_cache_read_tokens"] += cache_read_tokens
        self._session_metrics["total_cache_write_tokens"] += cache_write_tokens
        self._session_metrics["step_count"] += 1
        self.logger.debug(f"Token accumulator: step={step_name}, input={input_tokens}, output={output_tokens}, "
                         f"cache_read={cache_read_tokens}, cache_write={cache_write_tokens}, "
                         f"total_input={self._session_metrics['total_input_tokens']}, "
                         f"total_output={self._session_metrics['total_output_tokens']}, "
                         f"total_cache_read={self._session_metrics['total_cache_read_tokens']}, "
                         f"total_cache_write={self._session_metrics['total_cache_write_tokens']}")
    
    def reset_session_metrics(self):
        """Reset session metrics for a new query."""
        self._session_metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "step_count": 0,
            "session_start_time": time.time(),
            "semantic_cache_hits": []  # Reset cache hits for new query
        }
    
    def record_semantic_cache_hit(self, cache_tier: int, step_name: str = "sql_query", cache_hit_similarity_score: float = 0.0):
        """
        Record a semantic cache hit for display in UI.
        
        Args:
            cache_tier: 1 for high similarity, 2 for SQL validation
            step_name: Name of the step that hit cache
            cache_hit_similarity_score: similarity_score for semantic search
        """
        self._session_metrics["semantic_cache_hits"].append({
            "tier": cache_tier,
            "step": step_name,
            "cache_hit_similarity_score": cache_hit_similarity_score,
            "timestamp": time.time()
        })
        self.logger.info(f"Recorded semantic cache hit: tier={cache_tier}, step={step_name}, cache_hit_similarity_score={cache_hit_similarity_score}")
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get accumulated session metrics."""
        return self._session_metrics.copy()

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
    def orchestrator_agent(self) -> Optional[Agent]:
        """Get the orchestrator agent for this session."""
        return self._orchestrator_agent
 
    @property
    def sql_agent(self):
        """Lazy initialization of SQL agent."""
        if self._sql_agent is None:
            # Get reference to application-level schema cache
            schema_cache = self.config.get('schema_cache', {})
            # Use lowercase tenant_id for cache lookup
            cache_key = self._tenant_id.lower()
            cached_schema = schema_cache.get(cache_key, {})
            
            # Create SQL agent with optional cached schema and session_id
            self._sql_agent = sql_agent.SQLAgent(
                self.logger,
                self.config['db_pool'],
                self.config['aws_region'],
                self.config['bedrock_model_id'],
                self._tenant_id,
                self.config['chart_s3_bucket'],
                cached_schema=cached_schema,  # Pass cached schema if available
                cloudfront_domain=self.config.get('cloudfront_domain'),  # Pass CloudFront domain if configured
                valkey_config=self.config.get('valkey_config'),  # Pass Valkey configuration
                session_id=self._session_id,  # Pass session_id for metrics
                token_callback=self._token_accumulator_callback,  # Pass callback for token accumulation
                chart_agent=self.chart_agent  # Pass chart_agent reference for visualization
            )
            
            # If schema was extracted (not cached), store it in cache for future use with lowercase key
            if not cached_schema and 'schema_cache' in self.config:
                schema_cache[cache_key] = self._sql_agent.create_statements
        
        return self._sql_agent
    
    @property
    def kb_agent(self):
        """Lazy initialization of KB agent."""
        if self._kb_agent is None:
            self._kb_agent = kb_agent.KnowledgeBaseAgent(
                self.logger,
                self.config['kb_id'],
                self.config['aws_region'],
                self.config['bedrock_model_id'],
                self._tenant_id,
                session_id=self._session_id,  # Pass session_id for metrics
                token_callback=self._token_accumulator_callback  # Pass callback for token accumulation
            )
        return self._kb_agent
    
    @property
    def chart_agent(self):
        """Lazy initialization of Chart agent."""
        if self._chart_agent is None:
            self._chart_agent = chart_agent.ChartAgent(
                self.logger,
                self.config['aws_region'],
                self.config['bedrock_model_id'],
                self._tenant_id,
                self.config['chart_s3_bucket'],
                cloudfront_domain=self.config.get('cloudfront_domain'),
                session_id=self._session_id,  # Pass session_id for metrics
                token_callback=self._token_accumulator_callback  # Pass callback for token accumulation
            )
        return self._chart_agent
    
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
                url=self.config['gateway_url'],
                auth=auth,
                headers={
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
            4. Format all tool results appropriately for frontend display (HTML tables for tabular data, text for other content).
            5. Maintain context from previous conversation when responding to follow-up queries.
            </core_responsibilities>

            <chart_display_rules>
            CRITICAL: You MUST always check tool responses for chart data and display charts when present.

            **Chart Detection and Display Protocol:**
            1. After calling any tool, immediately inspect the response for a `chart_url` field
            2. When a `chart_url` is present in the tool response, you MUST display it - this is NOT optional
            3. Display the chart in a dedicated "Data Visualization" section immediately after the data table
            4. Use this EXACT format (replace {{chart_url}} with the actual URL from the response):
               ```html
               <h3>Data Visualization</h3>
               <img src="{{chart_url}}" alt="Data Visualization" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;" />
               ```
            5. Never skip chart display - if chart_url exists, the chart MUST appear in your response
            6. The chart should appear BEFORE the insights/summary section and AFTER the data table

            **Verification Checklist (complete before finalizing response):**
            - ✓ Did the tool response contain a chart_url field?
            - ✓ If yes, did I include the chart image in my response?
            - ✓ Is the chart displayed in the correct position (after table, before insights)?
            </chart_display_rules>

            <tool_instructions>
            ## When using **get_answers_for_structured_data**:
            - For queries about "my orders," "my products," or "my shipments," add a WHERE clause: `customers.email='{self._user_email}'`
            - Use this basic HTML table format:
               ```html
               <div style="overflow-x: auto; margin: 20px 0;">
               <table>
                 <thead>
                   <tr>
                     <th>Column 1</th>
                     <th>Column 2</th>
                     <th>Column 3</th>
                   </tr>
                 </thead>
                 <tbody>
                   <tr>
                     <td>Value 1</td>
                     <td>Value 2</td>
                     <td>Value 3</td>
                   </tr>
                 </tbody>
               </table>
               </div>
               ```
            - Show maximum {max_sql_rows} rows and include the total row count above the table
            - Include the generated SQL statement within `<details>` tags at the end

            ## When using **get_answers_for_unstructured_data**:
            - Display the retrieved information in a clear, structured format
            - Include sources, Link, pages, content_preview within `<details>` tags at the end

            ## When using **ProductReviewLambda___get_product_reviews**:
            - Pass relevant parameters based on the query:
              - product_id(s) - can be comma separated
              - customer_id(s) - can be comma separated
              - product_name(s) - can be comma separated
              - customer_name(s) - can be comma separated
              - rating(s) - can be comma separated
              - top_rows
              - review_date range
            - For queries about "my reviews," filter by passing {self._user_email} as customer_email parameter
            - Show maximum {max_dynamo_rows} rows and include the total row count
            - Format review data using the same HTML table format as structured data
            - Include all tool calls within `<details>` tags at the end
            </tool_instructions>

            <workflow>
            1. Analyze the user query to understand their intent
            2. Only answer the most recent user question unless explicitly asked to reference previous questions
            3. Determine which tool is most appropriate for addressing the query
            4. Call the selected tool with the necessary parameters
            5. Explain your thinking process while executing the tools so that the user is informed while waiting for results
            6. Format the results appropriately (HTML tables for tabular data, text for other content)
            7. Check for chart_url in the tool response and display it if present (see chart_display_rules)
            8. Add concise insights or summaries
            9. Include technical details in collapsible sections
            </workflow>

            <response_format>
            Follow this EXACT structure for every response involving data:

            **Reasoning**
            - Briefly explain how you analyzed the query and which tool you selected

            **Data Table**
            - Present tabular data using the HTML table format from tool_instructions
            - Include proper thead and tbody sections
            - Show row count above the table
            - Example:
              ```html
              <div style="overflow-x: auto; margin: 20px 0;">
              <table>
                <thead>
                  <tr>
                    <th>Header 1</th>
                    <th>Header 2</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Data 1</td>
                    <td>Data 2</td>
                  </tr>
                </tbody>
              </table>
              </div>
              ```

            **Data Visualization (MANDATORY when chart_url is present)**
            - Check the tool response for chart_url
            - When present, display using:
              ```html
              <img src="{{chart_url}}" alt="Data Visualization" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;" />
              ```
            - DO NOT SKIP THIS STEP if chart_url exists in the response

            **Key Insights**
            - Provide brief, concise insights about the data
            - Keep this section focused and actionable

            **Technical Details/ Tools Called**
            - Show SQL queries and tool call information in collapsible `<details>` sections
            - Place at the end of the response

            **Follow-up Prompt**
            - End with a relevant follow-up question based on the user's query

            IMPORTANT: Steps must appear in this exact order. Never skip Step 3 when chart_url is present in the tool response.
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
        # Import current_time here to avoid circular dependency
        from strands_tools import current_time
        
        # Check if agent already exists
        if self._orchestrator_agent is not None:
            return
        
        # Load gateway tools if MCP client is available
        gateway_tools = []
        if self.config.get('gateway_url') and self._access_token:
            # Access mcp_client property (lazy initialization)
            if self.mcp_client:
                if not self._mcp_session_active:
                    self.mcp_client.__enter__()
                    self._mcp_session_active = True
                
                gateway_tools = self._get_mcp_tools()
            else:
                self.logger.error("MCP client failed to initialize")
        else:
            if not self.config.get('gateway_url'):
                self.logger.warning("Gateway URL not configured - MCP tools will not be loaded")
            if not self._access_token:
                self.logger.warning("Access token not available - MCP tools will not be loaded")
        
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
        
        # DIAGNOSTIC: Log system prompt token count for cache validation
        prompt_length = len(system_prompt)
        estimated_tokens = prompt_length // 4  # Rough estimate: 1 token ≈ 4 characters
        self.logger.info(
            f"Orchestrator system prompt: {prompt_length} chars, ~{estimated_tokens} tokens. "
            f"Tools count: {len(all_tools)}. "
            f"Total estimated tokens (prompt + tools): ~{estimated_tokens + len(all_tools) * 100}"
        )
        
        if estimated_tokens < 1024:
            self.logger.warning(
                f"Orchestrator system prompt may be too short for prompt caching. "
                f"Estimated ~{estimated_tokens} tokens, but minimum 1,024 tokens required. "
                f"With {len(all_tools)} tools, total should exceed minimum. "
                f"If cache_read_tokens and cache_write_tokens are 0, increase prompt size."
            )
        
        # Create Bedrock model with reasoning
        bedrock_model = BedrockModel(
            model_id=self.config['bedrock_model_id'],
            #========== For Claude model ===============
            # NOTE: cache_tools parameter enables tool caching at the Strands framework level
            # This adds a cache checkpoint after the tools array in the Converse API request
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
            window_size=MAX_LAST_CONVERSATIONS,  # Limit history size
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
            # Store only session_id in agent state (serializable)
            state={"session_id": self._session_id}
        )
        
        # Store the orchestrator agent
        self._orchestrator_agent = agent
       
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._mcp_client and self._mcp_session_active:
            try:
                self._mcp_client.__exit__(None, None, None)
                self._mcp_session_active = False
            except Exception as e:
                self.logger.error(f"Error closing MCP client session: {str(e)}")


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
# TOOL DEFINITIONS
#=====================================================================================

# Import tool decorator and context
from strands import tool, ToolContext
import time


def get_app_instance():
    """Get the application instance reference."""
    global _app_instance_ref
    return _app_instance_ref


def set_app_instance(app_instance):
    """Set the application instance reference for tool functions."""
    global _app_instance_ref
    _app_instance_ref = app_instance


@tool(context=True)
def get_answers_for_structured_data(user_query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products and orders transactional data stored in the RDS Postgres database.

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
    
    logger.info(f"Tool called with session_id: {session_id}")
    
    # Look up the AgentManager from the global app instance
    app_instance = get_app_instance()
    logger.info(f"Available session_ids in agent_managers: {list(app_instance.agent_managers.keys())}")
    agent_manager = app_instance.agent_managers.get(session_id)
    
    if not agent_manager:
        logger.error(f"AgentManager not found for session_id: {session_id}")
        logger.error(f"Available sessions: {list(app_instance.agent_managers.keys())}")
        return {"message": {"content": "I'm sorry, I encountered a configuration error. Please try again later."}}

    log_conversation("User", f"user_query= {user_query}, for the user_email={agent_manager.user_email} for tenant_id={agent_manager.tenant_id}")

    try:
        start_time = time.time()
        response = agent_manager.sql_agent.process_query(user_query)
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the structured assistant")
        
        # Record semantic cache hit if it occurred
        if response.get("cache_hit", False):
            agent_manager.record_semantic_cache_hit(
                cache_tier=response.get("cache_tier", 1),
                step_name="sql_query",
                cache_hit_similarity_score=response.get("cache_hit_similarity_score", 0.0)
            )
        
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
    logger.info(f"Available session_ids in agent_managers: {list(app_instance.agent_managers.keys())}")
    agent_manager = app_instance.agent_managers.get(session_id)
    
    if not agent_manager:
        logger.error(f"AgentManager not found for session_id: {session_id}")
        logger.error(f"Available sessions: {list(app_instance.agent_managers.keys())}")
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
    logger.info(f"Available session_ids in agent_managers: {list(app_instance.agent_managers.keys())}")
    agent_manager = app_instance.agent_managers.get(session_id)
    
    if not agent_manager:
        logger.error(f"AgentManager not found for session_id: {session_id}")
        logger.error(f"Available sessions: {list(app_instance.agent_managers.keys())}")
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
