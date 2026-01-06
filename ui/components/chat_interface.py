"""Chat interface UI components."""

import streamlit as st
import re
import pandas as pd
from io import StringIO
import json


def _render_metrics(metrics: dict):
    """Render metrics in an expander.
    
    Args:
        metrics: Dictionary containing duration, tokens, cost, cache info, and model info
    """
    with st.expander("💰 Bedrock Token / Cost Info (not including the AgentCore Runtime, Gateway, Memory or Code Interpreter costs)", expanded=True):
        # Add custom CSS to style metrics
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 0.9em !important;
                color: #1f77b4 !important;
                font-weight: bold !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.9em !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # First row: Main token metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Initialize all metrics with defaults to prevent KeyError
        input_tokens = metrics.get('input_tokens', 0)
        output_tokens = metrics.get('output_tokens', 0)
        cost_usd = metrics.get('cost_usd', 0.0)
        
        with col1:
            st.metric(
                label="Total Tokens",
                value=f"{input_tokens + output_tokens:,}"
            )
        
        with col2:
            st.metric(
                label="Input Tokens",
                value=f"{input_tokens:,}"
            )
        
        with col3:
            st.metric(
                label="Output Tokens",
                value=f"{output_tokens:,}"
            )
        
        with col4:
            st.metric(
                label="Estimated Cost",
                value=f"${cost_usd:.4f}"
            )
        
        # Second row: Cache metrics (if cache tokens exist)
        cache_read = metrics.get('cache_read_tokens', 0)
        cache_write = metrics.get('cache_write_tokens', 0)
        cache_savings = metrics.get('cache_savings_usd', 0.0)
        
        if cache_read > 0 or cache_write > 0 or cache_savings > 0:
            st.markdown("---")  # Divider
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    label="Cache Read Tokens",
                    value=f"{cache_read:,}",
                    help="Tokens read from cache (90% cost savings)"
                )
            
            with col6:
                st.metric(
                    label="Cache Write Tokens",
                    value=f"{cache_write:,}",
                    help="Tokens written to cache (25% cost premium)"
                )
            
            with col7:
                # Calculate cache efficiency
                total_input = input_tokens + cache_read
                cache_efficiency = (cache_read / total_input * 100) if total_input > 0 else 0
                st.metric(
                    label="Cache Hit Rate",
                    value=f"{cache_efficiency:.1f}%",
                    help="Percentage of input tokens served from cache"
                )
            
            with col8:
                st.metric(
                    label="Cache Savings",
                    value=f"${cache_savings:.4f}",
                    help="Cost saved by using prompt caching",
                    delta=f"-{cache_savings:.4f}" if cache_savings > 0 else None
                )
        
        # Additional details in smaller text with blue bold steps count
        st.markdown(
            f"<span style='font-size: 0.8em; color: #888;'>Model: <code>{metrics.get('model_id', 'unknown')}</code> • Steps: <span style='color: #1f77b4; font-weight: bold;'>{metrics.get('step_count', 'N/A')}</span></span>",
            unsafe_allow_html=True
        )
        
        # Display semantic cache hits if any
        semantic_cache_hits = metrics.get('semantic_cache_hits', [])
        if semantic_cache_hits:
            # Build cache hit message
            cache_messages = []
            for hit in semantic_cache_hits:
                tier = hit.get('tier', 1)
                tier_label = "High Sematic Similarity" if tier == 1 else "SQL Similarity"
                # Format similarity score to 3 decimal places
                similarity_score = hit.get('cache_hit_similarity_score', 0.0)
                tier_label += f" (Similarity Score: {similarity_score:.3f})"
                cache_messages.append(f"Tier {tier} ({tier_label})")
            
            cache_text = ", ".join(cache_messages)
            
            # Use default background with green text and border (no white background)
            st.markdown(
                f"<div style='margin-top: 8px; padding: 8px; border-left: 4px solid #4caf50; border-radius: 4px;'>"
                f"<span style='font-size: 0.85em; color: #4caf50;'>⚡ <strong>Semantic Cache Hit:</strong> {cache_text} - Query results served from Valkey cache</span>"
                f"</div>",
                unsafe_allow_html=True
            )


def render_chat_messages(messages: list):
    """Display chat message history.
    
    Args:
        messages: List of message dictionaries with role and content
    """
    for message in messages:
        thinking = message.get("thinking", None)
        metrics = message.get("metrics", None)
        _display_message(message["role"], message["content"], thinking, metrics)


def _extract_and_render_tables(content: str) -> str:
    """
    Extract HTML tables from content and convert them to interactive Streamlit dataframes.
    
    Args:
        content: The message content that may contain HTML tables
        
    Returns:
        Modified content with table placeholders
    """
    # Pattern to match HTML tables wrapped in div
    table_pattern = r'<div style="overflow-x: auto;[^>]*>.*?<table[^>]*>.*?</table>.*?</div>'
    
    tables = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not tables:
        return content
    
    # Process each table
    modified_content = content
    for idx, table_html in enumerate(tables):
        try:
            # Parse HTML table with pandas (lxml parser)
            df = pd.read_html(StringIO(table_html))[0]
            
            # Create unique key for this table
            table_key = f"table_{idx}"
            
            # Replace HTML table with placeholder
            placeholder = f"\n\n__STREAMLIT_TABLE_{table_key}__\n\n"
            modified_content = modified_content.replace(table_html, placeholder, 1)
            
            # Store dataframe in session state
            if 'extracted_tables' not in st.session_state:
                st.session_state.extracted_tables = {}
            st.session_state.extracted_tables[table_key] = df
            
        except Exception as e:
            # If parsing fails, keep original HTML table
            print(f"Warning: Could not parse table {idx}: {str(e)}")
            continue
    
    return modified_content


def _render_content_with_tables(content: str):
    """
    Render content with embedded Streamlit dataframes where tables were detected.
    
    Args:
        content: Content with table placeholders
    """
    # Split content by table placeholders
    # Pattern matches: __STREAMLIT_TABLE_table_123__
    table_placeholder_pattern = r'__STREAMLIT_TABLE_(table_\d+)__'
    parts = re.split(table_placeholder_pattern, content)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular content - render as markdown
            if part.strip():
                st.markdown(part, unsafe_allow_html=True)
        else:
            # Table placeholder - render as dataframe
            # part contains the table_key (e.g., "table_0")
            table_key = part
            if 'extracted_tables' in st.session_state and table_key in st.session_state.extracted_tables:
                df = st.session_state.extracted_tables[table_key]
                
                # Render with Streamlit's native dataframe (includes download button)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, (len(df) + 1) * 35 + 3)  # Dynamic height based on rows
                )


def _display_message(role: str, content: str, thinking: list = None, metrics: dict = None):
    """Display a single chat message with optional thinking process and metrics."""
    with st.chat_message(role):
        # Show thinking process if available
        if thinking and len(thinking) > 0:
            with st.expander("🤔 Agent Thinking Process", expanded=False):
                for thought in thinking:
                    thought_type = thought.get("type", "unknown")
                    thought_data = thought.get("data", "")
                    
                    if thought_type == "tool_use":
                        st.info("🔧 **Using Tool**")
                        st.code(thought_data, language="text")
                    elif thought_type == "tool_result":
                        st.success("✅ **Tool Result**")
                        st.code(thought_data, language="text")
                    elif thought_type == "thinking":
                        st.write(f"🧠 {thought_data}")
                    else:
                        st.write(thought_data)
        
        # Extract tables and render with interactive dataframes
        if role == "assistant":
            modified_content = _extract_and_render_tables(content)
            _render_content_with_tables(modified_content)
        else:
            # User messages - render as-is
            st.markdown(content, unsafe_allow_html=True)
        
        # Show metrics if available (for assistant messages)
        if role == "assistant" and metrics:
            _render_metrics(metrics)


def process_agent_response(
    user_message: str,
    agentcore_client,
    session_id: str,
    access_token: str | None = None,
    tenant_id: str | None = None,
    actor_id: str | None = None
) -> tuple[str, dict]:
    """Process agent response with streaming and display thinking + answer.
    
    Args:
        user_message: The user's question
        agentcore_client: AgentCoreClient instance
        session_id: Session ID for conversation continuity
        access_token: Cognito access token (optional)
        tenant_id: Tenant ID (optional)
        actor_id: Actor ID (optional)
        
    Returns:
        Tuple of (final answer content, metrics data)
    """
    # Create containers for streaming display
    thinking_container = st.container()
    answer_placeholder = st.empty()
    metrics_container = st.container()  # Separate container for metrics
    status_placeholder = st.empty()
    
    answer_content = ""
    tool_use_content = ""
    thinking_content = ""
    tool_inputs = {}
    metrics_data = None
    tool_re = re.compile(r"^Tool:\s*(.+?)\s*Input:\s*(.*)$", re.DOTALL)
    
    # Create the thinking expander at the start
    with thinking_container:
        thinking_expander = st.expander("🤔 Agent Thinking Process", expanded=False)
        with thinking_expander:
            thinking_placeholder = st.empty()
            tool_use_placeholder = st.empty()
            
            try:
                # Show spinner with working message
                with status_placeholder.status("Working...", expanded=False) as status:
                    st.write("Processing your request via AWS AgentCore...")
                    
                    # Stream response from AgentCore
                    for event in agentcore_client.invoke_streaming(
                        user_message,
                        session_id,
                        access_token,
                        tenant_id,
                        actor_id
                    ):
                        event_type = event.get("type", "content")
                        event_data = event.get("data", "")
                        
                        # Handle different event types
                        if event_type == "tool_use":
                            m = tool_re.match(event_data.strip())
                            if m:
                                tool_name, tool_input_fragment = m.group(1), m.group(2)
                                tool_inputs[tool_name] = f'Tool: {tool_name}\nInput: {tool_input_fragment}'
                                tool_use_content = "\n".join(tool_inputs.values())
                                tool_use_placeholder.code(tool_use_content)
                        
                        elif event_type == "thinking":
                            thinking_content += event_data
                            thinking_placeholder.info(thinking_content + "▌")
                        
                        elif event_type == "metrics":
                            # Parse and store metrics data
                            try:
                                metrics_data = json.loads(event_data.strip())
                            except json.JSONDecodeError:
                                pass  # Ignore malformed metrics
                        
                        else:
                            # Content event - add to answer
                            answer_content += event_data
                            answer_placeholder.markdown(answer_content + "▌", unsafe_allow_html=True)
                    
                    # Final answer without cursor
                    thinking_placeholder.info(thinking_content)
                    
                    # Extract tables and render with interactive dataframes
                    modified_content = _extract_and_render_tables(answer_content)
                    answer_placeholder.empty()  # Clear the placeholder
                    
                    # Render content with tables in a new container
                    with answer_placeholder.container():
                        _render_content_with_tables(modified_content)
                
                # Clear status after completion
                status_placeholder.empty()
                
                # Display metrics OUTSIDE the thinking container, after the answer
                if metrics_data:
                    with metrics_container:
                        _render_metrics(metrics_data)
                
                return answer_content, metrics_data
            
            except Exception as e:
                from services.agentcore_client import UnauthorizedError
                
                status_placeholder.empty()
                
                # Handle unauthorized errors specially
                if isinstance(e, UnauthorizedError):
                    _handle_unauthorized_error()
                    return "", None
                
                error_msg = f"❌ An error occurred: {str(e)}"
                answer_placeholder.error(error_msg)
                return error_msg, None


def _handle_unauthorized_error():
    """Handle unauthorized error by clearing session and forcing re-login."""
    from auth.session_manager import SessionManager
    import time
    
    # Clear authentication
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.id_token = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    SessionManager.clear_session()
    
    # Show error and rerun
    st.error("🔒 Your session has expired. Please login again.")
    time.sleep(2)
    st.rerun()


def render_sample_questions():
    """Render sample questions panel using native Streamlit components."""
    st.markdown("---")  # Visual separator
    
    with st.expander("💡 Sample Questions - Click to try", expanded=False):
        st.markdown("**Try these example questions:**")
        
        examples = [
            "Analyse my spend for the current year vs last per product category",
            "I am planning to gift an Electronic item. Can you suggest the two 2 products in that category based on the reviews.",
            "List my orders along with the products for Feb 2025, how did others review these?",
            "Show the distribution by product categories of my orders this year compared to other shoppers.",
            "Display my review history with ratings. Did people find my reviews useful?",
            "List the top 5 user by sales in 2025 and show their product review summary for each of them.",
            "Which customers have spent more than $1000 in the 2nd Quarter of 2025? Did they write any reviews?",
            "What are the specifications of the Winter Jacket? And also show the review sentiments.",
            "What colors are available for the Cotton T-Shirt and also how many people bought it in Q2 2025?",
            "What other sample questions can I ask?",
        ]
        
        # Create 2 columns for better layout
        col1, col2 = st.columns(2)
        
        for idx, example in enumerate(examples):
            # Alternate between columns
            col = col1 if idx % 2 == 0 else col2
            
            with col:
                # Truncate to 80 characters for display
                display_text = example if len(example) <= 150 else example[:144] + "..."
                if st.button(
                    display_text,
                    key=f"sample_q_{idx}",
                    help=example,
                    use_container_width=True
                ):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.session_state.process_last_message = True
                    st.rerun()


def render_chat_input(
    agentcore_client,
    session_id: str,
    access_token: str,
    tenant_id: str,
    actor_id: str
):
    """Render chat input and handle user messages.
    
    Args:
        agentcore_client: AgentCoreClient instance
        session_id: Session ID
        access_token: Access token
        tenant_id: Tenant ID
        actor_id: Actor ID
    """
    if prompt := st.chat_input("Ask me anything about products, orders and reviews ..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        _display_message("user", prompt)
        
        # Get agent response with streaming
        with st.chat_message("assistant"):
            answer_content, metrics_data = process_agent_response(
                prompt,
                agentcore_client,
                session_id,
                access_token,
                tenant_id,
                actor_id
            )
            
            # Update session cost if metrics available
            if metrics_data and 'cost_usd' in metrics_data:
                if 'session_cost' not in st.session_state:
                    st.session_state.session_cost = 0.0
                st.session_state.session_cost += metrics_data['cost_usd']
            
            # Save message with metrics
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_content,
                "metrics": metrics_data
            })
