"""Chat interface UI components."""

import streamlit as st
import re


def render_chat_messages(messages: list):
    """Display chat message history.
    
    Args:
        messages: List of message dictionaries with role and content
    """
    for message in messages:
        thinking = message.get("thinking", None)
        _display_message(message["role"], message["content"], thinking)


def _display_message(role: str, content: str, thinking: list = None):
    """Display a single chat message with optional thinking process."""
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
        
        # Show final answer with HTML support
        st.markdown(content, unsafe_allow_html=True)


def process_agent_response(
    user_message: str,
    agentcore_client,
    session_id: str,
    access_token: str | None = None,
    tenant_id: str | None = None,
    actor_id: str | None = None
) -> str:
    """Process agent response with streaming and display thinking + answer.
    
    Args:
        user_message: The user's question
        agentcore_client: AgentCoreClient instance
        session_id: Session ID for conversation continuity
        access_token: Cognito access token (optional)
        tenant_id: Tenant ID (optional)
        actor_id: Actor ID (optional)
        
    Returns:
        Final answer content
    """
    # Create containers for streaming display
    thinking_container = st.container()
    answer_placeholder = st.empty()
    status_placeholder = st.empty()
    
    answer_content = ""
    tool_use_content = ""
    thinking_content = ""
    tool_inputs = {}
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
                        
                        else:
                            # Content event - add to answer
                            answer_content += event_data
                            answer_placeholder.markdown(answer_content + "▌", unsafe_allow_html=True)
                    
                    # Final answer without cursor
                    thinking_placeholder.info(thinking_content)
                    answer_placeholder.markdown(answer_content, unsafe_allow_html=True)
                
                # Clear status after completion
                status_placeholder.empty()
                
                return answer_content
            
            except Exception as e:
                from services.agentcore_client import UnauthorizedError
                
                status_placeholder.empty()
                
                # Handle unauthorized errors specially
                if isinstance(e, UnauthorizedError):
                    _handle_unauthorized_error()
                    return ""
                
                error_msg = f"❌ An error occurred: {str(e)}"
                answer_placeholder.error(error_msg)
                return error_msg


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
            "Its kinda cold today in California, what products can you suggest from your catalog?",
            "I am planning to gift an Electronic item. Can you suggest the two 2 products in that category based on the reviews.",
            "List my orders along with the products in Feb 2025, how did others review these?",
            "Show the distribution by product categories of my orders this year compared to last year.",
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
            answer_content = process_agent_response(
                prompt,
                agentcore_client,
                session_id,
                access_token,
                tenant_id,
                actor_id
            )
            
            # Save message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_content
            })
