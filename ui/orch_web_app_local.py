"""Web interface for Orchestrator Agent using Streamlit."""

import streamlit as st
import os
import asyncio
from agent.orch_agent_local import process_user_input
from dotenv import load_dotenv


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = True
    if "process_last_message" not in st.session_state:
        st.session_state.process_last_message = False


def display_message(role: str, content: str):
    """Display a chat message."""
    with st.chat_message(role):
        st.markdown(content,unsafe_allow_html=True)


def main():
    """Main Streamlit application."""

    load_dotenv()  # By default, loads .env in the current working directory

    st.set_page_config(
        page_title="E-Commerce Assistant",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 E-Commerce Assistant with AWS Strands")
    st.markdown("Ask questions about products and orders in natural language")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar with info
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""       
        The agent intelligently routes your questions to:
        - **Knowledge Base** for product specs
        - **Database** for orders and transactions
        """)
        
        st.header("🗄️ Data Sources")
        st.info(f"""
        **Database:** {os.getenv('DB_NAME', 'Not confugured')}  
        **Knowledge Base ID:** {os.getenv('KB_ID', 'Not configured')}  
        **AWS Region:** {os.getenv('AWS_REGION', 'Not configured')}
        """)
        
        st.header("💡 Example Queries")
        examples = [
            "What are the specifications of the Winter Jacket?",
            "List the top 5 products by sales",
            "Tell me about the Smartphone X features",
            "Which customers have spent more than $1000?",
            "What colors are available for the Cotton T-Shirt?",
            "What other sample questions can I ask?",
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.session_state.process_last_message = True
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])
    
    # Process last message if triggered by example button
    if st.session_state.process_last_message and st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Show spinner with working message
                    with status_placeholder.status("Working...", expanded=False):
                        st.write("Processing your request...")
                        
                        # Stream response with real-time updates
                        async def stream_response():
                            nonlocal full_response
                            async for chunk in process_user_input(last_message["content"]):
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                            message_placeholder.markdown(full_response, unsafe_allow_html=True)
                        
                        # Run async function
                        asyncio.run(stream_response())
                    
                    # Clear status after completion
                    status_placeholder.empty()
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    status_placeholder.empty()
                    error_msg = f"❌ An error occurred: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.process_last_message = False
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about products or orders..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        # Get agent response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show spinner with working message
                with status_placeholder.status("Working...", expanded=False):
                    st.write("Processing your request...")
                    
                    # Stream response with real-time updates
                    async def stream_response():
                        nonlocal full_response
                        async for chunk in process_user_input(prompt):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)
                    
                    # Run async function
                    asyncio.run(stream_response())
                
                # Clear status after completion
                status_placeholder.empty()
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                status_placeholder.empty()
                error_msg = f"❌ An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
