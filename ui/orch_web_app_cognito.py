"""Web interface for Orchestrator Agent using Streamlit with Cognito Authentication.

This is the refactored version with improved structure and separation of concerns.
"""

import streamlit as st
import uuid
import boto3

# Import configuration
from config import Settings

# Import services
from services import AgentCoreClient, TranscriptionService, MemoryService

# Import authentication
from auth import UserAuth, SessionManager

# Import UI components
from components import (
    render_login_page,
    render_sidebar,
    render_chat_messages,
    render_chat_input,
    process_agent_response,
    render_memory_dialog,
    render_sample_questions,
    apply_app_styles
)

# Import utilities
from utils import get_actor_id


def init_session_state():
    """Initialize session state variables and restore from saved session if available."""
    # Basic state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = True
    if "process_last_message" not in st.session_state:
        st.session_state.process_last_message = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Authentication state - restore from saved session if available
    if "authenticated" not in st.session_state:
        saved_session = SessionManager.load_session()
        if saved_session:
            _restore_session(saved_session)
        else:
            _init_empty_auth_state()


def _restore_session(saved_session: dict):
    """Restore authentication from saved session."""
    import jwt
    
    st.session_state.authenticated = True
    st.session_state.username = saved_session["username"].lower()
    st.session_state.access_token = saved_session["access_token"]
    st.session_state.id_token = saved_session["id_token"]
    st.session_state.refresh_token = saved_session["refresh_token"]
    
    # Decode ID token to extract tenant ID
    try:
        decoded_token = jwt.decode(
            st.session_state.id_token,
            options={"verify_signature": False}
        )
        tenant_id = decoded_token.get('custom:tenantId', 'N/A')
        st.session_state.tenant_id = tenant_id
    except Exception:
        st.session_state.tenant_id = 'N/A'


def _init_empty_auth_state():
    """Initialize empty authentication state."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.id_token = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.tenant_id = None


def main():
    """Main Streamlit application."""
    
    # Load configuration
    settings = Settings.load()
    
    # Configure page
    st.set_page_config(
        page_title="E-Commerce Assistant (Cognito)",
        page_icon="🛒",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Validate configuration
    is_valid, errors = settings.validate()
    if not is_valid:
        st.error("⚠️ Configuration errors:")
        for error in errors:
            st.error(f"  • {error}")
        st.stop()
    
    # Initialize Cognito client and UserAuth
    cognito_client = boto3.client('cognito-idp', region_name=settings.aws.region)
    user_auth = UserAuth(
        cognito_client,
        settings.cognito.client_id,
        settings.cognito.user_pool_id,
        settings.cognito.client_secret,
        settings.cognito.domain,
        settings.cognito.redirect_uri,
        settings.cognito.scopes,
        settings.aws.region
    )
    
    # Show login page if not authenticated
    if not st.session_state.authenticated:
        render_login_page(user_auth, settings.cognito.use_admin_auth)
        return
    
    # Main application (authenticated users only)
    _render_main_app(settings, user_auth)


def _render_main_app(settings: Settings, user_auth: UserAuth):
    """Render main application for authenticated users.
    
    Args:
        settings: Application settings
        user_auth: UserAuth instance
    """
    st.title("🛒 E-Commerce Assistant with AWS AgentCore")
    st.markdown(
        f"Welcome, **{st.session_state.username}**! "
        "Ask questions about products, product reviews and orders in natural language"
    )
    
    # Initialize services
    agentcore_client = AgentCoreClient(settings.aws.region, settings.aws.agentcore_arn)
    transcription_service = TranscriptionService(settings.aws.region, settings.aws.transcribe_bucket)
    memory_service = MemoryService(settings.aws.region, settings.aws.memory_id)
    
    # Render sidebar
    render_sidebar(
        st.session_state.username,
        st.session_state.tenant_id,
        st.session_state.session_id,
        transcription_service,
        settings.aws.region,
        user_auth
    )
    
    # Apply styles
    apply_app_styles()
    
    # Display chat history
    render_chat_messages(st.session_state.messages)
    
    # Process last message if triggered by example button
    if st.session_state.process_last_message and st.session_state.messages:
        _process_pending_message(agentcore_client, settings.aws.region)
    
    # Render sample questions BEFORE chat input (native Streamlit flow)
    render_sample_questions()
    
    # Chat input at the bottom (Streamlit's native position)
    actor_id = get_actor_id(st.session_state.username)
    render_chat_input(
        agentcore_client,
        st.session_state.session_id,
        st.session_state.access_token,
        st.session_state.tenant_id,
        actor_id
    )
    
    # Memory dialog
    if st.session_state.get("show_memory_dialog", False):
        actor_id = get_actor_id(st.session_state.username)
        render_memory_dialog(
            memory_service,
            st.session_state.session_id,
            actor_id,
            settings.aws.region
        )
    else:
        # If dialog is not showing but was previously open, it was closed via X button
        if st.session_state.get("dialog_is_open", False):
            st.session_state.show_memory_dialog = False
            st.session_state.dialog_is_open = False


def _process_pending_message(agentcore_client: AgentCoreClient, aws_region: str):
    """Process the last message if triggered by example button.
    
    Args:
        agentcore_client: AgentCoreClient instance
        aws_region: AWS region
    """
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        with st.chat_message("assistant"):
            actor_id = get_actor_id(st.session_state.username)
            
            answer_content, metrics_data = process_agent_response(
                last_message["content"],
                agentcore_client,
                st.session_state.session_id,
                st.session_state.access_token,
                st.session_state.tenant_id,
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
    
    st.session_state.process_last_message = False


if __name__ == "__main__":
    main()
