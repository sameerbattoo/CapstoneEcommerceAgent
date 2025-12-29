"""Sidebar UI components."""

import streamlit as st
import uuid
from audio_recorder_streamlit import audio_recorder


def render_sidebar(
    username: str,
    tenant_id: str,
    session_id: str,
    transcription_service,
    aws_region: str,
    user_auth = None
):
    """Render complete sidebar with user info and voice input.
    
    Args:
        username: Current username
        tenant_id: Tenant ID
        session_id: Session ID
        transcription_service: TranscriptionService instance
        aws_region: AWS region
        user_auth: UserAuth instance for logout (optional)
    """
    with st.sidebar:
        _render_user_info(username, tenant_id, session_id)
        _render_action_buttons(user_auth)
        st.markdown("---")
        _render_voice_input(transcription_service, aws_region)
        st.markdown("---")
        _render_about_section()


def _render_user_info(username: str, tenant_id: str, session_id: str):
    """Render user information section."""
    st.header("👤 User Info")
    st.write(f"**Username:** {username} (**{tenant_id}**)")
    
    # Display truncated session ID as clickable link
    truncated_session = f"{session_id[:10]}...{session_id[-10:]}"
    st.write("**Session:**")
    if st.button(
        f"🔍 {truncated_session}",
        key="session_id_btn",
        help="Click to view session memory",
        use_container_width=True
    ):
        st.session_state.show_memory_dialog = True


def _render_action_buttons(user_auth=None):
    """Render logout and clear chat buttons.
    
    Args:
        user_auth: UserAuth instance for Cognito logout (optional)
    """
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        from auth.session_manager import SessionManager
        
        # Clear authentication state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.id_token = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        
        # Clear saved session
        SessionManager.clear_session()
        
        # If using OAuth, redirect to Cognito logout to clear Cognito session
        if user_auth and hasattr(user_auth, 'get_logout_url'):
            logout_url = user_auth.get_logout_url()
            # Use HTML link with target="_self" to navigate in same window
            st.markdown(
                f'<meta http-equiv="refresh" content="0;url={logout_url}">',
                unsafe_allow_html=True
            )
            st.info("🔄 Logging out from Cognito...")
        else:
            st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


def _render_voice_input(transcription_service, aws_region: str):
    """Render voice input section."""
    st.header("🎤 Voice Input")
    st.write("Record your question using voice")
    
    # Use the audio recorder component
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button(
            "🎯 Transcribe & Execute",
            key="transcribe_send_btn",
            use_container_width=True,
            type="primary"
        ):
            try:
                with st.spinner("🔄 Transcribing..."):
                    transcribed = transcription_service.transcribe_audio(audio_bytes)
                    
                    # Automatically send the transcribed message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": transcribed
                    })
                    st.session_state.process_last_message = True
                    st.success("✅ Sent!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        if st.button("🗑️ Clear", key="clear_recording_btn", use_container_width=True):
            st.rerun()


def _render_about_section():
    """Render about section."""
    st.header("📊 About")
    st.markdown("""       
    This assistant connects to an AWS Bedrock AgentCore agent that intelligently routes your questions to:
    - **Knowledge Base** for product specs
    - **RDBMS Database** for orders and transactions
    - **no-SQL Database** for product reviews
    """)
