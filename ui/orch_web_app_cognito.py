"""Web interface for Orchestrator Agent using Streamlit with Cognito Authentication."""

import streamlit as st
import os
import json
import urllib.parse
import uuid
import requests
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from dotenv import load_dotenv
import time
from audio_recorder_streamlit import audio_recorder
import hmac
import hashlib
import base64
import pickle
from pathlib import Path
import re

# ==========================================================================
# Session Persistence Helper
def get_session_file():
    """Get the path to the session file."""
    session_dir = Path.home() / ".streamlit_sessions"
    session_dir.mkdir(exist_ok=True)
    return session_dir / "auth_session.pkl"

def save_auth_session(username, id_token, access_token, refresh_token):
    """Save authentication session to disk."""
    try:
        session_data = {
            "username": username,
            "id_token": id_token,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "timestamp": time.time()
        }
        with open(get_session_file(), "wb") as f:
            pickle.dump(session_data, f)
    except Exception as e:
        st.warning(f"Could not save session: {e}")

def load_auth_session():
    """Load authentication session from disk."""
    try:
        session_file = get_session_file()
        if session_file.exists():
            with open(session_file, "rb") as f:
                session_data = pickle.load(f)
            
            # Check if session is less than 24 hours old
            if time.time() - session_data.get("timestamp", 0) < 86400:
                return session_data
            else:
                # Session expired, delete it
                session_file.unlink()
    except Exception:
        pass
    return None

def clear_auth_session():
    """Clear authentication session from disk."""
    try:
        session_file = get_session_file()
        if session_file.exists():
            session_file.unlink()
    except Exception:
        pass

# ==========================================================================
# State Initialization for the UI
def init_session_state():
    """Initialize session state variables and restore from saved session if available."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = True
    if "process_last_message" not in st.session_state:
        st.session_state.process_last_message = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""
    if "voice_transcribed" not in st.session_state:
        st.session_state.voice_transcribed = None
    if "last_audio_data" not in st.session_state:
        st.session_state.last_audio_data = None
    # Challenge state
    if "challenge_name" not in st.session_state:
        st.session_state.challenge_name = None
    if "challenge_session" not in st.session_state:
        st.session_state.challenge_session = None
    if "temp_username" not in st.session_state:
        st.session_state.temp_username = None
    
    # Cognito authentication state - restore from saved session if available
    if "authenticated" not in st.session_state:
        saved_session = load_auth_session()
        if saved_session:
            st.session_state.authenticated = True
            st.session_state.username = saved_session["username"]
            st.session_state.access_token = saved_session["access_token"]
            st.session_state.id_token = saved_session["id_token"]
            st.session_state.refresh_token = saved_session["refresh_token"]
        else:
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.id_token = None
            st.session_state.access_token = None
            st.session_state.refresh_token = None


# ==========================================================================
# User Authentication Class
class UserAuth:
    """Handles all Cognito authentication operations."""
    
    def __init__(self, cognito_client, client_id: str, user_pool_id: str, client_secret: str = None):
        """
        Initialize UserAuth with Cognito configuration.
        
        Args:
            cognito_client: Boto3 Cognito IDP client
            client_id: Cognito app client ID
            user_pool_id: Cognito user pool ID
            client_secret: Cognito app client secret (optional)
        """
        self.cognito_client = cognito_client
        self.client_id = client_id
        self.user_pool_id = user_pool_id
        self.client_secret = client_secret
    
    @staticmethod
    def get_secret_hash(username: str, client_id: str, client_secret: str) -> str:
        """Generate secret hash for Cognito authentication."""
        message = bytes(username + client_id, 'utf-8')
        secret = bytes(client_secret, 'utf-8')
        dig = hmac.new(secret, msg=message, digestmod=hashlib.sha256).digest()
        return base64.b64encode(dig).decode()
    
    def authenticate(self, username: str, password: str, use_admin_auth: bool = False):
        """
        Authenticate user with Cognito.
        
        Args:
            username: User's username
            password: User's password
            use_admin_auth: Use ADMIN_NO_SRP_AUTH flow instead of USER_PASSWORD_AUTH
            
        Returns:
            Authentication result or None if failed
        """
        try:
            auth_params = {
                'USERNAME': username,
                'PASSWORD': password
            }
            
            if self.client_secret:
                auth_params['SECRET_HASH'] = self.get_secret_hash(
                    username, self.client_id, self.client_secret
                )
            
            if use_admin_auth:
                response = self.cognito_client.admin_initiate_auth(
                    UserPoolId=self.user_pool_id,
                    ClientId=self.client_id,
                    AuthFlow='ADMIN_NO_SRP_AUTH',
                    AuthParameters=auth_params
                )
            else:
                response = self.cognito_client.initiate_auth(
                    ClientId=self.client_id,
                    AuthFlow='USER_PASSWORD_AUTH',
                    AuthParameters=auth_params
                )
            
            return response
        except Exception as e:
            return None
    
    def respond_to_challenge(self, challenge_name: str, session: str, username: str, new_password: str):
        """
        Respond to authentication challenge.
        
        Args:
            challenge_name: Name of the challenge
            session: Challenge session token
            username: User's username
            new_password: New password for NEW_PASSWORD_REQUIRED challenge
            
        Returns:
            Challenge response or None if failed
        """
        try:
            challenge_responses = {
                'USERNAME': username,
                'NEW_PASSWORD': new_password
            }
            
            if self.client_secret:
                challenge_responses['SECRET_HASH'] = self.get_secret_hash(
                    username, self.client_id, self.client_secret
                )
            
            response = self.cognito_client.respond_to_auth_challenge(
                ClientId=self.client_id,
                ChallengeName=challenge_name,
                Session=session,
                ChallengeResponses=challenge_responses
            )
            
            return response
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Challenge response failed: {str(e)}")
            with st.expander("🔍 Debug Details"):
                st.code(error_details)
            return None
    
    def refresh_tokens(self, refresh_token: str, username: str):
        """
        Refresh authentication tokens.
        
        Args:
            refresh_token: Refresh token
            username: Username for secret hash generation
            
        Returns:
            New tokens or None if failed
        """
        try:
            auth_params = {
                'REFRESH_TOKEN': refresh_token
            }
            
            if self.client_secret:
                auth_params['SECRET_HASH'] = self.get_secret_hash(
                    username, self.client_id, self.client_secret
                )
            
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='REFRESH_TOKEN_AUTH',
                AuthParameters=auth_params
            )
            
            return response
        except Exception as e:
            st.error(f"Token refresh failed: {str(e)}")
            return None
    
    @staticmethod
    def logout():
        """Clear authentication state and logout user."""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.id_token = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        
        # Clear saved session
        clear_auth_session()


# ==========================================================================
# Other helper functions
def display_message(role: str, content: str, thinking: list = None):
    """Display a chat message with optional thinking process."""
    with st.chat_message(role):
        # Show thinking process if available
        if thinking and len(thinking) > 0:
            with st.expander("🤔 Agent Thinking Process", expanded=False):
                for thought in thinking:
                    thought_type = thought.get("type", "unknown")
                    thought_data = thought.get("data", "")
                    
                    if thought_type == "tool_use":
                        st.info(f"🔧 **Using Tool**")
                        st.code(thought_data, language="text")
                    elif thought_type == "tool_result":
                        st.success(f"✅ **Tool Result**")
                        st.code(thought_data, language="text")
                    elif thought_type == "thinking":
                        st.write(f"🧠 {thought_data}")
                    else:
                        st.write(thought_data)
        
        # Show final answer
        # Use st.markdown with unsafe_allow_html=True to render HTML elements like <details>
        # This allows collapsible sections and other HTML formatting from the agent
        st.markdown(content, unsafe_allow_html=True)


# ==========================================================================
# Helper function to process and display agent response
def process_agent_response(user_message: str, agent_arn: str, session_id: str, region: str, access_token: str = None):
    """
    Process agent response with streaming and display thinking + answer.
    
    Args:
        user_message: The user's question
        agent_arn: The ARN of the deployed AgentCore agent
        session_id: The session ID for conversation continuity
        region: AWS region
        access_token: Cognito access token for Bearer authentication (optional)
        
    Returns:
        Tuple of (answer_content, thinking_events) or (error_msg, None) on error
    """
    # Create container for thinking process at the top
    thinking_container = st.container()
    answer_placeholder = st.empty()
    status_placeholder = st.empty()
    
    answer_content = ""
    tool_use_content = ""
    thinking_content = ""
    tool_inputs = {}          # tool_name -> latest full Input: string
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
                    for event in invoke_agentcore_agent(
                        user_message,
                        agent_arn,
                        session_id,
                        region,
                        access_token
                    ):
                        event_type = event.get("type", "content")
                        event_data = event.get("data", "")

                        #print(f"event_type: {event_type}, event_data: {event_data}")

                        # Handle different event types - update placeholders in real-time
                        if event_type == "tool_use":

                            m = tool_re.match(event_data.strip()) # Use regular expression
                            #Get the tool name and the Input fragment
                            tool_name, tool_input_fragment = m.group(1), m.group(2)

                            # Always store the latest full line for each tool
                            tool_inputs[tool_name] = f'Tool: {tool_name}\nInput: {tool_input_fragment}'

                            # Rebuild current_block from all tools seen so far, in insertion order
                            # (Python 3.7+ dict preserves insertion order)
                            tool_use_content = "\n".join(tool_inputs.values())

                            tool_use_placeholder.code(tool_use_content)
                            
                        elif event_type == "thinking":
                            thinking_content += event_data
                            thinking_placeholder.info(thinking_content + "▌")
                                                         
                        else:
                            # Content event - add to answer
                            answer_content += event_data
                            # Use markdown with unsafe_allow_html to render HTML elements
                            answer_placeholder.markdown(answer_content + "▌", unsafe_allow_html=True)
                    
                    # Final answer without cursor
                    thinking_placeholder.info(thinking_content)
                    answer_placeholder.markdown(answer_content, unsafe_allow_html=True)
                
                # Clear status after completion
                status_placeholder.empty()
        
                return answer_content
        
            except Exception as e:
                status_placeholder.empty()
                error_msg = f"❌ An error occurred: {str(e)}"
                answer_placeholder.error(error_msg)
                return error_msg, None


# ==========================================================================
# Helper function for AWS Transcribe functionality
def transcribe_audio_with_aws(audio_bytes: bytes, region: str, s3_bucket: str) -> str:
    """
    Transcribe audio using AWS Transcribe.
    
    Args:
        audio_bytes: Audio data in bytes
        region: AWS region
        s3_bucket: S3 bucket name for temporary storage
        
    Returns:
        Transcribed text
    """
    s3_client = boto3.client('s3', region_name=region)
    transcribe_client = boto3.client('transcribe', region_name=region)
    
    # Generate unique identifiers
    object_key = f"voice-input/{uuid.uuid4().hex}.webm"
    job_name = f"transcribe-{uuid.uuid4().hex}"
    
    try:
        # Upload audio to S3
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=object_key,
            Body=audio_bytes,
            ContentType='audio/webm'
        )
        
        # Start transcription job
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': f's3://{s3_bucket}/{object_key}'},
            MediaFormat='webm',
            LanguageCode='en-US'
        )
        
        # Wait for completion (with timeout)
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status == 'COMPLETED':
                # Get transcript
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                transcript_response = requests.get(transcript_uri)
                transcript_data = transcript_response.json()
                text = transcript_data['results']['transcripts'][0]['transcript']
                
                # Cleanup
                transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
                s3_client.delete_object(Bucket=s3_bucket, Key=object_key)
                
                return text
            elif job_status == 'FAILED':
                failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                raise Exception(f"Transcription failed: {failure_reason}")
            
            time.sleep(1)
        
        # Timeout - cleanup and raise error
        transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        s3_client.delete_object(Bucket=s3_bucket, Key=object_key)
        raise Exception("Transcription timed out after 30 seconds")
        
    except Exception as e:
        # Cleanup on error
        try:
            s3_client.delete_object(Bucket=s3_bucket, Key=object_key)
        except:
            pass
        raise e


def invoke_agentcore_agent(user_input: str, agent_arn: str, session_id: str, region: str, access_token: str = None):
    """
    Invoke the AWS Bedrock AgentCore agent and stream the response using requests.
    
    Args:
        user_input: The user's question
        agent_arn: The ARN of the deployed AgentCore agent
        session_id: The session ID for conversation continuity
        region: AWS region
        access_token: Cognito access token for Bearer authentication (optional)
        
    Yields:
        Structured events: {"type": "thinking"|"content"|"tool_use"|"tool_result", "data": ...}
    """
    # Prepare the URL
    escaped_arn = urllib.parse.quote(agent_arn, safe="")
    url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{escaped_arn}/invocations"
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
    }

    # Add Bearer token if provided (Cognito authentication)
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    else:
        # Fall back to AWS SigV4 signing if no token provided
        session = boto3.Session()
        credentials = session.get_credentials()
    
    # Prepare the payload
    payload = {
        "user_input": user_input,
        "session_id": session_id
    }
    
    body = json.dumps(payload)
    
    try:
        # If using Bearer token, make direct request
        if access_token:
            response = requests.post(
                url,
                params={"qualifier": "DEFAULT"},
                headers=headers,
                data=body,
                timeout=100,
                stream=True,
            )
        else:
            # Use SigV4 signing for IAM-based authentication
            request = AWSRequest(
                method='POST',
                url=url + '?qualifier=DEFAULT',
                data=body,
                headers=headers
            )
            
            # Sign the request with SigV4
            SigV4Auth(credentials, 'bedrock-agentcore', region).add_auth(request)
            
            # Make the actual HTTP request with streaming
            response = requests.post(
                url,
                params={"qualifier": "DEFAULT"},
                headers=dict(request.headers),
                data=body,
                timeout=100,
                stream=True,
            )
        
        # Check for errors
        response.raise_for_status()
        
        # Stream the response - parse Server-Sent Events format
        current_event_type = None
        buffer = ""  # Buffer to accumulate partial markers
        
        for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
            if line:
                # Parse event type
                if line.startswith("event: "):
                    current_event_type = line[7:].strip()
                # Parse data
                elif line.startswith("data: "):
                    data = line[6:].strip()
                    
                    # Remove surrounding quotes and unescape JSON string
                    if data.startswith('"') and data.endswith('"'):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            data = data[1:-1]

                    if data: # Check to see ff there is data
                        data_stripped = data.lstrip("\r\n")
                        if data_stripped.startswith("[TOOL USE]"): # then its a tool_use message
                            yield {"type": "tool_use", "data": data_stripped.removeprefix("[TOOL USE]")}
                        elif data_stripped.startswith("[THINKING]"): # then its a thinking message
                            yield {"type": "thinking", "data": data_stripped.removeprefix("[THINKING]")}
                        else:
                            yield {"type": "content", "data": data}

                    
    except requests.exceptions.RequestException as e:
        import traceback
        error_details = traceback.format_exc()
        
        # Check if it's a 401 Unauthorized error
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 401:
            # Clear authentication and force re-login
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.id_token = None
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            clear_auth_session()
            
            # Show error message and rerun to show login page
            st.error("🔒 Your session has expired. Please login again.")
            time.sleep(2)
            st.rerun()
        
        raise Exception(f"Error invoking AgentCore agent: {str(e)}\n{error_details}")

# ===========================================================================
# Login Page UI
def show_login_page(user_auth: UserAuth, use_admin_auth: bool = False):
    """Display login page."""
    
    # Check if we're in a challenge state
    if st.session_state.challenge_name == 'NEW_PASSWORD_REQUIRED':
        st.title("🔑 Set New Password")
        st.markdown(f"Welcome **{st.session_state.temp_username}**! You need to set a new password.")
        
        with st.form("new_password_form"):
            new_password = st.text_input("New Password", type="password", 
                                        help="Password must meet your organization's requirements")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submit = st.form_submit_button("Set Password", use_container_width=True, type="primary")
            
            if submit:
                if not new_password or not confirm_password:
                    st.error("Please enter and confirm your new password")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Setting new password..."):
                        result = user_auth.respond_to_challenge(
                            st.session_state.challenge_name,
                            st.session_state.challenge_session,
                            st.session_state.temp_username,
                            new_password
                        )
                        
                        if result and 'AuthenticationResult' in result:
                            st.session_state.authenticated = True
                            st.session_state.username = st.session_state.temp_username
                            st.session_state.id_token = result['AuthenticationResult']['IdToken']
                            st.session_state.access_token = result['AuthenticationResult']['AccessToken']
                            st.session_state.refresh_token = result['AuthenticationResult'].get('RefreshToken')
                            
                            # Save session for persistence across page refreshes
                            save_auth_session(
                                st.session_state.username,
                                st.session_state.id_token,
                                st.session_state.access_token,
                                st.session_state.refresh_token
                            )
                            
                            # Clear challenge state
                            st.session_state.challenge_name = None
                            st.session_state.challenge_session = None
                            st.session_state.temp_username = None
                            st.success("✅ Password set successfully!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Failed to set new password")
        
        if st.button("← Back to Login"):
            st.session_state.challenge_name = None
            st.session_state.challenge_session = None
            st.session_state.temp_username = None
            st.rerun()
        
        return
    
    # Normal login page
    st.title("🔐 Login")
    st.markdown("Please login to access the E-Commerce Assistant")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True, type="primary")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                with st.spinner("Authenticating..."):
                    result = user_auth.authenticate(username, password, use_admin_auth)
                    
                    if result:
                        if 'AuthenticationResult' in result:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.id_token = result['AuthenticationResult']['IdToken']
                            st.session_state.access_token = result['AuthenticationResult']['AccessToken']
                            st.session_state.refresh_token = result['AuthenticationResult'].get('RefreshToken')
                            
                            # Save session for persistence across page refreshes
                            save_auth_session(
                                username,
                                st.session_state.id_token,
                                st.session_state.access_token,
                                st.session_state.refresh_token
                            )
                            
                            st.success("✅ Login successful!")
                            time.sleep(0.5)
                            st.rerun()
                        elif 'ChallengeName' in result:
                            if result['ChallengeName'] == 'NEW_PASSWORD_REQUIRED':
                                # Store challenge info and show password change form
                                st.session_state.challenge_name = result['ChallengeName']
                                st.session_state.challenge_session = result['Session']
                                st.session_state.temp_username = username
                                st.info("🔑 You need to set a new password")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.warning(f"⚠️ Authentication challenge required: {result['ChallengeName']}")
                                st.info("This challenge type is not yet supported. Please contact your administrator.")
                        else:
                            st.error("❌ Unexpected authentication response")
                            with st.expander("🔍 Debug Response"):
                                st.json(result)
                    else:
                        st.error("❌ Authentication failed - Incorrect username or password.")
    
    st.markdown("---")
    st.info("💡 If you don't have an account, please contact your administrator")

# ===========================================================================
# Main entry point for the UI
def main():
    """Main Streamlit application."""

    load_dotenv()  # Load environment variables

    st.set_page_config(
        page_title="E-Commerce Assistant (Cognito)",
        page_icon="🛒",
        layout="wide"
    )
    
    # Initialize session state (will restore from saved session if available)
    init_session_state()
    
    # Get Cognito configuration from environment
    user_pool_id = os.getenv('COGNITO_USER_POOL_ID')
    client_id = os.getenv('COGNITO_CLIENT_ID')
    client_secret = os.getenv('COGNITO_CLIENT_SECRET')  # Optional
    aws_region = os.getenv('AWS_REGION')
    # Set to 'true' if USER_PASSWORD_AUTH is not enabled on your app client
    use_admin_auth = os.getenv('COGNITO_USE_ADMIN_AUTH', 'false').lower() == 'true'
    
    if not user_pool_id or not client_id:
        st.error("⚠️ Cognito configuration missing. Please set COGNITO_USER_POOL_ID and COGNITO_CLIENT_ID in .env file")
        st.stop()
    
    # Initialize Cognito client and UserAuth
    cognito_client = boto3.client('cognito-idp', region_name=aws_region)
    user_auth = UserAuth(cognito_client, client_id, user_pool_id, client_secret)
    
    # Show login page if not authenticated
    if not st.session_state.authenticated:
        show_login_page(user_auth, use_admin_auth)
        return
    
    # Main application (authenticated users only)
    st.title("🛒 E-Commerce Assistant with AWS AgentCore")
    st.markdown(f"Welcome, **{st.session_state.username}**! Ask questions about products, product reviews and orders in natural language")
    
    # Get agent configuration from environment
    agent_arn = os.getenv('AGENTCORE_ARN_COGNITO')
    
    if not agent_arn:
        st.error("⚠️ AGENTCORE_ARN_COGNITO not configured in .env file")
        st.stop()
    
    # S3 bucket for voice transcription
    s3_bucket = os.getenv('TRANSCRIBE_S3_BUCKET', 'capstone-voice-recordings')
    
    # Sidebar with info
    with st.sidebar:
        st.header("👤 User Info")
        st.write(f"**Username:** {st.session_state.username}")
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            UserAuth.logout()
            st.rerun()
        
        st.markdown("---")
        
        st.header("📊 About")
        st.markdown("""       
        This assistant connects to an AWS Bedrock AgentCore agent that intelligently routes your questions to:
        - **Knowledge Base** for product specs
        - **RDBMS Database** for orders and transactions
        - **no-SQL Database** for product reviews
        """)
        
        # Voice input in sidebar
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
            
            if st.button("🎯 Transcribe & Execute", key="transcribe_send_btn", use_container_width=True, type="primary"):
                try:
                    with st.spinner("🔄 Transcribing..."):
                        transcribed = transcribe_audio_with_aws(audio_bytes, aws_region, s3_bucket)
                        
                        # Automatically send the transcribed message
                        st.session_state.messages.append({
                            "role": "user",
                            "content": transcribed
                        })
                        st.session_state.process_last_message = True
                        st.success(f"✅ Sent!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            if st.button("🗑️ Clear", key="clear_recording_btn", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        st.header("💡 Example Queries")
        examples = [
            "Its kinda cold today in California, what products can you suggest from your catalog?",
            "I am planning to gift an Electronic item. Can you suggest the two 2 products in that category based on the reviews",
            "List my orders along with the products in Feb 2025, how did others review these?",
            "Show the distribution by product categories of my orders this year compared to last year",
            "Display my review history with ratings. Did people find my reviews useful?",
            "List the top 5 user by sales in 2025 and show their product review summary for each of them",
            "Which customers have spent more than $1000 in the 2nd Quarter of 2025? Did they write any reviews?",
            "What are the specifications of the Winter Jacket and also show the review sentiments",
            "What colors are available for the Cotton T-Shirt and also how many people bought it in Q2 2025?",
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
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        thinking = message.get("thinking", None)
        display_message(message["role"], message["content"], thinking)
    
    # Process last message if triggered by example button
    if st.session_state.process_last_message and st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant"):
                answer_content = process_agent_response(
                    last_message["content"],
                    agent_arn,
                    st.session_state.session_id,
                    aws_region,
                    st.session_state.access_token
                )
                             
                # Save message with thinking
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_content
                })
        
        st.session_state.process_last_message = False
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about products, orders and reviews ..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        # Get agent response with streaming
        with st.chat_message("assistant"):
            answer_content = process_agent_response(
                prompt,
                agent_arn,
                st.session_state.session_id,
                aws_region,
                st.session_state.access_token
            )
            
           # Save message with thinking
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_content
            })


if __name__ == "__main__":
    main()
