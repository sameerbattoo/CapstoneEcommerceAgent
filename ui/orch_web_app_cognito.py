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


def get_secret_hash(username: str, client_id: str, client_secret: str) -> str:
    """Generate secret hash for Cognito authentication."""
    message = bytes(username + client_id, 'utf-8')
    secret = bytes(client_secret, 'utf-8')
    dig = hmac.new(secret, msg=message, digestmod=hashlib.sha256).digest()
    return base64.b64encode(dig).decode()


def init_session_state():
    """Initialize session state variables."""
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
    # Cognito authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "id_token" not in st.session_state:
        st.session_state.id_token = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = None
    # Challenge state
    if "challenge_name" not in st.session_state:
        st.session_state.challenge_name = None
    if "challenge_session" not in st.session_state:
        st.session_state.challenge_session = None
    if "temp_username" not in st.session_state:
        st.session_state.temp_username = None


def authenticate_user(username: str, password: str, cognito_client, client_id: str, user_pool_id: str, client_secret: str = None, use_admin_auth: bool = False):
    """
    Authenticate user with Cognito.
    
    Args:
        username: User's username
        password: User's password
        cognito_client: Boto3 Cognito IDP client
        client_id: Cognito app client ID
        user_pool_id: Cognito user pool ID
        client_secret: Cognito app client secret (optional)
        use_admin_auth: Use ADMIN_NO_SRP_AUTH flow instead of USER_PASSWORD_AUTH
        
    Returns:
        Authentication result or None if failed
    """
    try:
        if use_admin_auth:
            # Use admin authentication (requires admin permissions)
            auth_params = {
                'USERNAME': username,
                'PASSWORD': password
            }
            
            if client_secret:
                auth_params['SECRET_HASH'] = get_secret_hash(username, client_id, client_secret)
            
            response = cognito_client.admin_initiate_auth(
                UserPoolId=user_pool_id,
                ClientId=client_id,
                AuthFlow='ADMIN_NO_SRP_AUTH',
                AuthParameters=auth_params
            )
        else:
            # Use standard user authentication
            auth_params = {
                'USERNAME': username,
                'PASSWORD': password
            }
            
            if client_secret:
                auth_params['SECRET_HASH'] = get_secret_hash(username, client_id, client_secret)
            
            response = cognito_client.initiate_auth(
                ClientId=client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters=auth_params
            )
        
        return response
    except Exception as e:
        return None


def respond_to_auth_challenge(cognito_client, client_id: str, challenge_name: str, session: str, 
                              username: str, new_password: str, client_secret: str = None):
    """
    Respond to authentication challenge.
    
    Args:
        cognito_client: Boto3 Cognito IDP client
        client_id: Cognito app client ID
        challenge_name: Name of the challenge
        session: Challenge session token
        username: User's username
        new_password: New password for NEW_PASSWORD_REQUIRED challenge
        client_secret: Cognito app client secret (optional)
        
    Returns:
        Challenge response or None if failed
    """
    try:
        challenge_responses = {
            'USERNAME': username,
            'NEW_PASSWORD': new_password
        }
        
        if client_secret:
            challenge_responses['SECRET_HASH'] = get_secret_hash(username, client_id, client_secret)
        
        response = cognito_client.respond_to_auth_challenge(
            ClientId=client_id,
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


def refresh_tokens(cognito_client, client_id: str, refresh_token: str, client_secret: str = None):
    """
    Refresh authentication tokens.
    
    Args:
        cognito_client: Boto3 Cognito IDP client
        client_id: Cognito app client ID
        refresh_token: Refresh token
        client_secret: Cognito app client secret (optional)
        
    Returns:
        New tokens or None if failed
    """
    try:
        auth_params = {
            'REFRESH_TOKEN': refresh_token
        }
        
        if client_secret:
            # For refresh, we need to use a dummy username for secret hash
            auth_params['SECRET_HASH'] = get_secret_hash(st.session_state.username, client_id, client_secret)
        
        response = cognito_client.initiate_auth(
            ClientId=client_id,
            AuthFlow='REFRESH_TOKEN_AUTH',
            AuthParameters=auth_params
        )
        
        return response
    except Exception as e:
        st.error(f"Token refresh failed: {str(e)}")
        return None


def logout():
    """Clear authentication state and logout user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.id_token = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


def display_message(role: str, content: str):
    """Display a chat message."""
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)


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
        id_token: Cognito ID token for Bearer authentication (optional)
        
    Yields:
        Response chunks from the agent
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
        for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
            if line:
                # SSE format: "data: <content>"
                if line.startswith("data: "):
                    data = line[6:].strip()
                    # Remove surrounding quotes and unescape JSON string
                    if data.startswith('"') and data.endswith('"'):
                        # Use json.loads to properly unescape the JSON string
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            # If it fails, just remove quotes manually
                            data = data[1:-1]
                    if data:
                        yield data
                    
    except requests.exceptions.RequestException as e:
        import traceback
        error_details = traceback.format_exc()
        raise Exception(f"Error invoking AgentCore agent: {str(e)}\n{error_details}")


def show_login_page(cognito_client, client_id: str, user_pool_id: str, client_secret: str = None, use_admin_auth: bool = False):
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
                        result = respond_to_auth_challenge(
                            cognito_client, 
                            client_id, 
                            st.session_state.challenge_name,
                            st.session_state.challenge_session,
                            st.session_state.temp_username,
                            new_password,
                            client_secret
                        )
                        
                        if result and 'AuthenticationResult' in result:
                            st.session_state.authenticated = True
                            st.session_state.username = st.session_state.temp_username
                            st.session_state.id_token = result['AuthenticationResult']['IdToken']
                            st.session_state.access_token = result['AuthenticationResult']['AccessToken']
                            st.session_state.refresh_token = result['AuthenticationResult'].get('RefreshToken')
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
                    result = authenticate_user(username, password, cognito_client, client_id, user_pool_id, client_secret, use_admin_auth)
                    
                    if result:
                        if 'AuthenticationResult' in result:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.id_token = result['AuthenticationResult']['IdToken']
                            st.session_state.access_token = result['AuthenticationResult']['AccessToken']
                            st.session_state.refresh_token = result['AuthenticationResult'].get('RefreshToken')
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


def main():
    """Main Streamlit application."""

    load_dotenv()  # Load environment variables

    st.set_page_config(
        page_title="E-Commerce Assistant (Cognito)",
        page_icon="🛒",
        layout="wide"
    )
    
    # Initialize session state
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
    
    # Initialize Cognito client
    cognito_client = boto3.client('cognito-idp', region_name=aws_region)
    
    # Show login page if not authenticated
    if not st.session_state.authenticated:
        show_login_page(cognito_client, client_id, user_pool_id, client_secret, use_admin_auth)
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
            logout()
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
            "List my orders salong with the products in Feb 2025",
            "Show the distribution by product categories of my orders this year",
            "List the top 5 user by sales in 2025 and show their product review summary for each of them",
            "Which customers have spent more than $1000 in the 2nd Quarter of 2025?",
            "What are the specifications of the Winter Jacket and also show the review sentiments",
            "What colors are available for the Cotton T-Shirt and also how many people bought it in 2025",
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
                        st.write("Processing your request via AWS AgentCore...")
                        
                        # Stream response from AgentCore
                        for chunk in invoke_agentcore_agent(
                            last_message["content"],
                            agent_arn,
                            st.session_state.session_id,
                            aws_region,
                            st.session_state.access_token
                        ):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        
                        message_placeholder.markdown(full_response, unsafe_allow_html=True)
                    
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
    if prompt := st.chat_input("Ask me anything about products, orders and reviews ..."):
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
                    st.write("Processing your request via AWS AgentCore...")
                    
                    # Stream response from AgentCore
                    for chunk in invoke_agentcore_agent(
                        prompt,
                        agent_arn,
                        st.session_state.session_id,
                        aws_region,
                        st.session_state.access_token
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                
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
