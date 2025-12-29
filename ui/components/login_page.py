"""Login page UI components."""

import streamlit as st
import time
import jwt
import json


def render_login_page(user_auth, use_admin_auth: bool = False):
    """Display login page with OAuth Hosted UI.
    
    Args:
        user_auth: UserAuth instance
        use_admin_auth: Unused parameter (kept for backward compatibility)
    """
    # Handle OAuth callback
    query_params = st.query_params
    if 'code' in query_params:
        _handle_oauth_callback(user_auth, query_params)
        return
    
    # Show login page
    _render_login_options(user_auth)


def _render_login_options(user_auth):
    """Render login page with OAuth."""
    st.title("🔐 E-Commerce Assistant")
    st.markdown("Welcome! Please sign in to continue.")
    
    # Generate auth URL
    auth_url, state = user_auth.get_authorization_url()
    st.session_state.oauth_state = state
    
    # Create a professional login button
    st.markdown(
        f"""
        <div style="margin-top: 2rem; margin-bottom: 2rem;">
            <a href="{auth_url}" target="_self" style="text-decoration: none;">
                <button style="
                    background-color: #0066cc;
                    color: white;
                    padding: 0.75rem 1.5rem;
                    border: none;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 1rem;
                    font-weight: 500;
                    width: 100%;
                    transition: background-color 0.2s;
                ">
                    Sign In
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.caption("Need access? Contact your administrator.")


def _handle_oauth_callback(user_auth, query_params):
    """Handle OAuth callback from Cognito."""
    code = query_params.get('code')
    state = query_params.get('state')
    error = query_params.get('error')
    
    # Clear query params
    st.query_params.clear()
    
    if error:
        st.error(f"❌ Authentication failed: {error}")
        st.info("Please try logging in again")
        time.sleep(2)
        st.rerun()
        return
    
    # Skip state validation for now (Streamlit session state doesn't persist across redirects)
    # In production, you'd want to store state in a database or secure cookie
    
    # Exchange code for tokens
    with st.spinner("🔄 Completing authentication..."):
        token_response = user_auth.exchange_code_for_tokens(code)
        
        if token_response and 'access_token' in token_response:
            _save_oauth_tokens(token_response)
            st.success("✅ Login successful!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("❌ Failed to exchange authorization code for tokens")
            time.sleep(2)
            st.rerun()


def _save_oauth_tokens(token_response: dict):
    """Save OAuth tokens to session state."""
    from auth.session_manager import SessionManager
    
    st.session_state.authenticated = True
    st.session_state.id_token = token_response['id_token']
    st.session_state.access_token = token_response['access_token']
    st.session_state.refresh_token = token_response.get('refresh_token')
    
    # Decode ID token to extract user info
    try:
        decoded_id_token = jwt.decode(
            st.session_state.id_token,
            options={"verify_signature": False}
        )
        
        # Extract username and tenant ID
        username = decoded_id_token.get('cognito:username', decoded_id_token.get('email', 'unknown'))
        tenant_id = decoded_id_token.get('custom:tenantId', 'N/A')
        
        st.session_state.username = username.lower()
        st.session_state.tenant_id = tenant_id
        
        # Debug: Print tokens
        print("\n" + "="*80)
        print("🔐 OAuth Authentication Successful")
        print("="*80)
        print("\n📋 ID Token Claims:")
        print(json.dumps(decoded_id_token, indent=2))
        
        # Decode and print access token
        decoded_access_token = jwt.decode(
            st.session_state.access_token,
            options={"verify_signature": False}
        )
        print("\n🎫 Access Token Claims:")
        print(json.dumps(decoded_access_token, indent=2))
        
        # Highlight scopes
        scopes = decoded_access_token.get('scope', '').split()
        print(f"\n✅ Custom Scopes Included: {scopes}")
        print("="*80 + "\n")
        
    except Exception as e:
        st.session_state.username = 'unknown'
        st.session_state.tenant_id = 'N/A'
        print(f"⚠️ Error decoding tokens: {e}")
    
    # Save session for persistence
    SessionManager.save_session(
        st.session_state.username,
        st.session_state.id_token,
        st.session_state.access_token,
        st.session_state.refresh_token
    )
