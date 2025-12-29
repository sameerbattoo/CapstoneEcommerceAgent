"""AWS Cognito authentication handler."""

import hmac
import hashlib
import base64
import secrets
import urllib.parse


class UserAuth:
    """Handles all Cognito authentication operations."""
    
    def __init__(self, cognito_client, client_id: str, user_pool_id: str, client_secret: str | None = None,
                 domain: str = '', redirect_uri: str = '', scopes: str = 'openid email profile', region: str = 'us-west-2'):
        """Initialize UserAuth with Cognito configuration.
        
        Args:
            cognito_client: Boto3 Cognito IDP client
            client_id: Cognito app client ID
            user_pool_id: Cognito user pool ID
            client_secret: Cognito app client secret (optional)
            domain: Cognito domain for hosted UI
            redirect_uri: OAuth redirect URI
            scopes: Space-separated OAuth scopes
            region: AWS region
        """
        self.cognito_client = cognito_client
        self.client_id = client_id
        self.user_pool_id = user_pool_id
        self.client_secret = client_secret
        self.domain = domain
        self.redirect_uri = redirect_uri
        self.scopes = scopes
        self.region = region
    
    def get_authorization_url(self, state: str = None) -> tuple[str, str]:
        """Generate Cognito Hosted UI authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'scope': self.scopes,
            'redirect_uri': self.redirect_uri,
            'state': state
        }
        
        auth_url = f"https://{self.domain}.auth.{self.region}.amazoncognito.com/oauth2/authorize"
        full_url = f"{auth_url}?{urllib.parse.urlencode(params)}"
        
        return full_url, state
    
    def exchange_code_for_tokens(self, code: str) -> dict | None:
        """Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from Cognito callback
            
        Returns:
            Token response or None if failed
        """
        import requests
        
        token_url = f"https://{self.domain}.auth.{self.region}.amazoncognito.com/oauth2/token"
        
        data = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'code': code,
            'redirect_uri': self.redirect_uri
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        try:
            response = requests.post(token_url, data=data, headers=headers)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_logout_url(self) -> str:
        """Generate Cognito logout URL.
        
        Returns:
            Logout URL
        """
        params = {
            'client_id': self.client_id,
            'logout_uri': self.redirect_uri
        }
        
        logout_url = f"https://{self.domain}.auth.{self.region}.amazoncognito.com/logout"
        return f"{logout_url}?{urllib.parse.urlencode(params)}"
    
    @staticmethod
    def get_secret_hash(username: str, client_id: str, client_secret: str) -> str:
        """Generate secret hash for Cognito authentication."""
        message = bytes(username + client_id, 'utf-8')
        secret = bytes(client_secret, 'utf-8')
        dig = hmac.new(secret, msg=message, digestmod=hashlib.sha256).digest()
        return base64.b64encode(dig).decode()
    
    def authenticate(self, username: str, password: str, use_admin_auth: bool = False):
        """Authenticate user with Cognito (legacy direct auth).
        
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
        except Exception:
            return None
    
    def respond_to_challenge(self, challenge_name: str, session: str, username: str, new_password: str):
        """Respond to authentication challenge.
        
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
        except Exception:
            return None
    
    def refresh_tokens(self, refresh_token: str, username: str):
        """Refresh authentication tokens.
        
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
        except Exception:
            return None


class AuthenticationError(Exception):
    """Exception for authentication errors."""
    pass
