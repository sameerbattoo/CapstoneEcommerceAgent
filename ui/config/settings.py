"""Centralized configuration management."""

from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass
class CognitoConfig:
    """Cognito authentication configuration."""
    user_pool_id: str
    client_id: str
    client_secret: str | None = None
    use_admin_auth: bool = False
    domain: str = ''
    redirect_uri: str = ''
    scopes: str = 'openid email profile'


@dataclass
class AWSConfig:
    """AWS services configuration."""
    region: str
    agentcore_arn: str
    memory_id: str
    transcribe_bucket: str
    bedrock_model_arn: str
    kb_id: str


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        load_dotenv()
        
        # Cognito configuration
        self.cognito = CognitoConfig(
            user_pool_id=os.getenv('COGNITO_USER_POOL_ID', ''),
            client_id=os.getenv('COGNITO_CLIENT_ID', ''),
            client_secret=os.getenv('COGNITO_CLIENT_SECRET'),
            use_admin_auth=os.getenv('COGNITO_USE_ADMIN_AUTH', 'false').lower() == 'true',
            domain=os.getenv('COGNITO_DOMAIN', ''),
            redirect_uri=os.getenv('COGNITO_REDIRECT_URI', 'http://localhost:8501'),
            scopes=os.getenv('COGNITO_SCOPES', 'openid email profile')
        )
        
        # AWS configuration
        self.aws = AWSConfig(
            region=os.getenv('AWS_REGION', 'us-west-2'),
            agentcore_arn=os.getenv('AGENTCORE_ARN_COGNITO', ''),
            memory_id=os.getenv('AGENTCORE_MEMORY_ID', ''),
            transcribe_bucket=os.getenv('TRANSCRIBE_S3_BUCKET', 'capstone-voice-recordings'),
            bedrock_model_arn=os.getenv('BEDROCK_MODEL_ARN', ''),
            kb_id=os.getenv('KB_ID', '')
        )
    
    @classmethod
    def load(cls) -> 'Settings':
        """Load settings from environment."""
        return cls()
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate required configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.cognito.user_pool_id:
            errors.append("COGNITO_USER_POOL_ID is required")
        if not self.cognito.client_id:
            errors.append("COGNITO_CLIENT_ID is required")
        if not self.aws.agentcore_arn:
            errors.append("AGENTCORE_ARN_COGNITO is required")
        
        return len(errors) == 0, errors
