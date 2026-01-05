"""AWS Bedrock AgentCore client for agent invocations."""

import json
import urllib.parse
import requests
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import re


class AgentCoreClient:
    """Client for interacting with AWS Bedrock AgentCore."""
    
    def __init__(self, region: str, agent_arn: str):
        """Initialize AgentCore client.
        
        Args:
            region: AWS region
            agent_arn: ARN of the deployed AgentCore agent
        """
        self.region = region
        self.agent_arn = agent_arn
        self.session = boto3.Session()
    
    def invoke_streaming(
        self,
        user_input: str,
        session_id: str,
        access_token: str | None = None,
        tenant_id: str | None = None,
        actor_id: str | None = None,
        timeout: int = 100
    ):
        """Invoke AgentCore agent and stream the response.
        
        Args:
            user_input: The user's question
            session_id: Session ID for conversation continuity
            access_token: Cognito access token for Bearer authentication (optional)
            tenant_id: Tenant ID of the logged in user (optional)
            actor_id: Actor ID of the logged in user (optional)
            timeout: Request timeout in seconds
            
        Yields:
            Structured events: {"type": "thinking"|"content"|"tool_use", "data": ...}
        """
        url = self._build_url()
        headers = self._build_headers(session_id, tenant_id, actor_id, access_token)
        payload = {"user_input": user_input}
        body = json.dumps(payload)
        
        try:
            response = self._make_request(url, headers, body, access_token, timeout)
            response.raise_for_status()
            
            # Stream and parse response
            yield from self._parse_event_stream(response)
            
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
    
    def _build_url(self) -> str:
        """Build the AgentCore invocation URL."""
        escaped_arn = urllib.parse.quote(self.agent_arn, safe="")
        return f"https://bedrock-agentcore.{self.region}.amazonaws.com/runtimes/{escaped_arn}/invocations"
    
    def _build_headers(
        self,
        session_id: str,
        tenant_id: str | None,
        actor_id: str | None,
        access_token: str | None
    ) -> dict:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }
        
        if tenant_id:
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId"] = tenant_id
        if actor_id:
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId"] = actor_id
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        
        return headers
    
    def _make_request(
        self,
        url: str,
        headers: dict,
        body: str,
        access_token: str | None,
        timeout: int
    ) -> requests.Response:
        """Make HTTP request with optional SigV4 signing."""
        if access_token:
            # Direct request with Bearer token
            return requests.post(
                url,
                params={"qualifier": "DEFAULT"},
                headers=headers,
                data=body,
                timeout=timeout,
                stream=True,
            )
        else:
            # Use SigV4 signing for IAM-based authentication
            credentials = self.session.get_credentials()
            request = AWSRequest(
                method='POST',
                url=url + '?qualifier=DEFAULT',
                data=body,
                headers=headers
            )
            SigV4Auth(credentials, 'bedrock-agentcore', self.region).add_auth(request)
            
            return requests.post(
                url,
                params={"qualifier": "DEFAULT"},
                headers=dict(request.headers),
                data=body,
                timeout=timeout,
                stream=True,
            )
    
    def _parse_event_stream(self, response: requests.Response):
        """Parse Server-Sent Events format from response.
        
        Yields:
            Structured events with type and data
        """
        current_event_type = None
        
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
                    
                    if data:
                        data_stripped = data.lstrip("\r\n")
                        if data_stripped.startswith("[TOOL USE]"):
                            yield {"type": "tool_use", "data": data_stripped.removeprefix("[TOOL USE]")}
                        elif data_stripped.startswith("[THINKING]"):
                            yield {"type": "thinking", "data": data_stripped.removeprefix("[THINKING]")}
                        elif data_stripped.startswith("[METRICS]"):
                            yield {"type": "metrics", "data": data_stripped.removeprefix("[METRICS]")}
                        else:
                            yield {"type": "content", "data": data}
    
    def _handle_request_error(self, error: requests.exceptions.RequestException):
        """Handle request errors with detailed information."""
        import traceback
        error_details = traceback.format_exc()
        
        # Check for 401 Unauthorized
        if hasattr(error, 'response') and error.response is not None:
            if error.response.status_code == 401:
                raise UnauthorizedError("Session expired or invalid credentials")
        
        raise AgentCoreError(f"Error invoking AgentCore agent: {str(error)}\n{error_details}")


class AgentCoreError(Exception):
    """Base exception for AgentCore errors."""
    pass


class UnauthorizedError(AgentCoreError):
    """Exception for authentication/authorization errors."""
    pass
