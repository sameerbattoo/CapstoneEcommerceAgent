# ECommerce Agent - Authentication & Authorization Specification

## Overview

The solution implements a comprehensive OAuth2-based authentication and authorization system using AWS Cognito for user authentication and JWT tokens for secure communication between components.

## Authentication Architecture

```
┌──────────────┐
│   End User   │
└──────┬───────┘
       │ 1. Login
       ▼
┌──────────────────────┐
│  Streamlit UI        │
│  (Login Page)        │
└──────┬───────────────┘
       │ 2. Authenticate
       ▼
┌──────────────────────┐
│  AWS Cognito         │
│  User Pool           │
└──────┬───────────────┘
       │ 3. JWT Token (with tenantId)
       ▼
┌──────────────────────┐
│  Pre-Token Lambda    │
│  (Inject Claims)     │
└──────┬───────────────┘
       │ 4. Enhanced JWT
       ▼
┌──────────────────────┐
│  Streamlit UI        │
│  (Session Manager)   │
└──────┬───────────────┘
       │ 5. API Call with Bearer Token
       ▼
┌──────────────────────┐
│  AgentCore Runtime   │
│  (JWT Validator)     │
└──────┬───────────────┘
       │ 6. Extract Claims
       ▼
┌──────────────────────┐
│  Orchestrator Agent  │
│  (Process Request)   │
└──────┬───────────────┘
       │ 7. Propagate Token
       ▼
┌──────────────────────┐
│  AgentCore Gateway   │
│  (Fine-Grained Auth) │
└──────┬───────────────┘
       │ 8. Custom Headers
       ▼
┌──────────────────────┐
│  Lambda Function     │
│  (Tenant Isolation)  │
└──────────────────────┘
```

---

## 1. Inbound Authentication (User → AgentCore)

### AWS Cognito User Pool Configuration

#### User Pool Settings
```json
{
  "UserPoolId": "us-west-2_5cqnrBvAg",
  "UserPoolName": "CapstoneECommerceUserPool",
  "Policies": {
    "PasswordPolicy": {
      "MinimumLength": 8,
      "RequireUppercase": true,
      "RequireLowercase": true,
      "RequireNumbers": true,
      "RequireSymbols": true
    }
  },
  "MfaConfiguration": "OPTIONAL",
  "AccountRecoverySetting": {
    "RecoveryMechanisms": [
      {"Name": "verified_email", "Priority": 1}
    ]
  }
}
```

#### App Client Configuration
```json
{
  "ClientId": "2nc8c09npb6vru63mgfl7kiqb1",
  "ClientName": "CapstoneECommerceWebApp",
  "ExplicitAuthFlows": [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH"
  ],
  "TokenValidityUnits": {
    "AccessToken": "hours",
    "IdToken": "hours",
    "RefreshToken": "days"
  },
  "AccessTokenValidity": 1,
  "IdTokenValidity": 1,
  "RefreshTokenValidity": 30,
  "PreventUserExistenceErrors": "ENABLED"
}
```

### Pre-Token Generation Lambda Trigger

**Purpose**: Inject custom claims (tenantId) into JWT tokens

**File**: `lambda/CapstoneEcommerceCognitoPreTokenGeneration.py`

```python
import json

def lambda_handler(event, context):
    """
    Pre-token generation trigger to add custom claims to JWT.
    
    Triggered by Cognito before generating tokens.
    """
    
    # Extract user attributes
    user_attributes = event['request']['userAttributes']
    username = user_attributes.get('cognito:username')
    email = user_attributes.get('email')
    
    # Determine tenant ID based on user attributes
    # In production, this would query a user-tenant mapping table
    tenant_id = determine_tenant_id(username, email)
    
    # Add custom claims to access token
    event['response']['claimsOverrideDetails'] = {
        'claimsToAddOrOverride': {
            'tenantId': tenant_id,
            'actorId': username.replace('.', '_').replace('@', '-')
        }
    }
    
    return event

def determine_tenant_id(username: str, email: str) -> str:
    """
    Determine tenant ID for user.
    
    Logic:
    - Users with @tenanta.com → tenanta
    - Users with @tenantb.com → tenantb
    - Default users → tenanta (for demo)
    """
    if email and '@tenantb.com' in email:
        return 'tenantb'
    elif email and '@tenanta.com' in email:
        return 'tenanta'
    else:
        # Default tenant for demo users
        return 'tenanta'
```

**Trigger Configuration**:
```json
{
  "LambdaTriggers": {
    "PreTokenGeneration": "arn:aws:lambda:us-west-2:175918693907:function:CapstoneEcommerceCognitoPreTokenGeneration"
  }
}
```

### JWT Token Structure

#### Access Token Claims
```json
{
  "sub": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "cognito:groups": ["Users"],
  "iss": "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_5cqnrBvAg",
  "client_id": "2nc8c09npb6vru63mgfl7kiqb1",
  "origin_jti": "xyz123",
  "event_id": "abc456",
  "token_use": "access",
  "scope": "capstone_agent_server/read",
  "auth_time": 1735257600,
  "exp": 1735261200,
  "iat": 1735257600,
  "jti": "def789",
  "username": "john.doe",
  "tenantId": "tenanta",
  "actorId": "john_doe"
}
```

### Streamlit Authentication Flow

**File**: `ui/auth/cognito_auth.py`

```python
import boto3
from botocore.exceptions import ClientError

class UserAuth:
    def __init__(self, user_pool_id: str, client_id: str, region: str):
        self.cognito_client = boto3.client('cognito-idp', region_name=region)
        self.user_pool_id = user_pool_id
        self.client_id = client_id
    
    def authenticate(self, username: str, password: str) -> dict:
        """
        Authenticate user with Cognito.
        
        Returns:
            dict with access_token, id_token, refresh_token
        """
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            
            return {
                'success': True,
                'access_token': response['AuthenticationResult']['AccessToken'],
                'id_token': response['AuthenticationResult']['IdToken'],
                'refresh_token': response['AuthenticationResult']['RefreshToken'],
                'expires_in': response['AuthenticationResult']['ExpiresIn']
            }
        
        except ClientError as e:
            return {
                'success': False,
                'error': e.response['Error']['Message']
            }
    
    def refresh_token(self, refresh_token: str) -> dict:
        """Refresh access token using refresh token."""
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='REFRESH_TOKEN_AUTH',
                AuthParameters={
                    'REFRESH_TOKEN': refresh_token
                }
            )
            
            return {
                'success': True,
                'access_token': response['AuthenticationResult']['AccessToken'],
                'id_token': response['AuthenticationResult']['IdToken']
            }
        
        except ClientError as e:
            return {
                'success': False,
                'error': e.response['Error']['Message']
            }
```

**Session Management**: `ui/auth/session_manager.py`

```python
import pickle
from pathlib import Path
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self, session_file: str = ".streamlit_session"):
        self.session_file = Path.home() / session_file
    
    def save_session(self, session_data: dict):
        """Persist session data to disk."""
        with open(self.session_file, 'wb') as f:
            pickle.dump(session_data, f)
    
    def load_session(self) -> dict:
        """Load session data from disk."""
        if not self.session_file.exists():
            return None
        
        with open(self.session_file, 'rb') as f:
            session_data = pickle.load(f)
        
        # Check if token expired
        if self._is_expired(session_data):
            return None
        
        return session_data
    
    def clear_session(self):
        """Delete session file."""
        if self.session_file.exists():
            self.session_file.unlink()
    
    def _is_expired(self, session_data: dict) -> bool:
        """Check if access token expired."""
        login_time = session_data.get('login_time')
        expires_in = session_data.get('expires_in', 3600)
        
        if not login_time:
            return True
        
        expiry_time = datetime.fromisoformat(login_time) + timedelta(seconds=expires_in)
        return datetime.now() > expiry_time
```

---

## 2. AgentCore JWT Validation

### Custom JWT Authorizer Configuration

**File**: `.bedrock_agentcore.yaml`

```yaml
authorizer_configuration:
  customJWTAuthorizer:
    discoveryUrl: https://cognito-idp.us-west-2.amazonaws.com/us-west-2_5cqnrBvAg/.well-known/openid-configuration
    allowedClients:
      - 2nc8c09npb6vru63mgfl7kiqb1
    allowedScopes:
      - capstone_agent_server/read
```

### Request Header Configuration

**Allowed Headers**:
```yaml
request_header_configuration:
  requestHeaderAllowlist:
    - Authorization
    - X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId
    - X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId
```

### JWT Validation Flow

1. **Client sends request** with `Authorization: Bearer <token>`
2. **AgentCore Runtime** validates JWT:
   - Fetches public keys from Cognito JWKS endpoint
   - Verifies signature using RS256 algorithm
   - Checks token expiration (`exp` claim)
   - Validates issuer (`iss` claim)
   - Validates audience (`client_id` claim)
3. **If valid**, extracts claims and passes to agent
4. **If invalid**, returns 401 Unauthorized

---

## 3. Claim Extraction in Agent

**File**: `main.py`

```python
import jwt

def _extract_user_context(self, headers: Optional[Dict]) -> Tuple[str, str, str]:
    """
    Extract user email, actor_id, and tenant_id from JWT token.
    
    Args:
        headers: Request headers containing Authorization token
        
    Returns:
        Tuple of (user_email, actor_id, tenant_id)
    """
    user_email = None
    actor_id = 'user'
    tenant_id = 'N/A'
    
    if not headers:
        return user_email, actor_id, tenant_id
    
    # Check for custom headers (passed explicitly)
    if 'x-amzn-bedrock-agentcore-runtime-custom-actorid' in headers:
        actor_id = headers.get('x-amzn-bedrock-agentcore-runtime-custom-actorid')
    
    if 'x-amzn-bedrock-agentcore-runtime-custom-tenantid' in headers:
        tenant_id = headers.get('x-amzn-bedrock-agentcore-runtime-custom-tenantid')
    
    # Extract from JWT token
    auth_header = headers.get("Authorization")
    if auth_header:
        try:
            token = auth_header.replace("Bearer ", "")
            
            # Decode without verification (already validated by AgentCore)
            claims = jwt.decode(
                token, 
                options={"verify_signature": False, "verify_aud": False}
            )
            
            # Extract username and construct email
            username = claims.get("username")
            if username:
                user_email = f"{username}@email.com"
                
                # Infer actor_id if not provided
                if actor_id == 'user':
                    actor_id = username.replace(".", "_").replace("@", "-")
            
            # Extract tenant_id from custom claim
            if tenant_id == 'N/A' and claims.get("tenantId"):
                tenant_id = claims.get("tenantId")
            
            self.logger.info(
                f"Extracted user context: email={user_email}, "
                f"actor_id={actor_id}, tenant_id={tenant_id}"
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to parse JWT token: {str(e)}")
    
    return user_email, actor_id, tenant_id
```

---

## 4. Outbound Authentication (AgentCore → Lambda)

### AgentCore Gateway Fine-Grained Access Control

**Purpose**: Propagate tenant context to Lambda functions

**Configuration**:
```yaml
# In AgentCore Gateway configuration
fine_grained_access_control:
  enabled: true
  custom_headers:
    - X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId
    - X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId
```

### Lambda Function URL with AWS_IAM Auth

**Lambda Configuration**:
```json
{
  "FunctionName": "ProductReviewsAPI",
  "FunctionUrlConfig": {
    "AuthType": "AWS_IAM",
    "Cors": {
      "AllowOrigins": ["*"],
      "AllowMethods": ["GET", "POST"],
      "AllowHeaders": ["*"]
    }
  }
}
```

### Header Propagation

**AgentCore Runtime** → **AgentCore Gateway** → **Lambda**

Headers propagated:
- `Authorization: Bearer <token>`
- `X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId: tenanta`
- `X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId: john_doe`

### Lambda Context Extraction

**File**: `lambda/product_reviews_api.py`

```python
def lambda_handler(event, context):
    """
    Extract tenant_id from AgentCore Gateway custom headers.
    """
    tenant_id = None
    
    # Get the AgentCore Gateway custom fields
    client_ctx = getattr(context, "client_context", None)
    if client_ctx and getattr(client_ctx, "custom", None):
        propagated = client_ctx.custom.get("bedrockAgentCorePropagatedHeaders", {})
        
        # Access the custom tenant ID header (case-insensitive)
        tenant_id = (
            propagated.get("x-amzn-bedrock-agentcore-runtime-custom-tenantid")
            or propagated.get("X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId")
        )
    
    # Validate tenant_id is present
    if not tenant_id:
        return {
            'statusCode': 403,
            'body': json.dumps({'error': 'Tenant ID required but not provided'})
        }
    
    # Query DynamoDB with tenant filter
    response = table.query(
        IndexName='tenant_id-product_id-index',
        KeyConditionExpression=Key('tenant_id').eq(tenant_id)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(response['Items'])
    }
```

---

## 5. IAM Permissions

### AgentCore Execution Role

**Role Name**: `CapstoneECommerce-agentcore-execution-role`

**Trust Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock-agentcore.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Permissions Policy**: `prereqs/agent_role.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockModelAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-*",
        "arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-*"
      ]
    },
    {
      "Sid": "KnowledgeBaseAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate"
      ],
      "Resource": "arn:aws:bedrock:us-west-2:*:knowledge-base/*"
    },
    {
      "Sid": "DynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:Query",
        "dynamodb:GetItem",
        "dynamodb:Scan"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-west-2:*:table/ProductReviews",
        "arn:aws:dynamodb:us-west-2:*:table/ProductReviews/index/*"
      ]
    },
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::capstone-ecommerce-charts/*",
        "arn:aws:s3:::capstone-ecommerce-kb/*"
      ]
    },
    {
      "Sid": "LambdaInvoke",
      "Effect": "Allow",
      "Action": [
        "lambda:InvokeFunctionUrl"
      ],
      "Resource": "arn:aws:lambda:us-west-2:*:function:ProductReviewsAPI"
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-west-2:*:secret:capstone-ecommerce-agent-config-*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-west-2:*:log-group:/aws/bedrock/agentcore/*"
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "CapstoneECommerceAgent/PerTenant"
        }
      }
    },
    {
      "Sid": "AgentCoreMemory",
      "Effect": "Allow",
      "Action": [
        "bedrock:GetMemory",
        "bedrock:PutMemory",
        "bedrock:DeleteMemory",
        "bedrock:ListMemories"
      ],
      "Resource": "arn:aws:bedrock:us-west-2:*:memory/*"
    },
    {
      "Sid": "VPCAccess",
      "Effect": "Allow",
      "Action": [
        "ec2:CreateNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterface"
      ],
      "Resource": "*"
    }
  ]
}
```

### Lambda Execution Role

**Role Name**: `ProductReviewsAPIExecutionRole`

**Permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:Query",
        "dynamodb:GetItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-west-2:*:table/ProductReviews",
        "arn:aws:dynamodb:us-west-2:*:table/ProductReviews/index/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-west-2:*:log-group:/aws/lambda/ProductReviewsAPI:*"
    }
  ]
}
```

---

## 6. Security Best Practices

### Token Security
1. **Short-lived tokens**: Access tokens expire in 1 hour
2. **Refresh tokens**: Valid for 30 days, stored securely
3. **HTTPS only**: All communication over TLS 1.2+
4. **Token rotation**: Automatic refresh before expiration

### Multi-Tenancy Isolation
1. **JWT claims**: Tenant ID embedded in token (tamper-proof)
2. **Schema isolation**: Separate PostgreSQL schemas per tenant
3. **Attribute filtering**: DynamoDB queries filtered by tenant_id
4. **Metadata filtering**: Knowledge Base retrieval filtered by tenant_access

### Secrets Management
1. **AWS Secrets Manager**: All credentials stored in Secrets Manager
2. **IAM-based access**: No hardcoded credentials
3. **Rotation**: Automatic secret rotation enabled
4. **Audit trail**: CloudTrail logs all secret access

### Network Security
1. **VPC isolation**: AgentCore Runtime in private subnets
2. **Security groups**: Restrictive inbound/outbound rules
3. **NAT Gateway**: Controlled outbound internet access
4. **Private endpoints**: VPC endpoints for AWS services

---

## 7. Testing Authentication

### Test User Creation
```bash
aws cognito-idp admin-create-user \
  --user-pool-id us-west-2_5cqnrBvAg \
  --username john.doe \
  --user-attributes Name=email,Value=john.doe@tenanta.com \
  --temporary-password TempPass123! \
  --message-action SUPPRESS
```

### Test Authentication
```python
import boto3

cognito = boto3.client('cognito-idp', region_name='us-west-2')

response = cognito.initiate_auth(
    ClientId='2nc8c09npb6vru63mgfl7kiqb1',
    AuthFlow='USER_PASSWORD_AUTH',
    AuthParameters={
        'USERNAME': 'john.doe',
        'PASSWORD': 'YourPassword123!'
    }
)

access_token = response['AuthenticationResult']['AccessToken']
print(f"Access Token: {access_token}")
```

### Test JWT Decoding
```python
import jwt

token = "eyJraWQiOiJ..."  # Your access token

# Decode without verification (for testing)
claims = jwt.decode(token, options={"verify_signature": False})

print(f"Username: {claims['username']}")
print(f"Tenant ID: {claims['tenantId']}")
print(f"Actor ID: {claims['actorId']}")
print(f"Expires: {claims['exp']}")
```

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
