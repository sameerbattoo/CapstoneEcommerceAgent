# ECommerce Agent - API Reference

## Overview

This document provides comprehensive API reference for all components in the ECommerce Agent solution.

---

## 1. AgentCore Runtime API

### Base URL
```
https://runtime.bedrock-agentcore.us-west-2.amazonaws.com
```

### Authentication
All requests require JWT Bearer token from AWS Cognito.

```http
Authorization: Bearer <access_token>
```

### Invoke Agent (Streaming)

**Endpoint**: `POST /agents/{agentId}/invoke`

**Description**: Invoke the agent with streaming response

**Request Headers**:
```http
Authorization: Bearer eyJraWQiOiJ...
Content-Type: application/json
X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId: tenanta
X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId: john_doe
```

**Request Body**:
```json
{
  "inputText": "Show me my recent orders",
  "sessionId": "session-abc123",
  "enableTrace": false,
  "endSession": false
}
```

**Parameters**:
- `inputText` (string, required): User's query or request
- `sessionId` (string, required): Unique session identifier for conversation continuity
- `enableTrace` (boolean, optional): Enable detailed trace information (default: false)
- `endSession` (boolean, optional): End the session after this request (default: false)

**Response** (Server-Sent Events):
```
event: chunk
data: {"type": "content", "data": "Here are your recent orders:\n"}

event: chunk
data: {"type": "tool_use", "current_tool_use": {"name": "get_answers_for_structured_data", "input": {"query": "recent orders"}}}

event: chunk
data: {"type": "content", "data": "1. Order #1001 - Smartphone X - Delivered\n"}

event: chunk
data: {"type": "reasoning", "reasoningText": "I should check if there are more orders..."}

event: chunk
data: {"type": "metrics", "duration_seconds": 3.45, "input_tokens": 1250, "output_tokens": 450}

event: done
data: {"status": "complete"}
```

**Response Event Types**:
- `content`: Text response chunks
- `tool_use`: Tool invocation information
- `reasoning`: Model's thinking process (Claude extended thinking)
- `metrics`: Performance metrics
- `done`: End of stream

**Error Responses**:
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired JWT token"
  }
}
```

**Status Codes**:
- `200 OK`: Successful streaming response
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

## 2. Product Reviews API (Lambda)

### Base URL
```
https://abc123def456.lambda-url.us-west-2.on.aws/
```

### Authentication
AWS_IAM authentication via AgentCore Gateway (Bearer token propagated)

### Get Product Reviews

**Endpoint**: `GET /`

**Description**: Query product reviews with flexible filtering

**Query Parameters**:
- `product_id` (string, optional): Filter by product ID(s), comma-separated
  - Example: `product_id=1` or `product_id=1,2,3`
- `product_name` (string, optional): Filter by product name (partial match), comma-separated
  - Example: `product_name=Smartphone`
- `customer_id` (string, optional): Filter by customer ID(s), comma-separated
  - Example: `customer_id=1,2`
- `customer_email` (string, optional): Filter by customer email (partial match), comma-separated
  - Example: `customer_email=john.doe@email.com`
- `rating` (string, optional): Filter by rating value(s), comma-separated
  - Example: `rating=4,5` (4 or 5 stars)
- `review_date_from` (string, optional): Start date for review date range (YYYY-MM-DD)
  - Example: `review_date_from=2024-01-01`
- `review_date_to` (string, optional): End date for review date range (YYYY-MM-DD)
  - Example: `review_date_to=2024-12-31`
- `top_rows` (integer, optional): Limit number of rows returned (default: 20, max: 20)
  - Example: `top_rows=10`

**Request Example**:
```http
GET /?product_id=1&rating=4,5&top_rows=10
Authorization: Bearer eyJraWQiOiJ...
```

**Response**:
```json
{
  "statusCode": 200,
  "body": [
    {
      "product_id": 1,
      "review_id": "rev_20241201_001",
      "tenant_id": "tenanta",
      "customer_id": 1,
      "customer_email": "john.doe@email.com",
      "product_name": "Smartphone X",
      "rating": 5,
      "title": "Excellent phone!",
      "comment": "Best smartphone I've ever owned. Great camera and battery life.",
      "review_date": "2024-12-01T15:30:00Z",
      "helpful_votes": 12,
      "verified_purchase": true,
      "created_at": "2024-12-01T15:30:00Z",
      "updated_at": "2024-12-01T15:30:00Z"
    },
    {
      "product_id": 1,
      "review_id": "rev_20241215_002",
      "tenant_id": "tenanta",
      "customer_id": 3,
      "customer_email": "alice.johnson@email.com",
      "product_name": "Smartphone X",
      "rating": 4,
      "title": "Good phone, minor issues",
      "comment": "Overall great phone but battery drains quickly with heavy use.",
      "review_date": "2024-12-15T10:20:00Z",
      "helpful_votes": 5,
      "verified_purchase": true,
      "created_at": "2024-12-15T10:20:00Z",
      "updated_at": "2024-12-15T10:20:00Z"
    }
  ]
}
```

**Error Responses**:
```json
{
  "statusCode": 403,
  "body": {
    "error": "Tenant ID required but not provided"
  }
}
```

```json
{
  "statusCode": 500,
  "body": {
    "error": "Internal server error",
    "message": "DynamoDB query failed"
  }
}
```

**Status Codes**:
- `200 OK`: Successful query
- `400 Bad Request`: Invalid query parameters
- `403 Forbidden`: Missing tenant_id in headers
- `500 Internal Server Error`: Database error

---

## 3. Orchestrator Agent Tools

### Tool: get_answers_for_structured_data

**Description**: Query structured data from RDS PostgreSQL using natural language

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language query about products, orders, customers, or shipments"
    }
  },
  "required": ["query"]
}
```

**Example Input**:
```json
{
  "query": "Show me total sales by category for the last 30 days"
}
```

**Output**:
```json
{
  "success": true,
  "data": [
    {"category_name": "Electronics", "total_sales": 15000.00},
    {"category_name": "Clothing", "total_sales": 8500.00},
    {"category_name": "Sports", "total_sales": 5200.00}
  ],
  "sql": "SELECT c.category_name, SUM(o.total_amount) as total_sales FROM tenanta.orders o JOIN tenanta.order_items oi ON o.order_id = oi.order_id JOIN tenanta.products p ON oi.product_id = p.product_id JOIN tenanta.categories c ON p.category_id = c.category_id WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days' GROUP BY c.category_name ORDER BY total_sales DESC",
  "chart_url": "https://d1234567890abc.cloudfront.net/charts/tenanta/20241226_123456.png",
  "row_count": 3,
  "cache_hit": false,
  "execution_time_ms": 245
}
```

### Tool: get_answers_for_unstructured_data

**Description**: Search product specifications from Bedrock Knowledge Base

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Question about product specifications, features, or details"
    }
  },
  "required": ["query"]
}
```

**Example Input**:
```json
{
  "query": "What are the camera specifications for Smartphone X?"
}
```

**Output**:
```json
{
  "success": true,
  "response": "The Smartphone X features a triple camera system:\n- Main camera: 50MP wide-angle lens\n- Ultra-wide: 12MP with 120° field of view\n- Telephoto: 10MP with 3x optical zoom\n\nAdditional camera features include night mode, 8K video recording, and AI-powered scene detection.",
  "citations": [
    {
      "source": "s3://capstone-ecommerce-kb/Smartphone X - Product Spec.pdf",
      "score": 0.95,
      "excerpt": "Camera System: The Smartphone X is equipped with a professional-grade triple camera system..."
    }
  ],
  "citation_count": 1,
  "input_tokens": 450,
  "output_tokens": 120
}
```

### Tool: get_product_reviews (MCP)

**Description**: Retrieve customer reviews from DynamoDB

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "product_id": {
      "type": "string",
      "description": "Product ID(s) to filter by, comma-separated"
    },
    "customer_email": {
      "type": "string",
      "description": "Customer email to filter by"
    },
    "rating": {
      "type": "string",
      "description": "Rating value(s) to filter by, comma-separated (e.g., '4,5')"
    },
    "top_rows": {
      "type": "integer",
      "description": "Maximum number of reviews to return (default: 20)"
    }
  }
}
```

**Example Input**:
```json
{
  "product_id": "1",
  "rating": "4,5",
  "top_rows": 10
}
```

**Output**: Same as Product Reviews API response

### Tool: current_time

**Description**: Get current date and time

**Input Schema**:
```json
{
  "type": "object",
  "properties": {}
}
```

**Output**:
```json
{
  "current_time": "2024-12-26T15:30:45Z",
  "timezone": "UTC"
}
```

---

## 4. AgentCore Memory API

### Base URL
```
https://bedrock-agentcore.us-west-2.amazonaws.com/memory
```

### Get Last K Turns

**Endpoint**: `GET /memories/{memoryId}/turns`

**Description**: Retrieve last K conversation turns

**Parameters**:
- `memoryId` (path, required): Memory ID
- `actorId` (query, required): Actor/user ID
- `sessionId` (query, required): Session ID
- `k` (query, optional): Number of turns to retrieve (default: 3)

**Request**:
```http
GET /memories/memory-XPTahoCqjM/turns?actorId=john_doe&sessionId=session-123&k=3
Authorization: Bearer <aws_credentials>
```

**Response**:
```json
{
  "turns": [
    {
      "turnId": "turn-1",
      "messages": [
        {
          "role": "USER",
          "content": "Show me my recent orders"
        },
        {
          "role": "ASSISTANT",
          "content": "You have 2 recent orders: Order #1001 (Smartphone X) delivered on Dec 5, and Order #1002 (Laptop Pro 15) shipped on Dec 16."
        }
      ],
      "timestamp": "2024-12-20T10:30:00Z"
    },
    {
      "turnId": "turn-2",
      "messages": [
        {
          "role": "USER",
          "content": "What's the status of order #1002?"
        },
        {
          "role": "ASSISTANT",
          "content": "Order #1002 is currently in transit. Tracking number: TRK987654321. Estimated delivery: Dec 20."
        }
      ],
      "timestamp": "2024-12-20T10:32:00Z"
    }
  ]
}
```

### Search Memory

**Endpoint**: `POST /memories/{memoryId}/search`

**Description**: Semantic search across memory namespaces

**Request Body**:
```json
{
  "namespace": "/users/john_doe/facts",
  "query": "order information",
  "maxResults": 5
}
```

**Response**:
```json
{
  "results": [
    {
      "memoryId": "mem-abc123",
      "content": "User ordered Smartphone X on December 1, 2024. Order ID: 1001",
      "score": 0.92,
      "metadata": {
        "order_id": "1001",
        "product_name": "Smartphone X",
        "order_date": "2024-12-01"
      }
    },
    {
      "memoryId": "mem-def456",
      "content": "Order #1001 was delivered on December 5, 2024",
      "score": 0.88,
      "metadata": {
        "order_id": "1001",
        "delivery_date": "2024-12-05"
      }
    }
  ]
}
```

### Add Turn

**Endpoint**: `POST /memories/{memoryId}/turns`

**Description**: Add a conversation turn to memory

**Request Body**:
```json
{
  "actorId": "john_doe",
  "sessionId": "session-123",
  "messages": [
    {
      "role": "USER",
      "content": "Show me products in Electronics category"
    },
    {
      "role": "ASSISTANT",
      "content": "Here are the Electronics products: Smartphone X ($999.99), Laptop Pro 15 ($1499.99), USB-C Hub ($49.99)"
    }
  ]
}
```

**Response**:
```json
{
  "turnId": "turn-3",
  "status": "success"
}
```

---

## 5. Cognito Authentication API

### Base URL
```
https://cognito-idp.us-west-2.amazonaws.com/
```

### Initiate Auth

**Action**: `InitiateAuth`

**Description**: Authenticate user and get tokens

**Request**:
```json
{
  "ClientId": "2nc8c09npb6vru63mgfl7kiqb1",
  "AuthFlow": "USER_PASSWORD_AUTH",
  "AuthParameters": {
    "USERNAME": "john.doe",
    "PASSWORD": "YourPassword123!"
  }
}
```

**Response**:
```json
{
  "AuthenticationResult": {
    "AccessToken": "eyJraWQiOiJ...",
    "IdToken": "eyJraWQiOiJ...",
    "RefreshToken": "eyJjdHkiOiJ...",
    "ExpiresIn": 3600,
    "TokenType": "Bearer"
  },
  "ChallengeParameters": {}
}
```

### Refresh Token

**Action**: `InitiateAuth`

**Request**:
```json
{
  "ClientId": "2nc8c09npb6vru63mgfl7kiqb1",
  "AuthFlow": "REFRESH_TOKEN_AUTH",
  "AuthParameters": {
    "REFRESH_TOKEN": "eyJjdHkiOiJ..."
  }
}
```

**Response**:
```json
{
  "AuthenticationResult": {
    "AccessToken": "eyJraWQiOiJ...",
    "IdToken": "eyJraWQiOiJ...",
    "ExpiresIn": 3600,
    "TokenType": "Bearer"
  }
}
```

---

## 6. Python SDK Usage

### AgentCore Client

```python
import boto3
import json

class AgentCoreClient:
    def __init__(self, agent_id: str, region: str = 'us-west-2'):
        self.client = boto3.client('bedrock-agentcore-runtime', region_name=region)
        self.agent_id = agent_id
    
    def invoke_streaming(
        self,
        input_text: str,
        session_id: str,
        access_token: str,
        tenant_id: str = None,
        actor_id: str = None
    ):
        """Invoke agent with streaming response."""
        
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        if tenant_id:
            headers['X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId'] = tenant_id
        if actor_id:
            headers['X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId'] = actor_id
        
        response = self.client.invoke_agent(
            agentId=self.agent_id,
            sessionId=session_id,
            inputText=input_text,
            headers=headers
        )
        
        # Stream events
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    yield json.loads(chunk['bytes'].decode('utf-8'))

# Usage
client = AgentCoreClient(agent_id='capstone_ecomm_agent-iOd15C2EqB')

for chunk in client.invoke_streaming(
    input_text='Show me my orders',
    session_id='session-123',
    access_token='eyJraWQiOiJ...',
    tenant_id='tenanta',
    actor_id='john_doe'
):
    print(chunk)
```

### Memory Client

```python
from bedrock_agentcore.memory import MemoryClient

memory_client = MemoryClient(
    memory_id='memory-XPTahoCqjM',
    region='us-west-2'
)

# Get last 3 turns
turns = memory_client.get_last_k_turns(
    actor_id='john_doe',
    session_id='session-123',
    k=3
)

# Search facts
facts = memory_client.search_memory(
    namespace='/users/john_doe/facts',
    query='order information',
    max_results=5
)

# Add turn
memory_client.add_turn(
    actor_id='john_doe',
    session_id='session-123',
    messages=[
        {'role': 'USER', 'content': 'Show me products'},
        {'role': 'ASSISTANT', 'content': 'Here are the products...'}
    ]
)
```

---

## Rate Limits

| API | Rate Limit | Burst |
|-----|------------|-------|
| AgentCore Runtime | 100 req/min | 200 |
| Product Reviews Lambda | 1000 req/min | 2000 |
| Cognito Auth | 50 req/min | 100 |
| Memory API | 100 req/min | 200 |

---

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `UNAUTHORIZED` | Invalid or expired token | Refresh access token |
| `FORBIDDEN` | Insufficient permissions | Check IAM roles |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Implement exponential backoff |
| `INVALID_REQUEST` | Malformed request | Validate request parameters |
| `INTERNAL_ERROR` | Server error | Retry with exponential backoff |
| `TENANT_NOT_FOUND` | Invalid tenant ID | Verify tenant ID in JWT |
| `SESSION_EXPIRED` | Session no longer valid | Create new session |

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
