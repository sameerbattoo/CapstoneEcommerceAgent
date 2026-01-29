# ECommerce Agent - Architecture Specification

## Architecture Diagram

![Architecture Diagram](../generated-diagrams/Capstone-ECommerce-Agent%20V2.png)

## System Architecture Overview

The ECommerce Agent solution follows a multi-tier, microservices-based architecture deployed on AWS, leveraging AWS Bedrock AgentCore for intelligent agent orchestration.

## Architecture Layers

### 1. Presentation Layer

#### Streamlit Web Application
- **Component**: `ui/orch_web_app_cognito.py`
- **Purpose**: User-facing web interface for customer interactions
- **Key Features**:
  - Chat interface for natural language queries
  - Voice input via AWS Transcribe
  - Memory management UI
  - Real-time streaming responses
  - Metrics display (tokens, cost, latency)

#### Authentication Flow
```
User → Streamlit UI → AWS Cognito → JWT Token → AgentCore Runtime
```

**Components**:
- `ui/auth/cognito_auth.py`: Handles Cognito authentication
- `ui/auth/session_manager.py`: Manages user sessions
- `ui/components/login_page.py`: Login UI component

### 2. Agent Orchestration Layer

#### AWS Bedrock AgentCore Runtime
- **Deployment**: Container-based deployment to AWS
- **Configuration**: `.bedrock_agentcore.yaml`
- **Network**: VPC-enabled with private subnets
- **Authentication**: Custom JWT Authorizer (Cognito)
- **Observability**: CloudWatch integration enabled

#### Main Orchestrator Agent
- **File**: `main.py` → `agent/orch_agent.py`
- **Model**: Claude 4.5 Sonnet (anthropic.claude-3-5-sonnet-20241022-v2:0)
- **Responsibilities**:
  - Route user queries to appropriate sub-agents
  - Coordinate multi-step workflows
  - Manage conversation context
  - Aggregate responses from multiple agents

**Key Features**:
- **Memory Integration**: Strands Memory Hooks + AgentCore Short-Term Memory
- **Conversation Management**: Sliding Window (last 10 conversations)
- **Prompt Caching**: System prompt and tool definitions cached
- **Token Tracking**: Accumulates tokens across all agent invocations

### 3. Sub-Agent Layer

#### SQL Agent
- **File**: `agent/sql_agent.py`
- **Purpose**: Query structured data from RDS PostgreSQL
- **Model**: Claude 4.5 Haiku (for cost optimization)
- **Key Features**:
  - Multi-tenant schema isolation
  - Semantic caching via Valkey (ElastiCache)
  - Schema extraction and caching
  - SQL validation and execution
  - Conservative mode for read-only queries

**Semantic Caching Architecture**:
```
User Query → Embedding (Titan v2) → Vector Search (Valkey) 
    ↓ Cache Miss                    ↓ Cache Hit
SQL Generation                  Return Cached Result
    ↓
SQL Execution
    ↓
Cache Result (TTL: 10 hours)
```

**Cache Configuration**:
- **Similarity Threshold**: 0.99 (exact match)
- **Minimum Threshold**: 0.70 (partial match with user confirmation)
- **TTL**: 36,000 seconds (10 hours)
- **Embedding Model**: amazon.titan-embed-text-v2:0 (1024 dimensions)

#### Knowledge Base Agent
- **File**: `agent/kb_agent.py`
- **Purpose**: Retrieve information from unstructured documents
- **Data Source**: AWS Bedrock Knowledge Base (S3-backed)
- **Model**: Claude 4.5 Haiku
- **Key Features**:
  - Tenant-based metadata filtering
  - Retrieve + Generate pattern for accurate token tracking
  - Citation extraction and formatting
  - Configurable result limits (default: 5 documents)

**Multi-Tenancy**:
- PDF files stored in S3 with `.metadata.json` files
- Metadata includes `tenant_access` field
- Filter applied at retrieval time: `tenant_access = {tenant_id}`

#### Chart Agent
- **File**: `agent/chart_agent.py`
- **Purpose**: Generate visualizations from SQL query results
- **Tool**: AWS Bedrock Code Interpreter
- **Model**: Claude 4.5 Haiku
- **Key Features**:
  - Python code generation for matplotlib/seaborn charts
  - Sandboxed execution via Code Interpreter
  - S3 upload with CloudFront URLs
  - Automatic chart type selection based on data

**Chart Generation Flow**:
```
SQL Results → Chart Agent → Python Code → Code Interpreter 
    → PNG Image → S3 Upload → CloudFront URL → User
```

### 4. Data Layer

#### Structured Data - Amazon RDS PostgreSQL
- **Multi-Tenancy Strategy**: Schema-per-tenant
- **Schemas**: `tenanta`, `tenantb`, etc.
- **Connection**: Connection pooling (ThreadedConnectionPool)
  - Min connections: 2
  - Max connections: 10
- **Tables**: customers, products, orders, order_items, categories, shipments

**Schema Caching**:
- Background worker pre-loads all tenant schemas on startup
- Cache stored in-memory: `{tenant_id: {table_name: CREATE_STATEMENT}}`
- Reduces SQL Agent latency by avoiding runtime schema extraction

#### Semi-Structured Data - Amazon DynamoDB
- **Table**: ProductReviews
- **Multi-Tenancy Strategy**: Tenant ID attribute filtering
- **Partition Key**: `product_id` (Number)
- **Sort Key**: `review_id` (String)
- **GSI**: `tenant_id-product_id-index` for efficient tenant queries
- **Attributes**: customer_id, customer_email, product_name, rating, title, comment, review_date, helpful_votes, verified_purchase, tenant_id

**Access Pattern**:
```
Lambda Function → DynamoDB Query (tenant_id filter) → Results
```

#### Unstructured Data - Amazon S3 + Bedrock Knowledge Base
- **Bucket**: Product specification PDFs
- **Multi-Tenancy**: Metadata-based filtering
- **Metadata File Format**: `{filename}.pdf.metadata.json`
```json
{
  "metadataAttributes": {
    "tenant_access": "tenanta"
  }
}
```
- **Knowledge Base**: Ingests PDFs with metadata
- **Retrieval**: Applies filter `tenant_access = {tenant_id}`

#### Cache Layer - Amazon ElastiCache (Valkey)
- **Purpose**: Semantic caching for SQL queries
- **Engine**: Valkey (Redis-compatible)
- **Configuration**:
  - TLS enabled
  - Cluster mode disabled
  - Node type: cache.t3.medium (or higher)
- **Data Structure**:
  - Key: `sql_cache:{tenant_id}:{query_hash}`
  - Value: JSON with SQL, results, chart URL, embedding vector

### 5. Integration Layer

#### AWS Lambda - Product Reviews API
- **File**: `lambda/product_reviews_api.py`
- **Trigger**: Lambda Function URL with AWS_IAM auth
- **Purpose**: Query DynamoDB with fine-grained access control
- **Authentication**: Bearer token passed from AgentCore Gateway

**Fine-Grained Access Control**:
```
AgentCore Runtime → AgentCore Gateway → Lambda (with tenant_id header)
```

**Custom Headers Propagated**:
- `X-Amzn-Bedrock-AgentCore-Runtime-Custom-TenantId`
- `X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId`

**Lambda Context Extraction**:
```python
client_ctx = context.client_context
tenant_id = client_ctx.custom.get("bedrockAgentCorePropagatedHeaders", {})
    .get("x-amzn-bedrock-agentcore-runtime-custom-tenantid")
```

#### MCP (Model Context Protocol) Integration
- **Framework**: Strands MCP Client
- **Tool**: Product Reviews API wrapped as MCP tool
- **Configuration**: Defined in orchestrator agent
- **Benefits**: Standardized tool interface, automatic schema validation

### 6. Memory & State Management

#### Short-Term Memory (AgentCore Memory)
- **Type**: Conversation history
- **Storage**: Last 3 conversation turns loaded on agent initialization
- **Namespace**: `/sessions/{session_id}`
- **Hook**: `ECommerceMemoryHook` (on_agent_initialized, on_message_added)

#### Long-Term Memory Strategies
1. **OrderFactsExtractor** (Semantic Strategy)
   - **Namespace**: `/users/{actorId}/facts`
   - **Purpose**: Extract order IDs, product details, shipment info
   - **Model**: Claude Haiku 4.5

2. **CustomerPreferences** (User Preference Strategy)
   - **Namespace**: `/users/{actorId}/preferences`
   - **Purpose**: Learn product preferences, shopping patterns
   - **Model**: Claude Haiku 4.5

**Memory Retrieval**:
```python
# Load last 3 turns
recent_turns = memory_client.get_last_k_turns(k=3)

# Load facts
facts = memory_client.search_memory(
    namespace="/users/{actorId}/facts",
    query="order information",
    max_results=5
)

# Load preferences
preferences = memory_client.search_memory(
    namespace="/users/{actorId}/preferences",
    query="product preferences",
    max_results=5
)
```

### 7. Security & Authentication

#### Inbound Authentication (User → AgentCore)
- **Provider**: AWS Cognito User Pool
- **Flow**: OAuth2 Authorization Code Flow
- **Token**: JWT Access Token
- **Custom Claims**: `tenantId`, `username`

**Pre-Token Generation Lambda**:
- **File**: `lambda/CapstoneEcommerceCognitoPreTokenGeneration.py`
- **Purpose**: Inject `tenantId` into JWT token
- **Trigger**: Cognito Pre Token Generation

#### Outbound Authentication (AgentCore → Lambda)
- **Method**: Bearer token propagation
- **Gateway**: AgentCore Gateway with fine-grained access control
- **Headers**: Custom tenant/actor headers propagated to Lambda

#### IAM Permissions
- **Execution Role**: `CapstoneECommerce-agentcore-execution-role`
- **Permissions**:
  - Bedrock model invocation
  - DynamoDB read access
  - RDS connectivity (via VPC)
  - S3 read/write (charts, KB documents)
  - Secrets Manager read
  - CloudWatch logs/metrics
  - Lambda invoke (Function URL)
  - AgentCore Memory operations

### 8. Observability & Monitoring

#### CloudWatch Metrics
- **Namespace**: `CapstoneECommerceAgent/PerTenant`
- **Dimensions**: `TenantId`, `SessionId`, `StepName`
- **Metrics**:
  - `Latency` (milliseconds)
  - `InputTokens`, `OutputTokens`
  - `CacheReadTokens`, `CacheWriteTokens`
  - `Cost` (USD)
  - `SemanticCacheHits`

**Metrics Logger**: `metrics_logger.py`
```python
logger.emit_step_metrics(
    session_id=session_id,
    tenant_id=tenant_id,
    step_name="sql_agent_execution",
    start_time=start_time,
    end_time=end_time,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    cache_read_tokens=cache_read_tokens,
    cache_write_tokens=cache_write_tokens,
    status="success"
)
```

#### CloudWatch Logs
- **Log Groups**:
  - `/aws/bedrock/agentcore/runtime/{agent_id}`
  - `/aws/lambda/ProductReviewsAPI`
- **Log Level**: Configurable via `LOG_LEVEL` env var (INFO, WARNING, ERROR)

#### Cost Tracking
- **Per-Tenant Cost Calculation**:
  - Input tokens: $0.003 per 1K tokens (Claude Sonnet 4.5)
  - Output tokens: $0.015 per 1K tokens
  - Cache read: $0.0003 per 1K tokens (90% savings)
  - Cache write: $0.00375 per 1K tokens (25% markup)

### 9. Performance Optimizations

#### 1. Prompt Caching
- **System Prompt**: Cached with checkpoint marker
- **Tool Definitions**: Cached with checkpoint marker
- **Savings**: 90% reduction on cached input tokens
- **Minimum**: 1,024 tokens required for Claude Sonnet 4.5

#### 2. Semantic Caching (SQL Agent)
- **Cache Hit Rate**: ~40-60% for similar queries
- **Latency Reduction**: 2-3 seconds saved per cache hit
- **Cost Savings**: Eliminates SQL generation + execution

#### 3. Sliding Window Conversation Manager
- **Window Size**: Last 10 conversations
- **Purpose**: Reduce input token count
- **Implementation**: Strands `SlidingWindowConversationManager`

#### 4. Schema Caching
- **Background Worker**: Pre-loads all tenant schemas on startup
- **Cache Duration**: In-memory for application lifetime
- **Benefit**: Eliminates 1-2 second schema extraction per query

#### 5. Connection Pooling
- **Database**: ThreadedConnectionPool (2-10 connections)
- **Reuse**: Connections reused across requests
- **Benefit**: Eliminates connection overhead (~500ms)

### 10. Deployment Architecture

#### VPC Configuration
- **Subnets**: 2 private subnets (multi-AZ)
- **NAT Gateway**: For outbound internet access
- **Security Groups**:
  - AgentCore Runtime SG (allows outbound to RDS, Lambda, Bedrock)
  - RDS SG (allows inbound from AgentCore SG on port 5432)

#### Container Deployment
- **Platform**: linux/arm64 (Graviton)
- **Deployment Type**: Container (Docker)
- **ECR Repository**: Auto-created by AgentCore toolkit
- **CodeBuild**: Automated build pipeline

#### Configuration Management
- **Secrets Manager**: `capstone-ecommerce-agent-config`
- **Environment Variables**: Loaded from Secrets Manager on startup
- **Fallback**: `.env` file for local development

**Configuration Keys**:
```json
{
  "AWS_REGION": "us-west-2",
  "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "KB_ID": "knowledge-base-id",
  "DB_HOST": "rds-endpoint",
  "DB_PORT": "5432",
  "DB_NAME": "ecommerce",
  "DB_USER": "admin",
  "DB_PASSWORD": "password",
  "AGENTCORE_MEMORY_ID": "memory-id",
  "GATEWAY_URL": "gateway-url",
  "CHART_S3_BUCKET": "chart-bucket",
  "CLOUDFRONT_DOMAIN": "cloudfront-domain",
  "VALKEY_ENDPOINT": "valkey-endpoint",
  "VALKEY_PORT": "6379",
  "VALKEY_USE_TLS": "true",
  "CACHE_TTL_SECONDS": "36000",
  "CACHE_SIMILARITY_THRESHOLD": "0.99",
  "BEDROCK_EMBED_MODEL": "amazon.titan-embed-text-v2:0",
  "METRICS_NAMESPACE": "CapstoneECommerceAgent/PerTenant",
  "ENABLE_METRICS_LOGGING": "true",
  "LOG_LEVEL": "WARNING"
}
```

## Data Flow Diagrams

### Query Processing Flow
```
User Query → Streamlit UI → AgentCore Runtime → Orchestrator Agent
    ↓
Orchestrator analyzes query intent
    ↓
    ├─→ SQL Agent (structured data)
    │   ├─→ Check Valkey cache
    │   ├─→ Generate SQL (if cache miss)
    │   ├─→ Execute SQL on RDS
    │   ├─→ Cache result
    │   └─→ Call Chart Agent (if visualization needed)
    │       └─→ Generate chart → S3 → CloudFront URL
    │
    ├─→ KB Agent (unstructured data)
    │   └─→ Bedrock KB retrieve + generate
    │
    └─→ Reviews Tool (semi-structured data)
        └─→ Lambda → DynamoDB query
    ↓
Orchestrator aggregates responses
    ↓
Stream response to UI
```

### Multi-Tenant Request Flow
```
User (Tenant A) → Cognito Auth → JWT (tenantId=tenanta)
    ↓
AgentCore Runtime (validates JWT)
    ↓
Extract tenantId from JWT claims
    ↓
    ├─→ SQL Agent: Use schema "tenanta"
    ├─→ KB Agent: Filter by tenant_access="tenanta"
    └─→ Lambda: Pass tenantId header → DynamoDB filter
    ↓
Tenant-isolated results returned
```

## Scalability Considerations

### Horizontal Scaling
- **AgentCore Runtime**: Auto-scales based on request load
- **RDS**: Read replicas for read-heavy workloads
- **DynamoDB**: On-demand capacity mode
- **Lambda**: Concurrent execution (up to 1000)

### Vertical Scaling
- **RDS Instance**: db.t3.medium → db.r5.large
- **ElastiCache**: cache.t3.medium → cache.r5.large
- **AgentCore**: Increase memory/CPU allocation

### Cost Optimization
- **Model Selection**: Haiku for sub-agents (5x cheaper than Sonnet)
- **Prompt Caching**: 90% savings on repeated prompts
- **Semantic Caching**: Eliminates redundant LLM calls
- **CloudFront**: Reduces S3 request costs

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
