# ECommerce Agent - Multi-Agent Design Specification

## Multi-Agent Architecture Overview

The ECommerce Agent solution implements a hierarchical multi-agent architecture where a central Orchestrator Agent coordinates specialized sub-agents, each optimized for specific data sources and tasks.

## Agent Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                  Orchestrator Agent                      │
│              (Claude 4.5 Sonnet)                        │
│  - Route queries to appropriate agents                  │
│  - Coordinate multi-step workflows                      │
│  - Aggregate responses                                  │
│  - Manage conversation context                          │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────────┐
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  SQL Agent   │  │   KB Agent   │  │ Chart Agent  │  │ Reviews Tool │
│   (Haiku)    │  │   (Haiku)    │  │   (Haiku)    │  │   (Lambda)   │
│              │  │              │  │              │  │              │
│ RDS queries  │  │ PDF search   │  │ Visualize    │  │ DynamoDB     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

## Agent Specifications

### 1. Orchestrator Agent

**File**: `agent/orch_agent.py`  
**Model**: Claude 4.5 Sonnet (`anthropic.claude-3-5-sonnet-20241022-v2:0`)  
**Framework**: Strands Agent SDK

#### Responsibilities
1. **Query Intent Analysis**: Determine which agent(s) to invoke
2. **Workflow Coordination**: Manage multi-step processes
3. **Response Aggregation**: Combine results from multiple agents
4. **Context Management**: Maintain conversation history
5. **Memory Integration**: Load and persist conversation context

#### System Prompt Structure
```python
system_prompt = [
    SystemContentBlock(text="""
You are an intelligent e-commerce assistant helping customers with:
- Product information and specifications
- Order tracking and history
- Product reviews and ratings
- Data visualization and analytics

You have access to specialized tools and agents:
1. SQL Agent: Query structured data (orders, products, customers)
2. Knowledge Base Agent: Search product specifications
3. Chart Agent: Generate visualizations
4. Product Reviews API: Retrieve customer reviews

Guidelines:
- Always identify yourself as an e-commerce assistant
- Use the appropriate tool/agent for each query type
- Provide clear, concise, and helpful responses
- Cite sources when using Knowledge Base
- Offer visualizations when data is numerical
    """),
    SystemContentBlock(cachePoint={"type": "default"})  # Cache checkpoint
]
```

#### Tools Available
1. **get_answers_for_structured_data()**: Invokes SQL Agent
2. **get_answers_for_unstructured_data()**: Invokes KB Agent
3. **get_product_reviews()**: Calls Lambda via MCP
4. **current_time()**: Built-in utility

#### Memory Hook Integration
```python
class ECommerceMemoryHook(HookProvider):
    def on_agent_initialized(self, event):
        # Load last 3 conversation turns
        recent_turns = memory_client.get_last_k_turns(k=3)
        
        # Load user facts
        facts = memory_client.search_memory(
            namespace="/users/{actorId}/facts",
            max_results=5
        )
        
        # Load user preferences
        preferences = memory_client.search_memory(
            namespace="/users/{actorId}/preferences",
            max_results=5
        )
        
        # Inject into conversation context
        
    def on_message_added(self, event):
        # Persist conversation turn to AgentCore Memory
        memory_client.add_turn(...)
```

#### Conversation Management
- **Strategy**: Sliding Window (last 10 conversations)
- **Implementation**: `SlidingWindowConversationManager`
- **Purpose**: Reduce input token count while maintaining context

#### Token Accumulation
```python
def token_accumulator_callback(
    input_tokens: int,
    output_tokens: int,
    step_name: str,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0
):
    """Accumulate tokens from all agent invocations"""
    session_metrics["total_input_tokens"] += input_tokens
    session_metrics["total_output_tokens"] += output_tokens
    session_metrics["total_cache_read_tokens"] += cache_read_tokens
    session_metrics["total_cache_write_tokens"] += cache_write_tokens
    session_metrics["step_count"] += 1
```

---

### 2. SQL Agent

**File**: `agent/sql_agent.py`  
**Model**: Claude 4.5 Haiku (`anthropic.claude-haiku-4-5-20251001-v1:0`)  
**Purpose**: Query structured data from RDS PostgreSQL

#### Key Features

##### 1. Multi-Tenant Schema Isolation
```python
def extract_tenant_from_token(access_token: str) -> str:
    """Extract tenant ID from JWT token"""
    claims = jwt.decode(token, options={"verify_signature": False})
    return claims.get("tenantId", "N/A")

def build_sql_with_tenant(sql: str, tenant_id: str) -> str:
    """Prefix table names with tenant schema"""
    # Example: SELECT * FROM products
    # Becomes: SELECT * FROM tenanta.products
    return sql.replace("FROM ", f"FROM {tenant_id}.")
```

##### 2. Semantic Caching with Valkey

**Cache Architecture**:
```python
class ValkeyCache:
    def __init__(self, valkey_config):
        self.redis_client = redis.Redis(
            host=valkey_config['endpoint'],
            port=valkey_config['port'],
            ssl=True,
            decode_responses=False
        )
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.similarity_threshold = 0.99
        self.cache_ttl = 36000  # 10 hours
```

**Cache Lookup Flow**:
```python
def check_cache(query: str, tenant_id: str) -> Optional[Dict]:
    # 1. Generate embedding for query
    embedding = generate_embedding(query)
    
    # 2. Search for similar cached queries
    cache_key_pattern = f"sql_cache:{tenant_id}:*"
    for key in redis_client.scan_iter(match=cache_key_pattern):
        cached_data = redis_client.get(key)
        cached_embedding = cached_data['embedding']
        
        # 3. Calculate cosine similarity
        similarity = cosine_similarity(embedding, cached_embedding)
        
        # 4. Return if above threshold
        if similarity >= self.similarity_threshold:
            return cached_data['result']
    
    return None  # Cache miss
```

**Cache Storage**:
```python
def store_in_cache(query: str, tenant_id: str, sql: str, results: List, chart_url: str):
    embedding = generate_embedding(query)
    cache_key = f"sql_cache:{tenant_id}:{hash(query)}"
    
    cache_data = {
        'query': query,
        'sql': sql,
        'results': results,
        'chart_url': chart_url,
        'embedding': embedding,
        'timestamp': datetime.now().isoformat()
    }
    
    redis_client.setex(
        cache_key,
        self.cache_ttl,
        json.dumps(cache_data)
    )
```

##### 3. Schema Extraction and Caching

**Background Schema Caching**:
```python
def _background_schema_cache_worker(self):
    """Pre-load schemas for all tenants on startup"""
    tenants = self._discover_tenants()  # Query information_schema
    
    for tenant_id in tenants:
        schema_extractor = SchemaExtractor(db_pool, tenant_id)
        create_statements = schema_extractor.extract_schema()
        
        # Store in memory cache
        self.schema_cache[tenant_id.lower()] = create_statements
```

**Schema Extraction**:
```python
class SchemaExtractor:
    def extract_schema(self) -> Dict[str, str]:
        """Extract CREATE TABLE statements for all tables"""
        tables = self._get_tables()
        create_statements = {}
        
        for table in tables:
            columns = self._get_columns(table)
            constraints = self._get_constraints(table)
            
            create_sql = self._build_create_statement(
                table, columns, constraints
            )
            create_statements[table] = create_sql
        
        return create_statements
```

##### 4. SQL Generation and Validation

**System Prompt**:
```python
system_prompt = f"""
You are a SQL expert for a PostgreSQL database.

Database Schema (Tenant: {tenant_id}):
{schema_info}

Guidelines:
1. Generate ONLY SELECT queries (read-only)
2. Use proper table names: {tenant_id}.table_name
3. Limit results to {MAX_ROWS} rows
4. Use proper JOIN syntax
5. Handle NULL values appropriately
6. Return valid PostgreSQL syntax

Sample Questions:
{sample_questions}
"""
```

**SQL Execution**:
```python
def execute_sql(sql: str, tenant_id: str) -> List[Dict]:
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    finally:
        cursor.close()
        db_pool.putconn(conn)
```

##### 5. Chart Agent Integration

**Automatic Visualization**:
```python
def should_generate_chart(results: List[Dict]) -> bool:
    """Determine if results warrant visualization"""
    if len(results) < 2:
        return False
    
    # Check for numeric columns
    numeric_cols = [k for k, v in results[0].items() 
                    if isinstance(v, (int, float))]
    
    return len(numeric_cols) > 0

if should_generate_chart(results):
    chart_url = chart_agent.generate_chart(
        data=results,
        query=user_query
    )
```

#### Token Tracking
```python
# Report tokens to parent orchestrator
token_callback(
    input_tokens=sql_agent_input_tokens,
    output_tokens=sql_agent_output_tokens,
    step_name="sql_agent",
    cache_read_tokens=cache_read_tokens,
    cache_write_tokens=cache_write_tokens
)
```

---

### 3. Knowledge Base Agent

**File**: `agent/kb_agent.py`  
**Model**: Claude 4.5 Haiku  
**Purpose**: Retrieve information from product specification PDFs

#### Architecture

**Retrieve + Generate Pattern**:
```python
def retrieve_and_generate(query: str, tenant_id: str) -> Dict:
    # Step 1: Retrieve relevant documents
    retrieve_response = bedrock_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': query},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 5,
                'filter': {
                    'equals': {
                        'key': 'tenant_access',
                        'value': tenant_id
                    }
                }
            }
        }
    )
    
    # Step 2: Build context from retrieved docs
    context = build_context(retrieve_response['retrievalResults'])
    
    # Step 3: Generate response with Bedrock Converse
    converse_response = bedrock_client.converse(
        modelId=model_id,
        messages=[{
            'role': 'user',
            'content': f"Context: {context}\n\nQuestion: {query}"
        }]
    )
    
    # Step 4: Extract token usage
    usage = converse_response['usage']
    
    return {
        'response': converse_response['output']['message']['content'][0]['text'],
        'citations': extract_citations(retrieve_response),
        'input_tokens': usage['inputTokens'],
        'output_tokens': usage['outputTokens']
    }
```

#### Multi-Tenancy

**Metadata Structure**:
```json
// Bluetooth Headphones - Product Spec.pdf.metadata.json
{
  "metadataAttributes": {
    "tenant_access": "tenanta",
    "product_category": "electronics",
    "product_id": "1"
  }
}
```

**Filtering at Retrieval**:
- Knowledge Base ingests PDFs with metadata
- At query time, filter applied: `tenant_access = {tenant_id}`
- Only documents accessible to tenant are retrieved

#### Citation Extraction
```python
def extract_citations(retrieved_docs: List) -> List[Dict]:
    citations = []
    for doc in retrieved_docs:
        location = doc.get('location', {})
        s3_location = location.get('s3Location', {})
        
        citations.append({
            'source': s3_location.get('uri', 'Unknown'),
            'score': doc.get('score', 0.0),
            'excerpt': doc.get('content', {}).get('text', '')[:200]
        })
    
    return citations
```

---

### 4. Chart Agent

**File**: `agent/chart_agent.py`  
**Model**: Claude 4.5 Haiku  
**Tool**: AWS Bedrock Code Interpreter  
**Purpose**: Generate visualizations from data

#### Chart Generation Flow

```python
class ChartAgent:
    def generate_chart(self, data: List[Dict], query: str) -> str:
        # Step 1: Analyze data structure
        data_summary = self._analyze_data(data)
        
        # Step 2: Generate Python code
        code_prompt = f"""
Generate Python code using matplotlib to visualize this data:

Data: {json.dumps(data[:5])}
Data Summary: {data_summary}
User Query: {query}

Requirements:
- Use matplotlib or seaborn
- Create appropriate chart type (bar, line, pie, scatter)
- Add title, labels, and legend
- Save as PNG: plt.savefig('chart.png', dpi=150, bbox_inches='tight')
"""
        
        # Step 3: Execute code in Code Interpreter
        result = self.code_interpreter.execute_code(
            code=generated_code,
            files=[]  # No input files needed
        )
        
        # Step 4: Upload to S3
        chart_url = self._upload_to_s3(
            result['output_files'][0],
            tenant_id=self.tenant_id
        )
        
        return chart_url
```

#### Code Interpreter Integration

**Execution**:
```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

code_interpreter = CodeInterpreter(region=aws_region)

result = code_interpreter.execute_code(
    code=python_code,
    files=[],  # Input files (if any)
    timeout=30  # Execution timeout
)

# Result structure:
{
    'output_files': [
        {
            'name': 'chart.png',
            'bytes': b'...',
            'type': 'image/png'
        }
    ],
    'stdout': 'Execution logs',
    'stderr': ''
}
```

#### S3 Upload with CloudFront

```python
def _upload_to_s3(self, file_data: bytes, tenant_id: str) -> str:
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"charts/{tenant_id}/{timestamp}_{uuid.uuid4().hex[:8]}.png"
    
    # Upload to S3
    s3_client.put_object(
        Bucket=self.chart_s3_bucket,
        Key=filename,
        Body=file_data,
        ContentType='image/png'
    )
    
    # Return CloudFront URL (if configured)
    if self.cloudfront_domain:
        return f"https://{self.cloudfront_domain}/{filename}"
    else:
        # Fallback to pre-signed URL
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.chart_s3_bucket, 'Key': filename},
            ExpiresIn=604800  # 7 days
        )
```

#### Chart Type Selection

**Automatic Selection Logic**:
```python
def _determine_chart_type(self, data: List[Dict]) -> str:
    """Analyze data to suggest appropriate chart type"""
    
    # Count numeric vs categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for key, value in data[0].items():
        if isinstance(value, (int, float)):
            numeric_cols.append(key)
        else:
            categorical_cols.append(key)
    
    # Decision logic
    if len(numeric_cols) == 1 and len(categorical_cols) == 1:
        return "bar"  # One category, one value
    elif len(numeric_cols) >= 2:
        return "scatter"  # Multiple numeric columns
    elif len(data) <= 10 and len(numeric_cols) == 1:
        return "pie"  # Few categories with values
    else:
        return "line"  # Time series or trends
```

---

### 5. Product Reviews Tool (MCP)

**File**: `lambda/product_reviews_api.py`  
**Integration**: MCP (Model Context Protocol) via Strands  
**Data Source**: DynamoDB

#### MCP Tool Definition

```python
# In orchestrator agent
mcp_client = MCPClient(
    server_name="product_reviews",
    server_config={
        "command": "python",
        "args": ["-m", "lambda_function_url_wrapper"],
        "env": {
            "LAMBDA_URL": gateway_url,
            "ACCESS_TOKEN": access_token
        }
    }
)

# Tool automatically registered from OpenAPI schema
```

#### Lambda Function

**Query Parameters**:
- `product_id`: Filter by product ID(s) (comma-separated)
- `customer_id`: Filter by customer ID(s)
- `customer_email`: Filter by email(s)
- `rating`: Filter by rating(s)
- `review_date_from`: Start date (YYYY-MM-DD)
- `review_date_to`: End date (YYYY-MM-DD)
- `top_rows`: Limit results (default: 20, max: 20)

**Multi-Tenancy**:
```python
def lambda_handler(event, context):
    # Extract tenant_id from AgentCore Gateway headers
    client_ctx = context.client_context
    tenant_id = client_ctx.custom.get(
        "bedrockAgentCorePropagatedHeaders", {}
    ).get("x-amzn-bedrock-agentcore-runtime-custom-tenantid")
    
    # Query DynamoDB with tenant filter
    response = table.query(
        IndexName='tenant_id-product_id-index',
        KeyConditionExpression=Key('tenant_id').eq(tenant_id),
        FilterExpression=...  # Additional filters
    )
```

---

## Agent Communication Patterns

### 1. Sequential Execution
```
Orchestrator → SQL Agent → Chart Agent → Response
```
Example: "Show me sales by category and create a chart"

### 2. Parallel Execution
```
Orchestrator → [SQL Agent, KB Agent, Reviews Tool] → Aggregate → Response
```
Example: "Tell me about product X including specs, reviews, and sales"

### 3. Conditional Execution
```
Orchestrator → SQL Agent → (if results > 0) → Chart Agent → Response
```
Example: "Show sales trends" (only chart if data exists)

### 4. Iterative Refinement
```
Orchestrator → SQL Agent → (if error) → Retry with corrected SQL → Response
```
Example: SQL syntax error handling

---

## Performance Metrics by Agent

| Agent | Avg Latency | Token Cost | Cache Hit Rate |
|-------|-------------|------------|----------------|
| Orchestrator | 1.5s | $0.005 | 40% (prompt cache) |
| SQL Agent | 2.5s | $0.002 | 50% (semantic cache) |
| KB Agent | 1.8s | $0.003 | N/A |
| Chart Agent | 3.5s | $0.002 | N/A |
| Reviews Tool | 0.5s | $0 (Lambda) | N/A |

**Total E2E**: ~5-8 seconds (depending on agents invoked)

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
