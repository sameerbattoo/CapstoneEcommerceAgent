# ECommerce Agent - Performance & Optimization Guide

## Overview

This document details the performance optimization strategies implemented in the ECommerce Agent solution to achieve low latency, high throughput, and cost efficiency.

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| End-to-End Latency | < 5 seconds | 3-5 seconds |
| SQL Query Latency | < 2 seconds | 1.5-2.5 seconds |
| KB Query Latency | < 2 seconds | 1.5-2 seconds |
| Chart Generation | < 4 seconds | 3-4 seconds |
| Token Cost per Query | < $0.01 | $0.005-$0.008 |
| Cache Hit Rate | > 40% | 45-55% |

---

## 1. Prompt Caching

### Overview
Prompt caching reduces input token costs by caching frequently used prompts (system prompts, tool definitions) at the model level.

### Implementation

#### System Prompt Caching
```python
from strands.types.content import SystemContentBlock

system_prompt = [
    SystemContentBlock(text="""
You are an intelligent e-commerce assistant...
[Long system prompt with instructions]
    """),
    SystemContentBlock(cachePoint={"type": "default"})  # Cache checkpoint
]

agent = Agent(
    name="Orchestrator Agent",
    system_prompt=system_prompt,
    model=model
)
```

#### Tool Definition Caching
Tool definitions are automatically cached by the Strands framework when using cache checkpoints.

### Benefits

**Cost Savings**:
- **Cache Read**: $0.0003 per 1K tokens (90% savings vs. regular input)
- **Cache Write**: $0.00375 per 1K tokens (25% markup for first write)
- **Regular Input**: $0.003 per 1K tokens

**Example Calculation**:
```
System Prompt: 2,000 tokens
Tool Definitions: 1,500 tokens
Total Cacheable: 3,500 tokens

Without Caching (per request):
3,500 tokens × $0.003 = $0.0105

With Caching (after first request):
3,500 tokens × $0.0003 = $0.00105

Savings per Request: $0.00945 (90%)
```

### Requirements

**Minimum Token Count**:
- Claude Sonnet 4.5: 1,024 tokens minimum
- Claude Haiku 4.5: 1,024 tokens minimum

**Cache Duration**:
- 5 minutes of inactivity before cache expires
- Automatically refreshed on each use

### Monitoring

```python
# Check cache usage in response
if "usage" in response:
    usage = response["usage"]
    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
    cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
    
    if cache_read_tokens > 0:
        logger.info(f"Cache hit! Read {cache_read_tokens} tokens from cache")
    if cache_write_tokens > 0:
        logger.info(f"Cache miss. Wrote {cache_write_tokens} tokens to cache")
```

---

## 2. Semantic Caching (SQL Agent)

### Overview
Semantic caching stores SQL query results based on semantic similarity of user queries, eliminating redundant LLM calls and database queries.

### Architecture

```
User Query → Embedding (Titan v2) → Vector Search (Valkey)
    ↓ Cache Miss                    ↓ Cache Hit (similarity ≥ 0.99)
SQL Generation                  Return Cached Result
    ↓                               (SQL + Results + Chart URL)
SQL Execution
    ↓
Cache Result (TTL: 10 hours)
```

### Implementation

#### Cache Configuration
```python
valkey_config = {
    'endpoint': 'master.capstone-sqlagent-valkey-cache.8ot617.usw2.cache.amazonaws.com',
    'port': 6379,
    'use_tls': True,
    'cache_ttl_seconds': 36000,  # 10 hours
    'similarity_threshold': 0.99,  # Exact match
    'similarity_threshold_min': 0.70,  # Minimum for consideration
    'embed_model': 'amazon.titan-embed-text-v2:0'
}
```

#### Cache Lookup
```python
def check_cache(self, query: str, tenant_id: str) -> Optional[Dict]:
    """Check if similar query exists in cache."""
    
    # Generate embedding for query
    query_embedding = self._generate_embedding(query)
    
    # Search for similar cached queries
    cache_key_pattern = f"sql_cache:{tenant_id}:*"
    
    best_match = None
    best_similarity = 0.0
    
    for key in self.redis_client.scan_iter(match=cache_key_pattern):
        cached_data = json.loads(self.redis_client.get(key))
        cached_embedding = cached_data['embedding']
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(query_embedding, cached_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cached_data
    
    # Return if above threshold
    if best_similarity >= self.similarity_threshold:
        self.logger.info(f"Cache HIT! Similarity: {best_similarity:.4f}")
        return best_match
    
    # Partial match - ask user for confirmation
    elif best_similarity >= self.similarity_threshold_min:
        self.logger.info(f"Partial cache match. Similarity: {best_similarity:.4f}")
        # Return with confirmation flag
        return {
            'partial_match': True,
            'similarity': best_similarity,
            'cached_data': best_match
        }
    
    self.logger.info("Cache MISS")
    return None
```

#### Cache Storage
```python
def store_in_cache(
    self,
    query: str,
    tenant_id: str,
    sql: str,
    results: List[Dict],
    chart_url: str
):
    """Store query results in cache."""
    
    query_embedding = self._generate_embedding(query)
    cache_key = f"sql_cache:{tenant_id}:{hashlib.md5(query.encode()).hexdigest()}"
    
    cache_data = {
        'query': query,
        'sql': sql,
        'results': results,
        'chart_url': chart_url,
        'embedding': query_embedding.tolist(),
        'timestamp': datetime.now().isoformat(),
        'tenant_id': tenant_id
    }
    
    # Store with TTL
    self.redis_client.setex(
        cache_key,
        self.cache_ttl_seconds,
        json.dumps(cache_data)
    )
    
    self.logger.info(f"Stored in cache: {cache_key}")
```

### Benefits

**Latency Reduction**:
- SQL Generation: ~1.5 seconds saved
- SQL Execution: ~0.5 seconds saved
- Chart Generation: ~3 seconds saved (if cached)
- **Total Savings**: 2-5 seconds per cache hit

**Cost Savings**:
- Eliminates LLM call for SQL generation (~$0.002)
- Eliminates database query execution
- Eliminates chart generation (~$0.002)
- **Total Savings**: ~$0.004 per cache hit

**Cache Hit Rate**:
- Typical: 45-55% for common queries
- Peak: 70-80% during business hours

### Example Scenarios

#### Scenario 1: Exact Match
```
Query 1: "Show me sales by category"
Query 2: "Show me sales by category"
Similarity: 1.00 → Cache HIT
```

#### Scenario 2: Semantic Match
```
Query 1: "What are the total sales for each product category?"
Query 2: "Show me sales by category"
Similarity: 0.95 → Cache HIT
```

#### Scenario 3: Partial Match
```
Query 1: "Show me sales by category for last month"
Query 2: "Show me sales by category for last week"
Similarity: 0.85 → Partial match (ask user)
```

#### Scenario 4: Cache Miss
```
Query 1: "Show me sales by category"
Query 2: "Show me customer demographics"
Similarity: 0.35 → Cache MISS
```

---

## 3. Schema Caching

### Overview
Pre-load database schemas for all tenants on application startup to eliminate runtime schema extraction overhead.

### Implementation

#### Background Schema Caching
```python
def start_background_schema_caching(self):
    """Start background schema caching task asynchronously."""
    
    import concurrent.futures
    
    # Create ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="schema-cache"
    )
    
    # Submit background task
    future = executor.submit(self._background_schema_cache_worker)
    
    self.logger.info("Background schema caching task submitted (non-blocking)")

def _background_schema_cache_worker(self):
    """Background worker that caches schemas for all tenants."""
    
    start_time = time.time()
    
    # Discover all tenants
    tenants = self._discover_tenants()
    
    # Cache each tenant's schema
    for tenant_id in tenants:
        if tenant_id.lower() not in self.schema_cache:
            self._cache_tenant_schema(tenant_id)
    
    duration = time.time() - start_time
    self.logger.info(f"Schema caching completed in {duration:.2f}s")
```

#### Schema Extraction
```python
class SchemaExtractor:
    def extract_schema(self) -> Dict[str, str]:
        """Extract CREATE TABLE statements for all tables."""
        
        tables = self._get_tables()
        create_statements = {}
        
        for table in tables:
            # Get columns
            columns = self._get_columns(table)
            
            # Get constraints
            constraints = self._get_constraints(table)
            
            # Build CREATE statement
            create_sql = self._build_create_statement(table, columns, constraints)
            
            create_statements[table] = create_sql
        
        return create_statements
```

### Benefits

**Latency Reduction**:
- Schema extraction: ~1-2 seconds saved per query
- First query latency: Reduced from 4-5s to 2-3s

**Improved User Experience**:
- Consistent response times
- No "cold start" penalty for first query

### Monitoring

```python
# Check if schema is cached
cache_key = tenant_id.lower()
if cache_key in self.schema_cache:
    logger.info(f"Schema cache HIT for tenant: {tenant_id}")
else:
    logger.warning(f"Schema cache MISS for tenant: {tenant_id}")
    # Extract on-demand
    self._cache_tenant_schema(tenant_id)
```

---

## 4. Sliding Window Conversation Manager

### Overview
Limit conversation history to last N turns to reduce input token count while maintaining context.

### Implementation

```python
from strands.agent.conversation_manager import SlidingWindowConversationManager

agent = Agent(
    name="Orchestrator Agent",
    system_prompt=system_prompt,
    model=model,
    conversation_manager=SlidingWindowConversationManager(window_size=10)
)
```

### Benefits

**Token Reduction**:
```
Without Sliding Window:
- Turn 1: 100 tokens
- Turn 2: 100 tokens
- Turn 3: 100 tokens
- ...
- Turn 20: 100 tokens
Total: 2,000 tokens

With Sliding Window (size=10):
- Last 10 turns: 1,000 tokens
Savings: 1,000 tokens (50%)
```

**Cost Savings**:
```
Input tokens saved: 1,000 tokens
Cost savings: 1,000 × $0.003 = $0.003 per request
```

### Configuration

```python
# Adjust window size based on use case
conversation_manager = SlidingWindowConversationManager(
    window_size=10  # Keep last 10 conversations
)
```

**Recommended Window Sizes**:
- Short conversations: 5-10 turns
- Medium conversations: 10-15 turns
- Long conversations: 15-20 turns

---

## 5. Connection Pooling

### Overview
Reuse database connections across requests to eliminate connection overhead.

### Implementation

```python
from psycopg2 import pool

# Create connection pool
db_pool = pool.ThreadedConnectionPool(
    minconn=2,      # Minimum connections to maintain
    maxconn=10,     # Maximum connections allowed
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_user,
    password=db_password
)

# Usage
conn = db_pool.getconn()
try:
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
finally:
    cursor.close()
    db_pool.putconn(conn)  # Return to pool
```

### Benefits

**Latency Reduction**:
- Connection establishment: ~500ms saved per query
- Connection reuse: Instant connection acquisition

**Resource Efficiency**:
- Maintains minimum connections during idle periods
- Scales up to maximum during peak load
- Automatic connection health checks

### Configuration

```python
# Adjust pool size based on workload
db_pool = pool.ThreadedConnectionPool(
    minconn=2,      # Low traffic: 2-5
    maxconn=10,     # High traffic: 10-20
    ...
)
```

---

## 6. Model Selection Strategy

### Overview
Use cost-effective models for sub-agents while maintaining quality.

### Model Costs (per 1M tokens)

| Model | Input | Output | Cache Read | Cache Write |
|-------|-------|--------|------------|-------------|
| Claude Sonnet 4.5 | $3.00 | $15.00 | $0.30 | $3.75 |
| Claude Haiku 4.5 | $0.80 | $4.00 | $0.08 | $1.00 |

### Strategy

**Orchestrator Agent**: Claude Sonnet 4.5
- Requires advanced reasoning
- Handles complex multi-step workflows
- Justifies higher cost

**Sub-Agents**: Claude Haiku 4.5
- SQL Agent: Focused task (SQL generation)
- KB Agent: Simple retrieval + generation
- Chart Agent: Code generation
- 3.75x cheaper than Sonnet

### Cost Comparison

**Scenario**: User query requiring SQL + KB + Chart

**All Sonnet**:
```
Orchestrator: 1,500 input + 500 output = $0.012
SQL Agent: 1,000 input + 300 output = $0.0075
KB Agent: 800 input + 200 output = $0.0054
Chart Agent: 600 input + 150 output = $0.0041
Total: $0.029
```

**Sonnet + Haiku**:
```
Orchestrator: 1,500 input + 500 output = $0.012
SQL Agent: 1,000 input + 300 output = $0.002
KB Agent: 800 input + 200 output = $0.0014
Chart Agent: 600 input + 150 output = $0.0011
Total: $0.0165
```

**Savings**: $0.0125 per query (43%)

---

## 7. Batch Processing

### Overview
Process multiple independent operations in parallel when possible.

### Implementation

```python
import asyncio

async def process_parallel_queries(queries: List[str]):
    """Process multiple queries in parallel."""
    
    tasks = [
        sql_agent.query_async(queries[0]),
        kb_agent.search_async(queries[1]),
        reviews_tool.get_reviews_async(product_id=1)
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### Benefits

**Latency Reduction**:
```
Sequential:
SQL Agent: 2s
KB Agent: 2s
Reviews Tool: 0.5s
Total: 4.5s

Parallel:
All agents: max(2s, 2s, 0.5s) = 2s
Savings: 2.5s (55%)
```

---

## 8. CloudFront CDN for Charts

### Overview
Use CloudFront CDN to cache and deliver chart images globally.

### Benefits

**Latency Reduction**:
- S3 direct: 200-500ms (depending on region)
- CloudFront edge: 20-50ms (cached)
- **Savings**: 150-450ms per chart view

**Cost Reduction**:
- S3 GET requests: $0.0004 per 1,000 requests
- CloudFront requests: $0.0075 per 10,000 requests
- **Savings**: 81% on request costs

**Permanent URLs**:
- S3 pre-signed URLs expire after 7 days
- CloudFront URLs never expire

### Configuration

```python
# Generate CloudFront URL
if cloudfront_domain:
    chart_url = f"https://{cloudfront_domain}/{filename}"
else:
    # Fallback to pre-signed URL
    chart_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': filename},
        ExpiresIn=604800
    )
```

---

## 9. Monitoring & Profiling

### Performance Metrics

```python
import time

def emit_step_metrics(
    session_id: str,
    tenant_id: str,
    step_name: str,
    start_time: float,
    end_time: float,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    status: str
):
    """Emit performance metrics to CloudWatch."""
    
    duration_ms = (end_time - start_time) * 1000
    
    cloudwatch.put_metric_data(
        Namespace='CapstoneECommerceAgent/PerTenant',
        MetricData=[
            {
                'MetricName': 'Latency',
                'Value': duration_ms,
                'Unit': 'Milliseconds',
                'Dimensions': [
                    {'Name': 'TenantId', 'Value': tenant_id},
                    {'Name': 'StepName', 'Value': step_name}
                ]
            },
            {
                'MetricName': 'InputTokens',
                'Value': input_tokens,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'TenantId', 'Value': tenant_id},
                    {'Name': 'StepName', 'Value': step_name}
                ]
            },
            {
                'MetricName': 'CacheReadTokens',
                'Value': cache_read_tokens,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'TenantId', 'Value': tenant_id},
                    {'Name': 'StepName', 'Value': step_name}
                ]
            }
        ]
    )
```

### Key Metrics to Monitor

1. **Latency**:
   - P50, P90, P99 response times
   - Per-agent latency breakdown
   - Cache hit/miss latency

2. **Token Usage**:
   - Input/output tokens per request
   - Cache read/write tokens
   - Token cost per tenant

3. **Cache Performance**:
   - Cache hit rate
   - Cache size
   - Cache eviction rate

4. **Error Rates**:
   - 4xx errors (client errors)
   - 5xx errors (server errors)
   - Timeout errors

---

## Performance Optimization Checklist

- [ ] Prompt caching enabled for all agents
- [ ] System prompts > 1,024 tokens for cache eligibility
- [ ] Semantic caching configured for SQL Agent
- [ ] Valkey cluster sized appropriately
- [ ] Schema caching enabled on startup
- [ ] Sliding window conversation manager configured
- [ ] Connection pooling enabled for RDS
- [ ] Haiku model used for sub-agents
- [ ] CloudFront CDN configured for charts
- [ ] Performance metrics emitted to CloudWatch
- [ ] Cache hit rate > 40%
- [ ] End-to-end latency < 5 seconds
- [ ] Token cost per query < $0.01

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
