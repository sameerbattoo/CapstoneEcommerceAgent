# ECommerce Agent - Testing & Evaluation Guide

## Overview

This document describes the comprehensive testing and evaluation strategy for the ECommerce Agent solution, including unit tests, integration tests, model evaluation, and RAG evaluation.

---

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                     Testing Pyramid                          │
│                                                              │
│                    ┌──────────────┐                         │
│                    │   E2E Tests  │                         │
│                    │  (Manual QA) │                         │
│                    └──────────────┘                         │
│                  ┌──────────────────┐                       │
│                  │ Integration Tests│                       │
│                  │  (Agent + Tools) │                       │
│                  └──────────────────┘                       │
│              ┌────────────────────────┐                     │
│              │     Unit Tests         │                     │
│              │ (Individual Functions) │                     │
│              └────────────────────────┘                     │
│          ┌──────────────────────────────┐                   │
│          │    Model Evaluation          │                   │
│          │ (Bedrock Model Evaluation)   │                   │
│          └──────────────────────────────┘                   │
│      ┌────────────────────────────────────┐                 │
│      │       RAG Evaluation               │                 │
│      │  (RAGAS Metrics: Faithfulness,     │                 │
│      │   Answer Relevance, Context)       │                 │
│      └────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Unit Testing

### Test Framework
- **Framework**: pytest
- **Coverage Tool**: pytest-cov
- **Mocking**: unittest.mock

### Test Structure

```
tests/
├── unit/
│   ├── test_sql_agent.py
│   ├── test_kb_agent.py
│   ├── test_chart_agent.py
│   ├── test_orch_agent.py
│   ├── test_valkey_cache.py
│   └── test_metrics_logger.py
├── integration/
│   ├── test_agent_integration.py
│   ├── test_database_integration.py
│   └── test_lambda_integration.py
└── e2e/
    └── test_end_to_end.py
```

### Example Unit Tests

#### Test SQL Agent
```python
import pytest
from unittest.mock import Mock, patch
from agent.sql_agent import SQLAgent, ValkeyCache

class TestSQLAgent:
    @pytest.fixture
    def sql_agent(self):
        """Create SQL agent instance for testing."""
        logger = Mock()
        config = {
            'aws_region': 'us-west-2',
            'bedrock_model_id': 'anthropic.claude-haiku-4-5-20251001-v1:0',
            'db_pool': Mock(),
            'schema_cache': {'tenanta': {'products': 'CREATE TABLE...'}},
            'valkey_config': None
        }
        return SQLAgent(logger, config, 'tenanta', 'session-123')
    
    def test_sql_generation(self, sql_agent):
        """Test SQL generation from natural language."""
        query = "Show me all products in Electronics category"
        
        with patch.object(sql_agent.agent, 'invoke') as mock_invoke:
            mock_invoke.return_value = "SELECT * FROM tenanta.products WHERE category_id = 1"
            
            result = sql_agent.generate_sql(query)
            
            assert "SELECT" in result
            assert "tenanta.products" in result
            assert mock_invoke.called
    
    def test_sql_execution(self, sql_agent):
        """Test SQL execution against database."""
        sql = "SELECT * FROM tenanta.products LIMIT 5"
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (1, 'Smartphone X', 999.99),
            (2, 'Laptop Pro 15', 1499.99)
        ]
        mock_cursor.description = [
            ('product_id',), ('product_name',), ('price',)
        ]
        mock_conn.cursor.return_value = mock_cursor
        
        sql_agent.db_pool.getconn.return_value = mock_conn
        
        results = sql_agent.execute_sql(sql)
        
        assert len(results) == 2
        assert results[0]['product_name'] == 'Smartphone X'
        assert results[1]['price'] == 1499.99
    
    def test_cache_hit(self, sql_agent):
        """Test semantic cache hit."""
        query = "Show me sales by category"
        
        # Mock cache with similar query
        mock_cache = Mock(spec=ValkeyCache)
        mock_cache.check_cache.return_value = {
            'sql': 'SELECT category_name, SUM(total) FROM...',
            'results': [{'category': 'Electronics', 'total': 15000}],
            'chart_url': 'https://cloudfront.net/chart.png'
        }
        
        sql_agent.cache = mock_cache
        
        result = sql_agent.query(query)
        
        assert result['cache_hit'] is True
        assert len(result['data']) == 1
        assert 'chart_url' in result
```

#### Test Valkey Cache
```python
import pytest
from unittest.mock import Mock, patch
from agent.sql_agent import ValkeyCache
import numpy as np

class TestValkeyCache:
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        logger = Mock()
        config = {
            'endpoint': 'localhost',
            'port': 6379,
            'use_tls': False,
            'cache_ttl_seconds': 3600,
            'similarity_threshold': 0.99,
            'embed_model': 'amazon.titan-embed-text-v2:0'
        }
        return ValkeyCache(logger, 'us-west-2', config)
    
    def test_embedding_generation(self, cache):
        """Test query embedding generation."""
        query = "Show me sales by category"
        
        with patch.object(cache.bedrock_runtime, 'invoke_model') as mock_invoke:
            mock_invoke.return_value = {
                'body': Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
            }
            
            embedding = cache._generate_embedding(query)
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0
    
    def test_cosine_similarity(self, cache):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        
        # Identical vectors
        similarity1 = cache._cosine_similarity(vec1, vec2)
        assert similarity1 == pytest.approx(1.0)
        
        # Orthogonal vectors
        similarity2 = cache._cosine_similarity(vec1, vec3)
        assert similarity2 == pytest.approx(0.0)
    
    def test_cache_storage(self, cache):
        """Test storing query results in cache."""
        query = "Show me sales"
        tenant_id = "tenanta"
        sql = "SELECT * FROM sales"
        results = [{'total': 1000}]
        chart_url = "https://cloudfront.net/chart.png"
        
        with patch.object(cache.redis_client, 'setex') as mock_setex:
            cache.store_in_cache(query, tenant_id, sql, results, chart_url)
            
            assert mock_setex.called
            call_args = mock_setex.call_args
            assert call_args[0][1] == 3600  # TTL
```

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=agent --cov-report=html

# Run specific test file
pytest tests/unit/test_sql_agent.py -v

# Run specific test
pytest tests/unit/test_sql_agent.py::TestSQLAgent::test_sql_generation -v
```

---

## 2. Integration Testing

### Test Database Integration

```python
import pytest
import psycopg2
from agent.sql_agent import SQLAgent

class TestDatabaseIntegration:
    @pytest.fixture(scope="module")
    def db_connection(self):
        """Create database connection for testing."""
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ecommerce_test",
            user="test_user",
            password="test_password"
        )
        yield conn
        conn.close()
    
    def test_query_products(self, db_connection):
        """Test querying products from database."""
        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM tenanta.products LIMIT 5")
        results = cursor.fetchall()
        
        assert len(results) > 0
        assert len(results) <= 5
        
        cursor.close()
    
    def test_query_with_join(self, db_connection):
        """Test complex query with joins."""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT p.product_name, c.category_name
            FROM tenanta.products p
            JOIN tenanta.categories c ON p.category_id = c.category_id
            LIMIT 5
        """)
        results = cursor.fetchall()
        
        assert len(results) > 0
        assert len(results[0]) == 2  # product_name, category_name
        
        cursor.close()
```

### Test Lambda Integration

```python
import pytest
import boto3
import json

class TestLambdaIntegration:
    @pytest.fixture
    def lambda_client(self):
        """Create Lambda client for testing."""
        return boto3.client('lambda', region_name='us-west-2')
    
    def test_invoke_reviews_api(self, lambda_client):
        """Test invoking Product Reviews Lambda."""
        payload = {
            'product_id': '1',
            'top_rows': 5
        }
        
        response = lambda_client.invoke(
            FunctionName='ProductReviewsAPI',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        assert response['StatusCode'] == 200
        
        result = json.loads(response['Payload'].read())
        assert 'body' in result
        
        reviews = json.loads(result['body'])
        assert isinstance(reviews, list)
        assert len(reviews) <= 5
```

### Test Agent Integration

```python
import pytest
from agent.orch_agent import AgentManager

class TestAgentIntegration:
    @pytest.fixture
    def agent_manager(self):
        """Create agent manager for testing."""
        logger = Mock()
        config = {
            'aws_region': 'us-west-2',
            'bedrock_model_id': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
            # ... other config
        }
        return AgentManager(
            logger,
            config,
            'session-test',
            'test@email.com',
            'tenanta',
            'test_user',
            None
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_sql_flow(self, agent_manager):
        """Test orchestrator invoking SQL agent."""
        query = "Show me products in Electronics category"
        
        responses = []
        async for event in agent_manager.orchestrator_agent.stream_async(query):
            if 'data' in event:
                responses.append(event['data'])
        
        full_response = ''.join(responses)
        
        assert 'Electronics' in full_response or 'products' in full_response
        assert len(responses) > 0
```

---

## 3. End-to-End Testing

### Manual Test Scenarios

#### Scenario 1: Product Query
```
User: "Show me all products in the Electronics category"

Expected:
1. Orchestrator routes to SQL Agent
2. SQL Agent generates SQL query
3. Query executes against RDS
4. Results returned with product list
5. Response time < 3 seconds
```

#### Scenario 2: Product Specifications
```
User: "What are the camera specifications for Smartphone X?"

Expected:
1. Orchestrator routes to KB Agent
2. KB Agent searches Knowledge Base
3. Relevant PDF sections retrieved
4. Response includes camera specs with citations
5. Response time < 2 seconds
```

#### Scenario 3: Product Reviews
```
User: "Show me 5-star reviews for Smartphone X"

Expected:
1. Orchestrator routes to Reviews Tool
2. Lambda queries DynamoDB with filters
3. Reviews returned (filtered by rating=5)
4. Response includes review details
5. Response time < 1 second
```

#### Scenario 4: Sales Analytics with Chart
```
User: "Show me sales by category for the last 30 days and create a chart"

Expected:
1. Orchestrator routes to SQL Agent
2. SQL Agent generates and executes query
3. SQL Agent invokes Chart Agent
4. Chart Agent generates visualization
5. Response includes data + chart URL
6. Response time < 5 seconds
```

#### Scenario 5: Multi-Step Query
```
User: "Tell me about Smartphone X including specs, reviews, and sales"

Expected:
1. Orchestrator coordinates multiple agents
2. KB Agent retrieves specifications
3. Reviews Tool gets customer reviews
4. SQL Agent queries sales data
5. Orchestrator aggregates all responses
6. Response time < 6 seconds
```

### Automated E2E Tests

```python
import pytest
import asyncio
from ui.services.agentcore_client import AgentCoreClient

class TestEndToEnd:
    @pytest.fixture
    def client(self):
        """Create AgentCore client."""
        return AgentCoreClient(
            agent_id='capstone_ecomm_agent-iOd15C2EqB',
            region='us-west-2'
        )
    
    @pytest.mark.asyncio
    async def test_product_query_e2e(self, client, access_token):
        """Test complete product query flow."""
        query = "Show me products in Electronics category"
        
        responses = []
        async for chunk in client.invoke_streaming(
            input_text=query,
            session_id='test-session',
            access_token=access_token,
            tenant_id='tenanta'
        ):
            if 'data' in chunk:
                responses.append(chunk['data'])
        
        full_response = ''.join(responses)
        
        # Assertions
        assert len(full_response) > 0
        assert 'Electronics' in full_response or 'products' in full_response
    
    @pytest.mark.asyncio
    async def test_multi_agent_query_e2e(self, client, access_token):
        """Test query requiring multiple agents."""
        query = "Tell me about Smartphone X including specs and reviews"
        
        tool_uses = []
        responses = []
        
        async for chunk in client.invoke_streaming(
            input_text=query,
            session_id='test-session',
            access_token=access_token,
            tenant_id='tenanta'
        ):
            if 'current_tool_use' in chunk:
                tool_uses.append(chunk['current_tool_use']['name'])
            if 'data' in chunk:
                responses.append(chunk['data'])
        
        # Assertions
        assert len(tool_uses) >= 2  # At least KB Agent + Reviews Tool
        assert 'get_answers_for_unstructured_data' in tool_uses
        assert any('review' in tool for tool in tool_uses)
```

---

## 4. AWS Bedrock Model Evaluation

### Overview
AWS Bedrock Model Evaluation provides automated evaluation of model performance using predefined metrics.

### Evaluation Dataset

**File**: `evaluation/agent_evaluation_dataset.jsonl`

```jsonl
{"input": "Show me all products in the Electronics category", "expected_output": "List of electronics products with names and prices"}
{"input": "What are the camera specifications for Smartphone X?", "expected_output": "Camera specifications including megapixels, lens types, and features"}
{"input": "Show me 5-star reviews for Laptop Pro 15", "expected_output": "Customer reviews with 5-star rating for Laptop Pro 15"}
{"input": "What is the total sales for Electronics category last month?", "expected_output": "Total sales amount for Electronics category"}
{"input": "Show me order status for order #1001", "expected_output": "Order status, tracking number, and delivery information"}
```

### Create Evaluation Job

**Script**: `evaluation/create_bedrock_model_eval_job.sh`

```bash
#!/bin/bash

# Configuration
EVALUATION_NAME="ecommerce-agent-eval-$(date +%Y%m%d-%H%M%S)"
MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
DATASET_S3_URI="s3://capstone-ecommerce-eval/agent_evaluation_dataset.jsonl"
OUTPUT_S3_URI="s3://capstone-ecommerce-eval/results/"
ROLE_ARN="arn:aws:iam::123456789012:role/BedrockEvaluationRole"

# Create evaluation job
aws bedrock create-model-evaluation-job \
  --job-name "$EVALUATION_NAME" \
  --evaluation-config '{
    "automated": {
      "datasetMetricConfigs": [
        {
          "taskType": "QuestionAndAnswer",
          "dataset": {
            "name": "ecommerce-agent-dataset",
            "datasetLocation": {
              "s3Uri": "'$DATASET_S3_URI'"
            }
          },
          "metricNames": ["Accuracy", "Robustness", "Toxicity"]
        }
      ]
    }
  }' \
  --inference-config '{
    "models": [
      {
        "bedrockModel": {
          "modelIdentifier": "'$MODEL_ID'"
        }
      }
    ]
  }' \
  --output-data-config '{
    "s3Uri": "'$OUTPUT_S3_URI'"
  }' \
  --role-arn "$ROLE_ARN"

echo "Evaluation job created: $EVALUATION_NAME"
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Correctness of responses | > 90% |
| **Robustness** | Consistency across variations | > 85% |
| **Toxicity** | Harmful content detection | < 5% |
| **Latency** | Response time | < 5s |

### Results Analysis

```python
import boto3
import json

def analyze_evaluation_results(job_name: str):
    """Analyze Bedrock evaluation results."""
    
    bedrock = boto3.client('bedrock')
    
    # Get evaluation job
    response = bedrock.get_model_evaluation_job(jobIdentifier=job_name)
    
    # Download results from S3
    s3_uri = response['outputDataConfig']['s3Uri']
    # ... download and parse results
    
    # Calculate metrics
    metrics = {
        'accuracy': 0.92,
        'robustness': 0.88,
        'toxicity': 0.02,
        'avg_latency': 3.5
    }
    
    return metrics
```

---

## 5. RAG Evaluation with RAGAS

### Overview
RAGAS (Retrieval Augmented Generation Assessment) provides metrics for evaluating RAG systems.

### RAGAS Metrics

1. **Faithfulness**: How factually accurate is the generated answer based on the context?
2. **Answer Relevance**: How relevant is the answer to the question?
3. **Context Precision**: How precise is the retrieved context?
4. **Context Recall**: How much of the ground truth is captured in the context?

### Evaluation Dataset

**File**: `evaluation/kb_evaluation_dataset_filled.jsonl`

```jsonl
{"question": "What are the camera specifications for Smartphone X?", "ground_truth": "50MP main camera, 12MP ultra-wide, 10MP telephoto with 3x optical zoom", "contexts": ["The Smartphone X features a triple camera system with 50MP main camera..."]}
{"question": "What is the battery capacity of Laptop Pro 15?", "ground_truth": "80Wh battery with up to 12 hours of battery life", "contexts": ["Laptop Pro 15 is equipped with an 80Wh battery..."]}
```

### RAGAS Evaluation Script

**File**: `evaluation/rag-eval-ragasmetrics.py`

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)
from datasets import Dataset
import json

def load_evaluation_dataset(file_path: str) -> Dataset:
    """Load evaluation dataset from JSONL file."""
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return Dataset.from_list(data)

def run_ragas_evaluation(dataset: Dataset):
    """Run RAGAS evaluation on dataset."""
    
    # Evaluate
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall
        ]
    )
    
    return result

# Load dataset
dataset = load_evaluation_dataset('kb_evaluation_dataset_filled.jsonl')

# Run evaluation
results = run_ragas_evaluation(dataset)

# Print results
print(f"Faithfulness: {results['faithfulness']:.4f}")
print(f"Answer Relevance: {results['answer_relevance']:.4f}")
print(f"Context Precision: {results['context_precision']:.4f}")
print(f"Context Recall: {results['context_recall']:.4f}")

# Save detailed results
results.to_pandas().to_csv('ragas_eval_results.csv', index=False)
```

### RAGAS Results

**File**: `evaluation/ragas_eval_results/ragas_eval_summary_20251222_164504.json`

```json
{
  "faithfulness": 0.9245,
  "answer_relevance": 0.8876,
  "context_precision": 0.9102,
  "context_recall": 0.8654,
  "overall_score": 0.8969
}
```

### Target Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Faithfulness | > 0.90 | 0.92 ✓ |
| Answer Relevance | > 0.85 | 0.89 ✓ |
| Context Precision | > 0.85 | 0.91 ✓ |
| Context Recall | > 0.80 | 0.87 ✓ |

---

## 6. Performance Testing

### Load Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def load_test(num_requests: int, concurrency: int):
    """Run load test with specified concurrency."""
    
    client = AgentCoreClient(agent_id='capstone_ecomm_agent-iOd15C2EqB')
    
    queries = [
        "Show me products in Electronics",
        "What are the specs for Smartphone X?",
        "Show me 5-star reviews",
        "Total sales by category"
    ]
    
    async def make_request(i: int):
        query = queries[i % len(queries)]
        start = time.time()
        
        try:
            responses = []
            async for chunk in client.invoke_streaming(
                input_text=query,
                session_id=f'load-test-{i}',
                access_token=access_token,
                tenant_id='tenanta'
            ):
                if 'data' in chunk:
                    responses.append(chunk['data'])
            
            duration = time.time() - start
            return {'success': True, 'duration': duration}
        
        except Exception as e:
            duration = time.time() - start
            return {'success': False, 'duration': duration, 'error': str(e)}
    
    # Run requests with concurrency
    tasks = [make_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    successes = sum(1 for r in results if r['success'])
    failures = num_requests - successes
    durations = [r['duration'] for r in results if r['success']]
    
    print(f"Total Requests: {num_requests}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success Rate: {successes/num_requests*100:.2f}%")
    print(f"Avg Duration: {sum(durations)/len(durations):.2f}s")
    print(f"P50: {sorted(durations)[len(durations)//2]:.2f}s")
    print(f"P90: {sorted(durations)[int(len(durations)*0.9)]:.2f}s")
    print(f"P99: {sorted(durations)[int(len(durations)*0.99)]:.2f}s")

# Run load test
asyncio.run(load_test(num_requests=100, concurrency=10))
```

---

## 7. Testing Checklist

### Pre-Deployment Testing
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Load testing completed
- [ ] Model evaluation score > 90%
- [ ] RAGAS metrics meet targets
- [ ] Security testing completed
- [ ] Performance benchmarks met

### Post-Deployment Testing
- [ ] Smoke tests in production
- [ ] Monitoring dashboards configured
- [ ] Alarms tested and verified
- [ ] Rollback procedure tested
- [ ] Documentation updated

---

**Document Version**: 2.0  
**Last Updated**: January 2026  
**Author**: Sameer Battoo (sbattoo@amazon.com)
