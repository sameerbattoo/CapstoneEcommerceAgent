"""AWS Strands SQL Agent for RDS query validation and execution."""

import json
import re
import os
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2 import pool
import boto3
import logging
from datetime import datetime

# Import Strands SDK
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.types.content import SystemContentBlock

# Import Code Interpreter tool
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

# Define constants
MAX_ROWS_FOR_SQL_RESULT_DISPLAY = 20
MAX_SAMPLE_QUESTIONS = 5


class ValkeyCache:
    """Manages semantic caching using Valkey with vector search."""
    
    def __init__(
        self,
        logger: logging.Logger,
        aws_region: str,
        valkey_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Valkey cache manager.
        
        Args:
            logger: Logger instance
            aws_region: AWS region for Bedrock embedding model
            valkey_config: Configuration dictionary with:
                - endpoint: Valkey endpoint
                - port: Valkey port (default: 6379)
                - password: Auth token (optional)
                - use_tls: Use TLS encryption (default: True)
                - cache_ttl_seconds: TTL for cache entries (default: 3600)
                - similarity_threshold: Similarity threshold for cache hits (default: 0.93)
                - embed_model: Bedrock embedding model ID
        """
        self.logger = logger
        self.aws_region = aws_region
        self.valkey_config = valkey_config or {}
        self.cache_enabled = False
        self.redis_client = None
        self.bedrock_runtime = None
        
        # Cache configuration
        self.cache_ttl = self.valkey_config.get('cache_ttl_seconds', 3600)
        self.similarity_threshold = self.valkey_config.get('similarity_threshold', 0.90)
        self.similarity_threshold_min = self.valkey_config.get('similarity_threshold_min', 0.75)
        self.embed_model = self.valkey_config.get('embed_model', 'amazon.titan-embed-text-v2:0')
        self.embed_dimensions = 1024 if 'v2' in self.embed_model else 1536
        
        # Initialize connections if config provided
        if self.valkey_config.get('endpoint'):
            self._initialize_connections()
    
    def _initialize_connections(self) -> None:
        """Initialize Valkey and Bedrock connections."""
        try:
            import redis
            
            # Initialize Valkey client
            self.logger.info(f"Connecting to Valkey at {self.valkey_config['endpoint']}:{self.valkey_config.get('port', 6379)}")
            
            redis_kwargs = {
                'host': self.valkey_config['endpoint'],
                'port': self.valkey_config.get('port', 6379),
                'decode_responses': False,  # Keep as bytes for vector operations
                'socket_connect_timeout': 5,
                'socket_timeout': 5
            }
            
            if self.valkey_config.get('use_tls', True):
                redis_kwargs['ssl'] = True
                redis_kwargs['ssl_cert_reqs'] = None
            
            if self.valkey_config.get('password'):
                redis_kwargs['password'] = self.valkey_config['password']
            
            self.redis_client = redis.Redis(**redis_kwargs)
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Successfully connected to Valkey")
            
            # Initialize Bedrock runtime client
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.aws_region)
            self.logger.info(f"Initialized Bedrock runtime client with model: {self.embed_model}")
            
            self.cache_enabled = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Valkey cache: {str(e)}")
            self.logger.warning("Semantic caching will be disabled")
            self.cache_enabled = False
    
    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for comparison.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL string
        """
        if not sql:
            return ""
        
        import re
        
        # Convert to lowercase
        normalized = sql.lower()
        
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        # Remove trailing semicolon
        normalized = normalized.rstrip(';')
        
        # Remove comments (-- style)
        normalized = re.sub(r'--[^\n]*', '', normalized)
        
        # Remove comments (/* */ style)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        return normalized
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for text using Bedrock Titan Embed.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not self.bedrock_runtime:
            return None
        
        try:
            # Prepare request body for Titan Embed with L2 normalization
            request_body = json.dumps({
                "inputText": text,
                "normalize": True
            })
            
            # Call Bedrock
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embed_model,
                body=request_body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                self.logger.error("No embedding returned from Bedrock")
                return None
            
            self.logger.info(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            return None
    
    def check_cache(self, user_query: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar query exists in cache.
        
        Args:
            user_query: User's natural language question
            tenant_id: Tenant ID for isolation
            
        Returns:
            Cached results dictionary if cache hit, None if cache miss
        """
        if not self.cache_enabled:
            return None
        
        try:
            import time
            import numpy as np
            
            self.logger.info(f"Checking cache for query: {user_query[:500]}...")
            
            # Generate embedding for query
            query_embedding = self._generate_embedding(user_query)
            if not query_embedding:
                self.logger.warning("Failed to generate embedding, skipping cache check")
                return None
            
            # Convert embedding to bytes for Valkey
            embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Calculate timestamp threshold (only fresh entries)
            current_time = int(time.time())
            min_timestamp = current_time - self.cache_ttl
            
            # Build vector search query with proper KNN syntax
            # Wrap filter conditions in parentheses before => operator
            query = f"(@tenant_id:{{{tenant_id}}} @timestamp:[{min_timestamp} {current_time}])=>[KNN 1 @question_embedding $vec AS score]"
            
            # Execute search - don't use RETURN to get accurate score
            result = self.redis_client.execute_command(
                'FT.SEARCH', 'sql_cache_index',
                query,
                'PARAMS', '2', 'vec', embedding_bytes,
                'DIALECT', '2'
            )
            
            # Parse results
            if not result or result[0] == 0:
                self.logger.info("Cache miss - no similar queries found")
                return None
            
            # Format: [count, key, [field1, value1, field2, value2, ...]]
            cache_key = result[1]
            doc = result[2] if len(result) > 2 else []
            
            # Convert doc to dict
            doc_dict = {}
            for i in range(0, len(doc), 2):
                field = doc[i]
                value = doc[i + 1] if i + 1 < len(doc) else None
                field_str = field.decode('utf-8') if isinstance(field, bytes) else str(field)
                doc_dict[field_str] = value
            
            # Get vector distance score
            score_value = doc_dict.get('score')
            if score_value is None:
                self.logger.warning(f"No score field found in result. Available fields: {list(doc_dict.keys())}")
                return None
            
            # Convert score to float and calculate similarity
            if isinstance(score_value, bytes):
                vector_distance = float(score_value.decode('utf-8'))
            else:
                vector_distance = float(score_value)
            
            similarity_score = 1 - vector_distance
            self.logger.info(f"Found similar query with similarity score: {similarity_score:.4f} (Tier 1: ≥{self.similarity_threshold}, Tier 2: ≥{self.similarity_threshold_min})")
            
            # Tiered validation logic
            tier = 0
            cache_hit = False
            validation_reason = ""
            
            if similarity_score >= self.similarity_threshold:
                # Tier 1: High confidence - direct cache hit (no SQL check needed)
                tier = 1
                cache_hit = True
                validation_reason = f"Tier 1: High similarity ({similarity_score:.4f} ≥ {self.similarity_threshold})"
                self.logger.info(validation_reason)
                
            elif similarity_score >= self.similarity_threshold_min:
                # Tier 2: Medium confidence - SQL validation required
                tier = 2
                self.logger.info(f"Tier 2: Medium similarity ({similarity_score:.4f}), checking SQL match...")
                
                # Extract cached data to get SQL for comparison
                json_doc_bytes = doc_dict.get('$')
                if not json_doc_bytes:
                    self.logger.error("No JSON document found in cache result")
                    return None
                
                json_doc_str = json_doc_bytes.decode('utf-8') if isinstance(json_doc_bytes, bytes) else str(json_doc_bytes)
                cached_doc = json.loads(json_doc_str)
                cached_sql = cached_doc.get('sql_query', '')
                
                # For Tier 2, we need to generate SQL from the current query to compare
                # Since we don't have the generated SQL yet at this point in the flow,
                # we'll return a special flag indicating SQL validation is needed
                # The calling code will need to generate SQL and call back if needed
                
                # For now, we'll be conservative and treat Tier 2 as cache miss
                # The proper implementation would require refactoring the flow
                cache_hit = False
                validation_reason = f"Tier 2: Medium similarity ({similarity_score:.4f}) - SQL validation required but not implemented in check_cache"
                self.logger.info(f"{validation_reason}, treating as cache miss")
                return None
                
            else:
                # Tier 3: Low confidence - automatic cache miss
                tier = 3
                cache_hit = False
                validation_reason = f"Tier 3: Low similarity ({similarity_score:.4f} < {self.similarity_threshold_min})"
                self.logger.info(f"{validation_reason}, treating as cache miss")
                return None
            
            # If we reach here, it's a Tier 1 cache hit
            # Extract cached data from the JSON document (stored in '$' field)
            json_doc_bytes = doc_dict.get('$')
            if not json_doc_bytes:
                self.logger.error("No JSON document found in cache result")
                return None
            
            # Parse the JSON document
            json_doc_str = json_doc_bytes.decode('utf-8') if isinstance(json_doc_bytes, bytes) else str(json_doc_bytes)
            cached_doc = json.loads(json_doc_str)
            
            cached_sql = cached_doc.get('sql_query')
            cached_results_json = cached_doc.get('results')
            cached_chart_url = cached_doc.get('chart_url')
            cached_row_count = cached_doc.get('row_count')
            
            # Parse results if it's a JSON string
            if isinstance(cached_results_json, str):
                cached_results = json.loads(cached_results_json)
            else:
                cached_results = cached_results_json or []
            
            self.logger.info(f"Cache HIT! Returning cached results (similarity: {similarity_score:.4f})")
            
            return {
                'sql': cached_sql,
                'results': cached_results,
                'chart_url': cached_chart_url,
                'row_count': int(cached_row_count) if cached_row_count else 0,
                'cache_hit': True,
                'tier': 1,
                'similarity_score': similarity_score
            }
            
        except Exception as e:
            self.logger.error(f"Error checking cache: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def store_in_cache(
        self,
        user_query: str,
        tenant_id: str,
        sql: str,
        results: List[Dict[str, Any]],
        chart_url: Optional[str],
        row_count: int
    ) -> None:
        """
        Store query results in cache.
        
        Args:
            user_query: User's natural language question
            tenant_id: Tenant ID for isolation
            sql: Generated SQL query
            results: Query execution results
            chart_url: Chart URL (if generated)
            row_count: Number of rows returned
        """
        if not self.cache_enabled:
            return
        
        try:
            import time
            import hashlib
            import numpy as np
            
            self.logger.info(f"Storing query in cache: {user_query[:100]}...")
            
            # Generate embedding for query
            query_embedding = self._generate_embedding(user_query)
            if not query_embedding:
                self.logger.warning("Failed to generate embedding, skipping cache storage")
                return
            
            # Create cache key
            query_hash = hashlib.sha256(f"{tenant_id}:{user_query}".encode()).hexdigest()[:16]
            cache_key = f"sql_cache:{tenant_id}:{query_hash}"
            
            # Limit results to avoid huge cache entries (store first 200 rows)
            limited_results = results[:200] if len(results) > 200 else results
            
            # Build cache document
            cache_doc = {
                'tenant_id': tenant_id,
                'question_text': user_query,
                'question_embedding': query_embedding,
                'sql_query': sql,
                'results': json.dumps(limited_results, default=str),
                'chart_url': chart_url or '',
                'row_count': row_count,
                'timestamp': int(time.time())
            }
            
            # Store in Valkey using JSON.SET
            self.redis_client.execute_command(
                'JSON.SET', cache_key, '$', json.dumps(cache_doc)
            )
            
            # Set TTL
            self.redis_client.expire(cache_key, self.cache_ttl)
            
            self.logger.info(f"Successfully stored query in cache with key: {cache_key}, TTL: {self.cache_ttl}s")
            
        except Exception as e:
            self.logger.error(f"Error storing in cache: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def check_cache_with_sql(
        self,
        user_query: str,
        tenant_id: str,
        generated_sql: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check cache with Tier 2 SQL validation for medium similarity matches.
        
        This method is called after SQL generation when initial cache check
        returned None but there might be Tier 2 matches that need SQL validation.
        
        Args:
            user_query: User's natural language question
            tenant_id: Tenant ID for isolation
            generated_sql: The SQL query generated for this question
            
        Returns:
            Cached results dictionary if Tier 2 cache hit, None if cache miss
        """
        if not self.cache_enabled:
            return None
        
        try:
            import time
            import numpy as np
            
            self.logger.info(f"Checking Tier 2 cache with SQL validation...")
            
            # Generate embedding for query
            query_embedding = self._generate_embedding(user_query)
            if not query_embedding:
                return None
            
            # Convert embedding to bytes for Valkey
            embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Calculate timestamp threshold
            current_time = int(time.time())
            min_timestamp = current_time - self.cache_ttl
            
            # Build vector search query
            query = f"(@tenant_id:{{{tenant_id}}} @timestamp:[{min_timestamp} {current_time}])=>[KNN 1 @question_embedding $vec AS score]"
            
            # Execute search
            result = self.redis_client.execute_command(
                'FT.SEARCH', 'sql_cache_index',
                query,
                'PARAMS', '2', 'vec', embedding_bytes,
                'DIALECT', '2'
            )
            
            # Parse results
            if not result or result[0] == 0:
                return None
            
            # Extract document
            doc = result[2] if len(result) > 2 else []
            doc_dict = {}
            for i in range(0, len(doc), 2):
                field = doc[i]
                value = doc[i + 1] if i + 1 < len(doc) else None
                field_str = field.decode('utf-8') if isinstance(field, bytes) else str(field)
                doc_dict[field_str] = value
            
            # Get similarity score
            score_value = doc_dict.get('score')
            if score_value is None:
                return None
            
            vector_distance = float(score_value.decode('utf-8') if isinstance(score_value, bytes) else score_value)
            similarity_score = 1 - vector_distance
            
            # Check if in Tier 2 range
            if similarity_score < self.similarity_threshold_min or similarity_score >= self.similarity_threshold:
                # Not in Tier 2 range
                return None
            
            self.logger.info(f"Tier 2 candidate found with similarity {similarity_score:.4f}, validating SQL...")
            
            # Extract cached SQL
            json_doc_bytes = doc_dict.get('$')
            if not json_doc_bytes:
                return None
            
            json_doc_str = json_doc_bytes.decode('utf-8') if isinstance(json_doc_bytes, bytes) else str(json_doc_bytes)
            cached_doc = json.loads(json_doc_str)
            cached_sql = cached_doc.get('sql_query', '')
            
            # Normalize and compare SQL
            normalized_cached = self._normalize_sql(cached_sql)
            normalized_generated = self._normalize_sql(generated_sql)
            
            if normalized_cached == normalized_generated:
                # SQL matches - Tier 2 cache hit!
                self.logger.info(f"Tier 2 CACHE HIT! Similarity {similarity_score:.4f} + SQL match")
                
                # Extract all cached data
                cached_results_json = cached_doc.get('results')
                cached_chart_url = cached_doc.get('chart_url')
                cached_row_count = cached_doc.get('row_count')
                
                # Parse results
                if isinstance(cached_results_json, str):
                    cached_results = json.loads(cached_results_json)
                else:
                    cached_results = cached_results_json or []
                
                return {
                    'sql': cached_sql,
                    'results': cached_results,
                    'chart_url': cached_chart_url,
                    'row_count': int(cached_row_count) if cached_row_count else 0,
                    'cache_hit': True,
                    'tier': 2,
                    'similarity_score': similarity_score
                }
            else:
                self.logger.info(f"Tier 2: SQL different - cache miss")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in Tier 2 cache check: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

def log_info(logger: logging.Logger, function: str, content: str, log_complete_content: bool = False) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not log_complete_content:
        logger.info(f"[{timestamp}] {function}: {content}..." if len(content) > 500 else f"[{timestamp}] {function}: {content}")
    else:
        logger.info(f"[{timestamp}] {function}: {content}...")

class SchemaExtractor:
    """Extracts CREATE TABLE statements from RDS database."""
    
    def __init__(self, db_pool: pool.ThreadedConnectionPool, tenant_id: str, logger: logging.Logger):
        self.db_pool = db_pool
        self.tenant_id = tenant_id.lower()
        self.logger = logger
        self.create_statements = {}
    
    def extract_schema(self) -> Dict[str, str]:

        """Extract CREATE TABLE statements from database."""
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            # Get all tables
            sql = """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = %s
                ORDER BY tablename
            """
            cursor.execute(sql, (self.tenant_id,))
            tables = [row[0] for row in cursor.fetchall()]
            
            # Log all table names
            self.logger.info(f"[SchemaExtractor] Found {len(tables)} tables in schema '{self.tenant_id}':")
            for table in tables:
                self.logger.info(f"  - {table}")
            
            # Generate CREATE TABLE statement for each table
            for table in tables:
                create_stmt = self._generate_create_statement(cursor, table)
                self.create_statements[table] = create_stmt
            
            cursor.close()
            
            return self.create_statements
        
        except Exception as e:
            error_msg = f"Failed to extract schema: {str(e)}"
            self.logger.error(f"[SchemaExtractor] {error_msg}")
            raise Exception(error_msg)
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _generate_create_statement(self, cursor, table_name: str) -> str:
        """Generate CREATE TABLE statement for a table."""

        try:
            # Get columns
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (self.tenant_id, table_name,))
            
            columns = cursor.fetchall()
            
            # Log all columns
            self.logger.info(f"[SchemaExtractor] Table '{table_name}' has {len(columns)} columns:")
            for col_name, data_type, max_length, is_nullable, default in columns:
                self.logger.info(f"  - {col_name} ({data_type}{'(' + str(max_length) + ')' if max_length else ''}, nullable={is_nullable})")
            
            # Get primary key
            full_name = f"{self.tenant_id}.{table_name}"  # -> myschema.mytable
            cursor.execute("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass AND i.indisprimary
            """, (full_name,))
            
            pk_columns = [row[0] for row in cursor.fetchall()]
            
            # Log primary key columns
            if pk_columns:
                self.logger.info(f"[SchemaExtractor] Table '{table_name}' primary key: {', '.join(pk_columns)}")
            else:
                self.logger.info(f"[SchemaExtractor] Table '{table_name}' has no primary key")
            
            # Get foreign keys
            cursor.execute("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.constraint_schema = kcu.constraint_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.constraint_schema = %s AND tc.table_name = %s
            """, (self.tenant_id, table_name,))
            
            fk_info = cursor.fetchall()
            
            # Log foreign keys
            if fk_info:
                self.logger.info(f"[SchemaExtractor] Table '{table_name}' has {len(fk_info)} foreign key(s):")
                for fk_col, fk_table, fk_ref_col in fk_info:
                    self.logger.info(f"  - {fk_col} -> {fk_table}({fk_ref_col})")
            else:
                self.logger.info(f"[SchemaExtractor] Table '{table_name}' has no foreign keys")
            
            # Build CREATE TABLE statement
            create_stmt = f"CREATE TABLE {self.tenant_id}.{table_name} (\n"
            
            col_defs = []
            for col_name, data_type, max_length, is_nullable, default in columns:
                col_def = f"    {col_name} "
                
                # Data type
                if max_length and data_type in ('character varying', 'character'):
                    col_def += f"{data_type}({max_length})"
                else:
                    col_def += data_type
                
                # Nullable
                if is_nullable == 'NO':
                    col_def += " NOT NULL"
                
                # Default
                if default:
                    col_def += f" DEFAULT {default}"
                
                col_defs.append(col_def)
            
            create_stmt += ",\n".join(col_defs)
            
            # Add primary key
            if pk_columns:
                create_stmt += f",\n    PRIMARY KEY ({', '.join(pk_columns)})"
            
            # Add foreign keys
            for fk_col, fk_table, fk_ref_col in fk_info:
                create_stmt += f",\n    FOREIGN KEY ({fk_col}) REFERENCES {self.tenant_id}.{fk_table}({fk_ref_col})"
            
            create_stmt += "\n);"
            
            # Log the final CREATE statement
            self.logger.info(f"[SchemaExtractor] Generated CREATE statement for '{table_name}':")
            self.logger.info(create_stmt)
            self.logger.info("")  # Empty line for readability
            
            return create_stmt

        except Exception as e:
            error_msg = f"Failed to create table: {table_name}. Error Info:{str(e)}"
            self.logger.error(f"[SchemaExtractor] {error_msg}")
            raise Exception(error_msg)

class SQLExecutorTool:
    """Executes SQL queries against RDS database."""
    
    def __init__(self, db_pool: pool.ThreadedConnectionPool, logger: logging.Logger):
        self.db_pool = db_pool
        self.logger = logger
        self.name = "execute_sql"
        self.description = "Executes SQL query against the RDS database and returns results"
    
    def execute(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query safely."""

        # Basic SQL injection prevention
        if any(keyword in sql.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']):
            error_msg = "Only SELECT queries are allowed to be executed"
            self.logger.error(f"[SQLExecutorTool] {error_msg}. SQL: {sql}")
            raise Exception(error_msg)
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute(sql)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
            else:
                results = []
            
            cursor.close()
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results)
            }
        except Exception as e:
            error_msg = f"Failed to execute SQL: {sql}. Error Info: {str(e)}"
            self.logger.error(f"[SQLExecutorTool] {error_msg}")
            raise Exception(error_msg)
        
        finally:
            if conn:
                self.db_pool.putconn(conn)


class ChartGeneratorTool:
    """Executes Python visualization code using AWS Code Interpreter."""
    
    def __init__(self, logger: logging.Logger, aws_region: str, chart_s3_bucket: str, cloudfront_domain: Optional[str] = None):
        self.logger = logger
        self.name = "execute_chart_code"
        self.description = "Executes Python code to generate visual charts and graphs"
        self.code_interpreter = CodeInterpreter(region=aws_region)
        self.aws_region = aws_region
        self.chart_s3_bucket = chart_s3_bucket
        self.cloudfront_domain = cloudfront_domain  # e.g., "d1234567890abc.cloudfront.net"
        self.s3_client = boto3.client('s3', region_name=aws_region)

    def _call_tool(self, tool_name, args):
        response = self.code_interpreter.invoke(tool_name, args)
        for event in response["stream"]:
            # First event that has a result; adjust if you want the full stream
            return event["result"]
    
    def _upload_to_s3(self, chart_base64: str) -> Optional[str]:
        """Upload base64-encoded chart to S3 and return CloudFront or pre-signed URL.
        
        Args:
            chart_base64: Base64-encoded image string
            
        Returns:
            CloudFront URL (if configured) or pre-signed S3 URL
        """
        try:
            import base64
            import uuid
            from datetime import datetime
            
            # Use bucket name from constructor
            bucket_name = self.chart_s3_bucket
            if not bucket_name:
                self.logger.error("[ChartGeneratorTool] chart_s3_bucket not provided")
                return None
            
            # Decode base64 to binary
            chart_bytes = base64.b64decode(chart_base64)
            self.logger.info(f"[ChartGeneratorTool] Decoded {len(chart_bytes)} bytes from base64")
            
            # Generate unique S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            s3_key = f"charts/{timestamp}_{unique_id}.png"
            
            # Upload to S3
            self.logger.info(f"[ChartGeneratorTool] Uploading to s3://{bucket_name}/{s3_key}")
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=chart_bytes,
                ContentType='image/png',
                CacheControl='max-age=31536000'  # Cache for 1 year
            )
            
            # Generate URL based on CloudFront availability
            if self.cloudfront_domain:
                # Use CloudFront URL (no expiration, better performance)
                chart_url = f"https://{self.cloudfront_domain}/{s3_key}"
                self.logger.info(f"[ChartGeneratorTool] Generated CloudFront URL: {chart_url}")
            else:
                # Fallback to pre-signed S3 URL (expires in 7 days)
                chart_url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': s3_key
                    },
                    ExpiresIn=604800  # 7 days (maximum for pre-signed URLs)
                )
                self.logger.info(f"[ChartGeneratorTool] Generated pre-signed URL: {chart_url[:100]}...")
            
            return chart_url
            
        except Exception as e:
            self.logger.error(f"[ChartGeneratorTool] Failed to upload to S3: {str(e)}")
            import traceback
            self.logger.error(f"[ChartGeneratorTool] Traceback: {traceback.format_exc()}")
            return None  

    def generate(self, code: str) -> Dict[str, Any]:
        """Generate chart from query results using direct Code Interpreter call."""
        
        try:
            self.logger.info(f"[ChartGeneratorTool] Executing code to generate chart via Code Interpreter")
            
            # Start Code Interpreter session
            self.code_interpreter.start(session_timeout_seconds=1200)
            
            # Invoke Code Interpreter
            self.logger.info(f"[ChartGeneratorTool] Calling invoke() method")
            response = self.code_interpreter.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": False
            })

            if response is None:
                raise Exception("Code Interpreter returned None")

            # Process the streaming response to extract files and output
            result = self._process_streaming_response(response)
                        
            # Extract values from processed result
            chart_data = result.get('chart_data')
            output_text = result.get('output', '')
            
            # Convert chart_data to base64 for easy transport
            chart_base64 = None
            chart_url = None
            if chart_data:
                import base64
                if isinstance(chart_data, bytes):
                    chart_base64 = base64.b64encode(chart_data).decode('utf-8')
                    self.logger.info(f"[ChartGeneratorTool] Converted {len(chart_data)} bytes to base64")
                else:
                    chart_base64 = base64.b64encode(str(chart_data).encode()).decode('utf-8')
                
                # Upload to S3 and get pre-signed URL
                chart_url = self._upload_to_s3(chart_base64)
                if chart_url:
                    self.logger.info(f"[ChartGeneratorTool] Chart uploaded to S3 successfully")
                else:
                    self.logger.warning(f"[ChartGeneratorTool] Failed to upload chart to S3, will use base64")
            
            # Stop Code Interpreter session
            self.code_interpreter.stop()
            
            return {
                "success": True,
                "message": "Chart generated successfully" if chart_data else "Chart code executed but no file retrieved",
                "chart_url": chart_url,  # Pre-signed S3 URL (preferred)
                "output": output_text,
            }
            
        except Exception as e:
            error_msg = f"Failed to generate chart: {str(e)}"
            self.logger.error(f"[ChartGeneratorTool] {error_msg}")
            import traceback
            self.logger.error(f"[ChartGeneratorTool] Traceback: {traceback.format_exc()}")
            
            # Ensure session is stopped
            try:
                self.code_interpreter.stop()
            except:
                pass
            
            return {
                "success": False,
                "message": error_msg,
                "chart_url": None,
                "output": ''
            }
    
    def _process_streaming_response(self, response) -> Dict[str, Any]:
        """Process AWS EventStream response from Code Interpreter to extract chart data from stdout."""
        
        chart_data = None
        stdout_text = ""
        
        try:
            # Extract stdout from streaming response
            if isinstance(response, dict) and ('EventStream' in response or 'stream' in response):
                stream_key = 'EventStream' if 'EventStream' in response else 'stream'
                event_stream = response[stream_key]
                
                for event in event_stream:
                    if isinstance(event, dict) and 'result' in event:
                        result_data = event['result']
                        if isinstance(result_data, dict) and 'structuredContent' in result_data:
                            structured = result_data['structuredContent']
                            if isinstance(structured, dict) and 'stdout' in structured:
                                stdout_text = structured['stdout']
                                break
            
            # Extract base64 chart data from stdout using markers
            import re
            import base64
            
            chart_match = re.search(r'__CHART_DATA_START__\s*\n(.*?)\n\s*__CHART_DATA_END__', stdout_text, re.DOTALL)
            if chart_match:
                chart_base64 = chart_match.group(1).strip()
                
                try:
                    chart_data = base64.b64decode(chart_base64)
                    self.logger.info(f"[ChartGeneratorTool] Successfully decoded chart: {len(chart_data)} bytes")
                except Exception as e:
                    self.logger.error(f"[ChartGeneratorTool] Failed to decode base64: {e}")
            else:
                self.logger.warning(f"[ChartGeneratorTool] No chart data markers found in stdout")
            
            return {
                'chart_data': chart_data,
                'output': stdout_text
            }
            
        except Exception as e:
            self.logger.error(f"[ChartGeneratorTool] Error processing streaming response: {e}")
            import traceback
            self.logger.error(f"[ChartGeneratorTool] Traceback: {traceback.format_exc()}")
            return {
                'chart_data': None,
                'output': ''
            }


class SQLAgent:
    """Main SQL Agent using AWS Strands with Bedrock."""
    
    def __init__(
        self, 
        logger: logging.Logger, 
        db_pool: pool.ThreadedConnectionPool, 
        aws_region: str, 
        model_id: str, 
        tenant_id: str,
        chart_s3_bucket: str,
        cached_schema: Optional[Dict[str, str]] = None,
        cloudfront_domain: Optional[str] = None,
        valkey_config: Optional[Dict[str, Any]] = None
    ):

        # Get AWS region & other config value from environment or parameter
        self.db_pool = db_pool
        self.model_id = model_id
        self.region = aws_region
        self.logger = logger
        self.tenant_id = tenant_id
        self.chart_s3_bucket = chart_s3_bucket
        log_info(logger, "SQLAgent.Init", f"Starting function, Region:{aws_region}, Model ID:{model_id}, Tenant ID:{tenant_id}, Chart S3 Bucket:{chart_s3_bucket}")

        # Initialize Valkey cache
        self.cache = ValkeyCache(logger, aws_region, valkey_config)
        if self.cache.cache_enabled:
            log_info(logger, "SQLAgent.Init", "Semantic caching enabled with Valkey")
        else:
            log_info(logger, "SQLAgent.Init", "Semantic caching disabled")

        # Step 1: Use cached schema or extract from database
        if cached_schema:
            log_info(logger, "SQLAgent.Init", f"Step 1: Using cached schema with {len(cached_schema)} tables")
            self.create_statements = cached_schema
        else:
            log_info(logger, "SQLAgent.Init", f"Step 1: Extracting schema from database...")
            schema_extractor = SchemaExtractor(db_pool, tenant_id, logger)
            self.create_statements = schema_extractor.extract_schema()
            log_info(logger, "SQLAgent.Init", f"Step 1: Extracted schema for {len(self.create_statements)} tables")

        # Step 2: Initialize SQL executor tool
        self.executor_tool = SQLExecutorTool(db_pool, logger)
        log_info(logger, "SQLAgent.Init", f"Step 2: SQLExecutorTool configured")
        
        # Step 2.5: Initialize Chart Generator tool
        self.chart_tool = ChartGeneratorTool(
            logger, 
            aws_region, 
            chart_s3_bucket,
            cloudfront_domain=cloudfront_domain
        )
        log_info(logger, "SQLAgent.Init", f"Step 2.5: ChartGeneratorTool configured")
               
        # Step 3: Initialize Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            region_name=self.region
        )
        log_info(logger, "SQLAgent.Init", f"Step 3: Bedrock model initialized")
        
        # Step 4: Build system prompt with schema context
        system_prompt = self._build_system_prompt()
        log_info(logger, "SQLAgent.Init", f"Step 4: Built system prompt")
        
        # Step 5: Create Strands agent with Bedrock model and tools
        self.agent = Agent(
            name="SQL Query Agent",
            system_prompt=[ # Define system prompt with cache points
                SystemContentBlock(
                    text=system_prompt
                ),
                SystemContentBlock(cachePoint={"type": "default"})
            ],
            model=self.model,
        )
        log_info(logger, "SQLAgent.Init", f"Step 5: Created Strands agent with Bedrock model and Code Interpreter tool")
        log_info(logger, "SQLAgent.Init", f"Ending function")

    def _get_schema_summary(self) -> str:
        """Get a summary of available tables."""

        return "\n\n".join([
            f"-- Table: {table}\n{stmt}" 
            for table, stmt in self.create_statements.items()
        ])

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the SQL agent with database schema context."""
        
        # Prepare schema context from extracted CREATE TABLE statements
        schema_context = self._get_schema_summary()
        
        return f"""
            # SQL Agent

            <task_description>
            You are an elite SQL query specialist with deep expertise in PostgreSQL for AWS RDS environments. Your mission encompasses two primary capabilities:

            1. **SQL Generation**: Transform natural language data requests into highly optimized, production-ready SQL queries that leverage the provided database schema effectively.

            2. **Data Visualization Code Generation**: When query results contain numeric data suitable for visualization, generate Python code using matplotlib to create professional charts. 
            You will generate the visualization code, which will be executed separately in a secure Python environment.
            </task_description>

            <database_schema>
            {schema_context}
            </database_schema>

            <sql_generation_instructions>

            Follow this systematic, step-by-step approach when crafting SQL queries:

            **Step 1 - Request Analysis**
            - Thoroughly examine the user's natural language request to extract the core data requirements
            - Identify key entities, relationships, filters, and aggregations needed
            - Clarify any ambiguous terms or conditions before proceeding

            **Step 2 - Schema Validation & Feasibility Check**
            - Cross-reference the user's request against the provided database schema
            - Verify that all required tables, columns, and relationships exist in the schema
            - If the schema lacks necessary elements to fulfill the request, explicitly communicate this limitation rather than generating an incomplete or inaccurate query

            **Step 3 - Query Decomposition**
            - For complex requests, break down the problem into logical sub-components
            - Identify whether subqueries, CTEs (Common Table Expressions), or multiple steps are needed
            - Plan the query structure before writing SQL code

            **Step 4 - SQL Query Construction**
            Construct your SQL query with meticulous attention to detail:

            a) **SELECT Clause**: Specify exactly the columns requested, using appropriate aliases for clarity
            b) **FROM & JOIN Clauses**: 
            - Use proper table aliases for readability
            - Implement appropriate JOIN types (INNER, LEFT, RIGHT, FULL) based on the data relationships
            - Base JOINs on foreign key relationships defined in the schema
            c) **WHERE Clause**: 
            - Apply precise filtering conditions that match the user's requirements
            - Use correct data types and formats for comparison values
            - For order status filtering, use only these exact values: 'delivered', 'processing', or 'shipped'
            d) **GROUP BY & Aggregation**: 
            - Apply aggregation functions (COUNT, SUM, AVG, MAX, MIN) when summarization is needed
            - Ensure all non-aggregated columns in SELECT appear in GROUP BY
            e) **HAVING Clause**: Use for filtering aggregated results when necessary
            f) **ORDER BY Clause**: Sort results meaningfully based on the context (e.g., chronological, alphabetical, by magnitude)
            g) **LIMIT Clause**: Include reasonable limits to prevent overwhelming result sets, especially for exploratory queries

            **Step 5 - PostgreSQL Optimization**
            - Leverage PostgreSQL-specific features and AWS RDS best practices
            - Ensure queries are efficient and performant
            - Use appropriate indexing strategies in your query design
            - Add inline comments (-- or /* */) to explain complex logic, subqueries, or business rules

            **Step 6 - Query Validation**
            - Ensure all column references are fully qualified with table names or aliases when multiple tables are involved
            - Verify that the query syntax is valid PostgreSQL
            - Confirm the query addresses all aspects of the user's request

            </sql_generation_instructions>

            <chart_generation_instructions>
            ## When to Generate Visualization Code

            Generate Python visualization code when ALL of the following conditions are met:

            1. **Numeric Data Present**: Query results contain at least one numeric column 
            (counts, sums, averages, prices, quantities, etc.)

            2. **Multiple Rows**: Results contain more than one row of data

            3. **Suitable for Visualization**: Data represents:
            - Time-series trends (daily, monthly, quarterly, yearly)
            - Categorical comparisons (by status, region, product, etc.)
            - Multi-dimensional breakdowns (e.g., sales by region and quarter)
            - Distributions or aggregations

            4. **User Intent**: User query implies visualization need:
            - "Show trend...", "Compare...", "Visualize...", "Chart..."
            - Questions about patterns, changes over time, or comparisons
            - Queries with GROUP BY, aggregations, or time dimensions

            **Do NOT generate charts for:**
            - Single row results
            - Text-only data (names, descriptions, emails)
            - Simple lookups (e.g., "What is customer X's email?")
            - Data unsuitable for visualization

            ## Dataset Size Guidance
            - Aggregated queries: Execute immediately (typically < 1000 rows)
            - Raw data queries with LIMIT: Execute immediately
            - Only refuse execution if query would scan millions of rows without aggregation
            - Do NOT ask users to narrow scope unless, Query returns 100,000+ raw rows

            ## Python Code Generation Requirements

            When generating visualization code, follow these strict requirements:
            ### 1. Code Structure

            ```python
            import pandas as pd
            import matplotlib.pyplot as plt
            import json
            import base64
            from datetime import datetime

            # Load data from JSON
            data_json = '''[JSON_DATA_HERE]'''
            data = json.loads(data_json)
            df = pd.DataFrame(data)

            # [YOUR VISUALIZATION CODE HERE]

            # Save chart
            output_path = 'chart.png'
            plt.savefig(output_path, format='png', dpi=100, quality=85, bbox_inches='tight')

            # Read and output as base64
            with open(output_path, 'rb') as f:
                chart_bytes = f.read()
                chart_base64 = base64.b64encode(chart_bytes).decode('utf-8')
                print(f"\n__CHART_DATA_START__")
                print(chart_base64)
                print(f"__CHART_DATA_END__")

            ## Chart Type Selection
            Choose the most appropriate chart type based on data structure:
            1. Stacked Bar Chart - Use when:
            - Data has time/period dimension (year, quarter, month, day)
            - Data has category dimension (status, type, region)
            - Data has numeric values
            - Example: Orders per month by status

            2. Line Chart - Use when:
            - Time-series data with single metric
            - Showing trends over time
            - Example: Daily sales over time

            3. Bar Chart - Use when:
            - Categorical comparisons without time dimension
            - Comparing values across categories
            - Example: Sales by product category

            4. Multi-line Chart - Use when:
            - Multiple metrics over time
            - Comparing trends of different measures
            - Example: Revenue and profit over months

            ## Data Preparation
            1. For Stacked Bar Charts:
            - Combine year+quarter or year+month if separate columns
            if 'year' in df.columns and 'quarter' in df.columns:
                df['period'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
            elif 'year' in df.columns and 'month' in df.columns:
                df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

            - Pivot for stacking
            pivot_df = df.pivot_table(
                index='period',  # or date column
                columns='category_column',  # status, type, etc.
                values='numeric_column',
                aggfunc='sum',
                fill_value=0
            )
            - Create stacked bar chart
            pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8)

            2. For Time-Series:
            - Format date labels based on granularity
            if date_range <= 31 days:
                date_format = '%b %d'  # Feb 19
            elif date_range <= 90 days:
                date_format = '%m/%d'  # 02/19
            else:
                date_format = '%b %Y'  # Feb 2025

            ## Styling Requirements:
            - Apply professional styling to all charts:
            - Legend (if multiple categories)
            - Rotate x-axis labels if needed
            - Tight layout (plt.tight_layout())

            </chart_generation_instructions>

            <critical_constraints>
            ## For SQL Generation
            - **Query Type Restriction**: Generate SELECT statements ONLY. Never create data modification queries (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE)
            - **Schema Fidelity**: Use exact table and column names as defined in the provided schema. Do not assume or invent schema elements
            - **Status Value Precision**: For order status filtering, use ONLY these exact values: 'delivered', 'processing', 'shipped'
            - **Table Aliasing**: Always use table aliases when joining multiple tables for improved readability
            - **Column Qualification**: Fully qualify all column references with table names or aliases when multiple tables are involved to prevent ambiguity
            - **Query Documentation**: Add explanatory comments for complex logic, business rules, or non-obvious query patterns
            - **Data Integrity**: Never fabricate, simulate, or assume data values in either SQL queries or visualizations
            - **PostgreSQL Compliance**: Ensure all SQL syntax is valid for PostgreSQL and compatible with AWS RDS environments

            ## For Python Code Generation
            - **Visualization Code Quality**: 
            - Generate only valid, executable Python code
            - Use only standard libraries: pandas, matplotlib, numpy, json, base64, datetime
            - Include all required imports
            - Handle data type conversions properly
            - Test logic mentally before generating

            - **Code Safety**:
            - No file system operations except saving to /tmp/chart.png
            - No network operations
            - No system calls or subprocess execution
            - No eval() or exec() usage

            - **Data Handling**:
            - Embed data as JSON string in the code
            - Handle missing values and edge cases
            - Validate data structure before visualization

            </critical_constraints>

            """

    def get_sample_questions(self, no_of_questions: int = MAX_SAMPLE_QUESTIONS) -> List[str]:
        """Get a list of sample questions that can be answered from the configured database."""

        log_info(self.logger, "SQLAgent.get_sample_questions", f"Starting function")
        # Use agent to generate the sample questions
        prompt = f"""
        # Database Question Generator

        <task_description>
        Generate {no_of_questions} realistic user questions that can be answered using the available database tables. These questions should represent common queries a user might ask about the data contained in these tables.
        </task_description>

        <context>
        You have access to database tables with various fields and relationships. Based on the structure and content of these tables, create questions that:
        - Can be directly answered by querying the database
        - Cover different types of information available in the tables
        - Represent practical information needs users might have
        </context>

        <instructions>
        1. Analyze the database schema and understand the available tables and their relationships
        2. Create {no_of_questions} distinct, natural-sounding questions that users might ask
        3. Ensure each question can be answered by querying the database tables
        4. Vary the complexity and types of questions (e.g., simple lookups, aggregations, comparisons)
        </instructions>

        <response_format>
        Provide your response as a JSON array of strings containing exactly {no_of_questions} questions.
        No additional text, explanations, or formatting should be included outside the JSON structure.

        Example format:
        ```json
        [
            "What is the total revenue for Q1 2023?",
            "Which customer made the largest purchase in the last month?",
        ...
        ]
        ```
        </response_format>

        Return ONLY the JSON array with {no_of_questions} questions, without any preamble or additional explanation.
        """
        try:
            response = self.agent(prompt)
            # Extract list of possible questions from response
            response_text = response.output if hasattr(response, 'output') else str(response)
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            json_list = json_match.group(1).strip()
            question_list = json.loads(json_list)

            log_info(self.logger, "SQLAgent.get_sample_questions", f"Ending function, Question List:{str(question_list)}")
            return question_list

        except Exception as e:
            log_info(self.logger, "SQLAgent.get_sample_questions", f"Failed, Error Info:{str(e)}")
            raise Exception(f"Failed to query SQL database for sample questions. Error Info: {str(e)}")
    
    def _generate_sql_from_query(self, user_query: str) -> Optional[str]:
        """Generate SQL query from natural language user query.
        
        Returns:
            SQL query string if successful, None if query cannot be answered
        """
        sql_prompt = f"""
        <task_description>
        You are an expert SQL query generator tasked with analyzing natural language requests and converting them into precise SQL queries when possible. Your goal is to determine if the user's request can be answered using the available database tables and generate the appropriate SQL query only when feasible.
        </task_description>

        <context>
        User Query: {user_query}
        </context>

        <instructions>
        Please follow this structured approach:

        1. **Analysis Phase**:
        - Carefully examine the user query to understand the information being requested
        - Review the available database tables and their schemas
        - Determine if the requested information can be retrieved from the available tables

        2. **Decision Process**:
        - If the query CAN be answered with the available tables:
            - Identify the relevant tables and columns needed
            - Determine necessary joins, conditions, and aggregations
            - Construct a syntactically correct and efficient SQL query
            - Include appropriate WHERE clauses to filter results as specified
            - Use proper SQL syntax with semicolons at the end

        - If the query CANNOT be answered with the available tables:
            - Explain briefly why the query cannot be fulfilled
            - Do not attempt to generate a SQL query
            - Suggest what additional information or tables might be needed

        3. **SQL Generation Phase**:
        - Remember: for SELECT DISTINCT, ORDER BY expressions must appear in select list 

        4. **Response Format**:
        When providing a SQL query, format it in a code block as shown:
        ```sql
        SELECT ...
        ```
        </instructions>
        """
        
        response = self.agent(sql_prompt)
        response_text = response.output if hasattr(response, 'output') else str(response)
        sql_match = re.search(r'```sql\n(.*?)\n```', response_text, re.DOTALL)
        
        if not sql_match:
            log_info(self.logger, "SQLAgent._generate_sql_from_query", 
                    f"No SQL was generated for user query, Reasons:{response_text}")
            return None
        
        sql = sql_match.group(1).strip()
        log_info(self.logger, "SQLAgent._generate_sql_from_query", 
                f"Generated SQL: {sql}", True)
        return sql
    
    def _generate_visualization_code(self, user_query: str, sql: str, 
                                     results: List[Dict[str, Any]], row_count: int) -> Optional[str]:
        """Generate Python visualization code from query results.
        
        Returns:
            Python code string if successful, None if no code generated
        """
        results_preview = results[:200] if results else []
        
        code_prompt = f"""
            <task_description>
            You are an expert Python code generator for data visualization. Generate matplotlib code to create a chart based on:
            1. User query
            2. SQL query that was executed
            3. The result data from the SQL query
            </task_description>

            <context>
            - User Query: {user_query}
            - SQL: {sql}
            - Results (first 200 rows): {json.dumps(results_preview, indent=2, default=str)}
            - Total Rows: {row_count}
            </context>

            <response_format>
            When providing Python code, format it in a code block as shown:
            ```python
            import pandas as pd
            import matplotlib.pyplot as plt
            ...
            ```
            </response_format>
            """
        
        code_response = self.agent(code_prompt)
        code_response_text = code_response.output if hasattr(code_response, 'output') else str(code_response)
        python_match = re.search(r'```python\n(.*?)\n```', code_response_text, re.DOTALL)
        
        if not python_match:
            log_info(self.logger, "SQLAgent._generate_visualization_code", 
                    "No Python code generated by agent")
            return None
        
        code = python_match.group(1).strip()
        log_info(self.logger, "SQLAgent._generate_visualization_code", 
                "Generated Python visualization code", True)
        return code
    
    def _generate_chart(self, user_query: str, sql: str, 
                       results: List[Dict[str, Any]], row_count: int) -> Optional[Dict[str, Any]]:
        """Generate chart visualization from query results.
        
        Returns:
            Chart result dictionary if successful, None otherwise
        """
        if not results or len(results) == 0:
            log_info(self.logger, "SQLAgent._generate_chart", "No results to visualize")
            return None
        
        log_info(self.logger, "SQLAgent._generate_chart", 
                "Generating Python code for chart visualization...")
        
        # Generate visualization code
        code = self._generate_visualization_code(user_query, sql, results, row_count)
        if not code:
            return None
        
        # Execute code in Code Interpreter
        chart_result = self.chart_tool.generate(code)
        
        if chart_result.get("success"):
            chart_url = chart_result.get('chart_url')
            log_info(self.logger, "SQLAgent._generate_chart", 
                    f"Chart generated successfully. Chart URL: {chart_url}")
        else:
            log_info(self.logger, "SQLAgent._generate_chart", 
                    f"Chart generation failed: {chart_result.get('message')}")
        
        return chart_result
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the agent pipeline with semantic caching."""
        
        log_info(self.logger, "SQLAgent.process_query", 
                f"Starting function, User Query:{user_query}, for tenant_id:{self.tenant_id}")
        
        try:
            # Step 0: Check cache for semantically similar queries
            cached_result = self.cache.check_cache(user_query, self.tenant_id)
            
            if cached_result:
                log_info(self.logger, "SQLAgent.process_query", 
                        f"Cache HIT! Returning cached results (similarity: {cached_result.get('similarity_score', 0):.4f})")
                return {
                    "type": "structured",
                    "status": "success",
                    "sql": cached_result['sql'],
                    "results": cached_result['results'],
                    "row_count": cached_result['row_count'],
                    "summary": "",
                    "chart_url": cached_result['chart_url'],
                    "cache_hit": True,
                    "cache_tier": cached_result.get('tier', 1)  # Default to Tier 1 if not specified
                }
            
            log_info(self.logger, "SQLAgent.process_query", "Cache MISS - proceeding with query execution")
            
            # Step 1: Generate SQL from user query
            sql = self._generate_sql_from_query(user_query)
            
            if not sql:
                # No SQL generated - query cannot be answered or needs clarification
                return {
                    "type": "structured",
                    "status": "no_sql",
                    "summary": "Query cannot be answered with available tables",
                    "sql": "",
                    "results": [],
                    "row_count": 0,
                    "chart_url": ""
                }
            
            # Step 1.5: Check Tier 2 cache with SQL validation
            tier2_cached = self.cache.check_cache_with_sql(user_query, self.tenant_id, sql)
            
            if tier2_cached:
                log_info(self.logger, "SQLAgent.process_query", 
                        f"Tier 2 Cache HIT! SQL match confirmed (similarity: {tier2_cached.get('similarity_score', 0):.4f})")
                return {
                    "type": "structured",
                    "status": "success",
                    "sql": tier2_cached['sql'],
                    "results": tier2_cached['results'],
                    "row_count": tier2_cached['row_count'],
                    "summary": "",
                    "chart_url": tier2_cached['chart_url'],
                    "cache_hit": True,
                    "cache_tier": 2
                }
            
            log_info(self.logger, "SQLAgent.process_query", "Tier 2 cache miss - executing query")
            
            # Step 2: Execute SQL
            execution = self.executor_tool.execute(sql)
            log_info(self.logger, "SQLAgent.process_query", 
                    f"Step 2: Executed SQL, Row Count: {execution['row_count']}")
            
            # Step 3: Generate chart visualization
            chart_result = self._generate_chart(
                user_query, 
                sql, 
                execution["results"], 
                execution['row_count']
            )
            chart_url = chart_result.get("chart_url") if chart_result else None

            log_info(self.logger, "SQLAgent.process_query", 
                    f"Step 3: Generated Chart, Chart_Url: {chart_url}")
            
            # Step 4: Store in cache for future queries
            if execution["results"]:  # Only cache if we have results
                self.cache.store_in_cache(
                    user_query,
                    self.tenant_id,
                    sql,
                    execution["results"],
                    chart_url,
                    execution["row_count"]
                )
            
            return {
                "type": "structured",
                "status": "success",
                "sql": sql,
                "results": execution["results"],
                "row_count": execution["row_count"],
                "summary": "",
                "chart_url": chart_url,
                "cache_hit": False
            }

        except Exception as e:
            log_info(self.logger, "SQLAgent.process_query", 
                    f"Failed to query SQL database. Error Info: {str(e)}")
            raise Exception(f"Failed to query SQL database. Error Info: {str(e)}")
