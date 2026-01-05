"""AWS Strands SQL Agent for RDS query validation and execution."""

import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import psycopg2
from psycopg2 import pool
import boto3
import logging
from datetime import datetime
import time

# Import Strands SDK
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.types.content import SystemContentBlock

# Type checking imports (avoid circular imports at runtime)
if TYPE_CHECKING:
    from agent.chart_agent import ChartAgent

# Define constants
MAX_ROWS_FOR_SQL_RESULT_DISPLAY = 20
MAX_SAMPLE_QUESTIONS = 5

def log_info(logger: logging.Logger, function: str, content: str, log_complete_content: bool = False) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not log_complete_content:
        logger.info(f"[{timestamp}] {function}: {content}..." if len(content) > 500 else f"[{timestamp}] {function}: {content}")
    else:
        logger.info(f"[{timestamp}] {function}: {content}...")


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
        self.similarity_threshold = self.valkey_config.get('similarity_threshold', 0.99)
        self.similarity_threshold_min = self.valkey_config.get('similarity_threshold_min', 0.70)
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
        
        This normalization preserves parameter values (like years, dates, IDs)
        to ensure queries with different parameters don't match.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL string
        """
        if not sql:
            return ""
        
        import re
        
        # Step 1: Remove comments FIRST (before collapsing whitespace)
        # Remove -- style comments (must be done before removing newlines)
        normalized = re.sub(r'--[^\n]*', '', sql)
        
        # Remove /* */ style comments
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        # Step 2: Normalize whitespace and case
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        # Remove trailing semicolon
        normalized = normalized.rstrip(';')
        
        # IMPORTANT: We preserve parameter values (years, dates, IDs, etc.)
        # This ensures queries with different parameters don't match
        # Example: "WHERE year = 2023" != "WHERE year = 2024"
        
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
    
    def check_cache(self, user_query: str, tenant_id: str, generated_sql: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar query exists in cache with optional SQL validation.
        
        This unified method handles both Tier 1 (high similarity) and Tier 2 (SQL validation) caching:
        - Tier 1: When generated_sql is None, returns cache hit for very high similarity (≥ 0.95)
        - Tier 2: When generated_sql is provided, validates SQL match for medium+ similarity (≥ 0.75)
        
        Args:
            user_query: User's natural language question
            tenant_id: Tenant ID for isolation
            generated_sql: Optional SQL query for Tier 2 validation. If None, only Tier 1 check is performed.
            
        Returns:
            Cached results dictionary if cache hit, None if cache miss
        """
        if not self.cache_enabled:
            return None
        
        try:
            import time
            import numpy as np
            
            # Determine which tier we're checking
            tier_mode = "Tier 1 (high similarity)" if generated_sql is None else "Tier 2 (SQL validation)"
            self.logger.info(f"Checking cache for query ({tier_mode}): {user_query[:500]}...")
            
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
            
            # Extract cached document for potential use
            json_doc_bytes = doc_dict.get('$')
            if not json_doc_bytes:
                self.logger.error("No JSON document found in cache result")
                return None
            
            # Parse the cached document with error handling
            try:
                json_doc_str = json_doc_bytes.decode('utf-8') if isinstance(json_doc_bytes, bytes) else str(json_doc_bytes)
                cached_doc = json.loads(json_doc_str)
                
                # Verify the parsed document is valid
                if not cached_doc or not isinstance(cached_doc, dict):
                    self.logger.error(f"Invalid cached document after JSON parsing: {type(cached_doc)}")
                    return None
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse cached document JSON: {str(e)}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error parsing cached document: {str(e)}")
                return None
            
            # TIER 1: High similarity check (no SQL validation)
            if generated_sql is None:
                # Conservative Tier 1: Only cache hit for VERY high similarity (≥ 0.99)
                # This reduces false positives for queries with different parameters
                TIER1_THRESHOLD = self.similarity_threshold
                
                if similarity_score >= TIER1_THRESHOLD:
                    self.logger.info(f"Tier 1 CACHE HIT! Very high similarity: {similarity_score:.4f} ≥ {TIER1_THRESHOLD}")
                    
                    # Extract cached data
                    cached_sql = cached_doc.get('sql_query')
                    cached_results_json = cached_doc.get('results')
                    cached_chart_url = cached_doc.get('chart_url')
                    cached_row_count = cached_doc.get('row_count')
                    
                    # Parse results if it's a JSON string
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
                        'tier': 1,
                        'similarity_score': similarity_score
                    }
                else:
                    self.logger.info(f"Tier 1 cache miss: similarity {similarity_score:.4f} < {TIER1_THRESHOLD} (conservative threshold)")
                    return None
            
            # TIER 2: SQL validation check
            else:
                # Tier 2 works for medium+ similarity (≥ 0.70)
                if similarity_score >= self.similarity_threshold_min:
                    self.logger.info(f"Tier 2 candidate found with similarity {similarity_score:.4f}, validating SQL...")
                    
                    # Extract cached SQL
                    cached_sql = cached_doc.get('sql_query', '')
                    
                    # Normalize and compare SQL
                    normalized_cached = self._normalize_sql(cached_sql)
                    normalized_generated = self._normalize_sql(generated_sql)
                    
                    # Log the comparison for debugging
                    self.logger.info(f"Tier 2 SQL Comparison:")
                    self.logger.info(f"  Cached SQL (normalized, len={len(normalized_cached)}): {normalized_cached[:500]}{'...' if len(normalized_cached) > 500 else ''}")
                    self.logger.info(f"  Generated SQL (normalized, len={len(normalized_generated)}): {normalized_generated[:500]}{'...' if len(normalized_generated) > 500 else ''}")
                    self.logger.info(f"  Match: {normalized_cached == normalized_generated}")
                    
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
                        self.logger.info(f"Tier 2 cache miss: SQL different")
                        return None
                else:
                    self.logger.info(f"Tier 2 cache miss: similarity {similarity_score:.4f} < {self.similarity_threshold_min}")
                    return None
            
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
        valkey_config: Optional[Dict[str, Any]] = None,
        session_id: str = "unknown",
        token_callback: Optional[callable] = None,
        chart_agent: Optional['ChartAgent'] = None
    ):

        # Get AWS region & other config value from environment or parameter
        self.db_pool = db_pool
        self.model_id = model_id
        self.region = aws_region
        self.logger = logger
        self.tenant_id = tenant_id
        self.session_id = session_id  # Store session_id for metrics
        self.token_callback = token_callback  # Store callback for token accumulation
        self.chart_s3_bucket = chart_s3_bucket
        self.chart_agent = chart_agent  # Store chart_agent reference for visualization
        log_info(logger, "SQLAgent.Init", f"Starting function, Region:{aws_region}, Model ID:{model_id}, Tenant ID:{tenant_id}, Session ID:{session_id}, Chart S3 Bucket:{chart_s3_bucket}, Chart Agent: {'Provided' if chart_agent else 'None'}")

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
            You are an elite SQL query specialist with deep expertise in PostgreSQL for AWS RDS environments.
            Transform natural language data requests into highly optimized, production-ready SQL queries that leverage the provided database schema effectively.
            </task_description>

            <database_schema>
            {schema_context}
            </database_schema>

            <instructions>
            Follow this systematic, step-by-step approach when crafting SQL queries:

            **Step 1 - Request Analysis**
            - Thoroughly examine the user's natural language request to extract the core data requirements
            - Review the available database tables and their schemas
            - Determine if the requested information can be retrieved from the available tables
            - Identify key entities, relationships, filters, and aggregations needed
            - Clarify any ambiguous terms or conditions before proceeding

            **Step 2 - Schema Validation & Feasibility Check**
            - Cross-reference the user's request against the provided database schema
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

            **Step 3 - Query Decomposition**
            - For complex requests, break down the problem into logical sub-components
            - Identify whether subqueries, CTEs (Common Table Expressions), or multiple steps are needed
            - Plan the query structure before writing SQL code

            **Step 4 - SQL Query Construction**
            Construct your SQL query with meticulous attention to detail:

            a) **SELECT Clause**: Specify exactly the columns requested, using appropriate aliases for clarity
            b) **FROM & JOIN Clauses**: 
            - Use proper schema with tables
            - Use proper table aliases for readability
            - Implement appropriate JOIN types (INNER, LEFT, RIGHT, FULL) based on the data relationships
            - Base JOINs on foreign key relationships defined in the schema
            c) **WHERE Clause**: 
            - Apply precise filtering conditions that match the user's requirements
            - Use correct data types and formats for comparison values
            - For order status filtering, use only these exact values: 'delivered', 'processing', or 'shipped'
            - **CRITICAL - String Literal Escaping**: When filtering by text values (product names, descriptions, etc.):
              * ALWAYS use single quotes (') to delimit string literals in PostgreSQL
              * If the string value contains a single quote, escape it by doubling it ('')
              * Example: For product with double quote char, use: WHERE product_name = 'Laptop Pro 15"'
              * Example: For product "Women's Jacket", use: WHERE product_name = 'Women''s Jacket'
              * Double quotes (") inside single-quoted strings do NOT need escaping
              * NEVER use double quotes to delimit string values - they are for identifiers only
            d) **GROUP BY & Aggregation**: 
            - Apply aggregation functions (COUNT, SUM, AVG, MAX, MIN) when summarization is needed
            - Ensure all non-aggregated columns in SELECT appear in GROUP BY
            e) **HAVING Clause**: Use for filtering aggregated results when necessary
            f) **ORDER BY Clause**: Sort results meaningfully based on the context (e.g., chronological, alphabetical, by magnitude)
            g) **LIMIT Clause**: Include reasonable limits to prevent overwhelming result sets, especially for exploratory queries

            - **CRITICAL - for SELECT DISTINCT, ORDER BY expressions must appear in select list**
        
            - **CRITICAL - Schema Prefix Requirement**: ALL table references MUST include the schema prefix
                * **Rule**: ALWAYS prefix table names with the schema name from the database schema provided
                * Example CORRECT:
                ```sql
                FROM tenanta.customers c
                JOIN tenanta.orders o ON c.customer_id = o.customer_id
                JOIN tenanta.order_items oi ON o.order_id = oi.order_id
                ```
                * Example INCORRECT:
                ```sql
                FROM customers c  -- ❌ Missing schema prefix!
                JOIN orders o ON c.customer_id = o.customer_id  -- ❌ Missing schema prefix!
                ```
                * **Rule**: Apply schema prefix to EVERY table in FROM, JOIN, subqueries, and CTEs
                * **Rule**: The schema name is shown in the CREATE TABLE statements (e.g., `CREATE TABLE tenanta.customers`)
                * **Validation**: Before finalizing SQL, verify EVERY table reference has the schema prefix
                * **Common mistake**: Forgetting schema prefix causes "relation does not exist" error
            
            - **CRITICAL - Table Alias Consistency**: When using table aliases in JOINs, you MUST use the EXACT SAME alias throughout the query
                * Example CORRECT: 
                ```sql
                SELECT ca.category_id, ca.category_name
                FROM tenantb.categories ca
                JOIN tenantb.products p ON ca.category_id = p.category_id
                ```
                * Example INCORRECT:
                ```sql
                SELECT c.category_id, ca.category_name  -- ❌ WRONG! Using 'c' but alias is 'ca'
                FROM tenantb.categories ca
                JOIN tenantb.products p ON ca.category_id = p.category_id
                ```
                * **Rule**: If you define an alias as "ca" in the FROM/JOIN clause, use "ca" everywhere (SELECT, WHERE, GROUP BY, ORDER BY)
                * **Rule**: Never mix aliases - if the table is aliased as "ca", don't use "c" or "cat" or any other variation
                * **Rule**: In CTEs (WITH clauses), ensure aliases are consistent within each CTE and when referencing CTE columns
            
            - **CRITICAL - JOIN Condition Data Type Matching**: When writing JOIN conditions, ensure columns have MATCHING data types
                * **Rule**: ALWAYS join foreign keys to primary keys (ID columns to ID columns), NEVER to name/description columns
                * Example CORRECT: 
                ```sql
                JOIN tenantb.categories ca ON p.category_id = ca.category_id  -- integer = integer ✅
                SELECT ca.category_name  -- Get the name in SELECT, not in JOIN
                ```
                * Example INCORRECT:
                ```sql
                JOIN tenantb.categories ca ON p.category_id = ca.category_name  -- integer = varchar ❌ TYPE MISMATCH!
                ```
                * **Common mistake**: Seeing `category_name` in SELECT clause and accidentally using it in JOIN condition
                * **Validation checklist for EVERY JOIN**:
                    1. Is the left side an ID column? (e.g., `p.category_id`)
                    2. Is the right side also an ID column? (e.g., `ca.category_id`)
                    3. Are both columns the same data type? (both integer, or both varchar)
                * **Pattern**: To get a name/description, join on IDs first, then SELECT the name
                * **CTE Warning**: When creating multiple similar CTEs, don't copy-paste JOIN conditions blindly - verify each JOIN uses the correct ID columns
                * **Foreign Key Relationships to use**:
                    - products.category_id → categories.category_id (join on IDs)
                    - orders.customer_id → customers.customer_id (join on IDs)
                    - order_items.product_id → products.product_id (join on IDs)
                    - order_items.order_id → orders.order_id (join on IDs)

            **Step 5 - PostgreSQL Optimization**
            - Leverage PostgreSQL-specific features and AWS RDS best practices
            - Ensure queries are efficient and performant
            - Use appropriate indexing strategies in your query design
            - Add inline comments (-- or /* */) to explain complex logic, subqueries, or business rules

            **Step 6 - Query Validation**
            - Ensure all tables references are fully qualified with schema names
            - Ensure all column references are fully qualified with table names or aliases when multiple tables are involved
            - Verify that the query syntax is valid PostgreSQL
            - Confirm the query addresses all aspects of the user's request
            - **Validate String Literal Escaping**: Double-check that:
              * All string values use single quotes (')
              * Any single quotes within string values are properly escaped ('')
              * Product names with special characters are correctly handled
              * Example: Single quotes for string values, double quotes only for identifiers

            </instructions>

            <critical_constraints>
            - **Query Type Restriction**: Generate SELECT statements ONLY. Never create data modification queries (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE)
            - **Schema Fidelity**: Use exact table and column names as defined in the provided schema. Do not assume or invent schema elements
            - **Status Value Precision**: For order status filtering, use ONLY these exact values: 'delivered', 'processing', 'shipped'
            - **String Literal Safety**: 
              * ALWAYS use single quotes (') for string literals in SQL
              * Escape single quotes within strings by doubling them ('')
              * Double quotes (") within single-quoted strings need NO escaping
              * NEVER use double quotes (") for string values - reserved for identifiers only
              * Examples:
                - Product with double quote: WHERE product_name = 'Laptop Pro 15"'
                - Product "Women's Jacket" → WHERE product_name = 'Women''s Jacket'
                - Product with inch symbol: WHERE product_name = '15" Monitor'
            - **Table Aliasing**: Always use table aliases when joining multiple tables for improved readability
            - **Column Qualification**: Fully qualify all column references with table names or aliases when multiple tables are involved to prevent ambiguity
            - **Query Documentation**: Add explanatory comments for complex logic, business rules, or non-obvious query patterns
            - **Data Integrity**: Never fabricate, simulate, or assume data values in either SQL queries or visualizations
            - **PostgreSQL Compliance**: Ensure all SQL syntax is valid for PostgreSQL and compatible with AWS RDS environments

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
            #Note the start time
            start_time = time.time()
            total_input_tokens = 0
            total_output_tokens = 0

            #Execute the agent to get response
            response = self.agent(prompt)
            # Extract list of possible questions from response
            response_text = response.output if hasattr(response, 'output') else str(response)
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            json_list = json_match.group(1).strip()
            question_list = json.loads(json_list)

            #Note the end time along with the token usage
            end_time = time.time()
            processing_duration_in_secs = abs(end_time - start_time)
            summary = response.metrics.get_summary()
            if summary and "accumulated_usage" in summary:
                total_input_tokens  = summary["accumulated_usage"].get("inputTokens",0)
                total_output_tokens = summary["accumulated_usage"].get("outputTokens",0)

            # Emit step metrics
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_sample_questions",
                start_time=start_time,
                end_time=end_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                status="success",
                additional_data={
                    "questions_generated": len(question_list),
                    "requested_count": no_of_questions
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(total_input_tokens, total_output_tokens, "sql_sample_questions")

            return question_list

        except Exception as e:
            end_time = time.time()
            log_info(self.logger, "SQLAgent.get_sample_questions", f"Failed, Error Info:{str(e)}")
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_sample_questions",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "error": str(e)
                }
            )
            
            raise Exception(f"Failed to query SQL database for sample questions. Error Info: {str(e)}")
    
    def _generate_sql_from_query(self, user_query: str) -> Optional[str]:
        """Generate SQL query from natural language user query.
        
        Returns:
            SQL query string if successful, None if query cannot be answered
        """
        sql_prompt = f"""
        <task_description>
        You are an expert SQL query generator tasked with analyzing natural language requests and converting them into precise SQL queries when possible. 
        Your goal is to determine if the user's request can be answered using the available database tables and generate the appropriate SQL query only when feasible.
        </task_description>

        <context>
        User Query: {user_query}
        </context>

        <response_format>
        When providing a SQL query, format it in a code block as shown:
        ```sql
        SELECT ...
        ```
        </response_format>
        """
        
        #Note the start time
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            #Execute the agent to get response
            response = self.agent(sql_prompt)
            response_text = response.output if hasattr(response, 'output') else str(response)
            sql_match = re.search(r'```sql\n(.*?)\n```', response_text, re.DOTALL)
            
            #Note the end time along with the token usage
            end_time = time.time()
            processing_duration_in_secs = abs(end_time - start_time)
            summary = response.metrics.get_summary()
            if summary and "accumulated_usage" in summary:
                total_input_tokens  = summary["accumulated_usage"].get("inputTokens",0)
                total_output_tokens = summary["accumulated_usage"].get("outputTokens",0)
            
            if not sql_match:
                log_info(self.logger, "SQLAgent._generate_sql_from_query", 
                        f"No SQL was generated for user query, Reasons:{response_text}")
                
                # Emit metrics for no SQL generated
                self.logger.emit_step_metrics(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    step_name="sql_generation",
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    status="success",
                    additional_data={
                        "sql_generated": False,
                        "reason": f"query_not_answerable: {response_text}"
                    }
                )
                
                # Report tokens to parent via callback
                if self.token_callback:
                    self.token_callback(total_input_tokens, total_output_tokens, "sql_generation")
                
                return None
            
            sql = sql_match.group(1).strip()
            
            # Emit metrics for successful SQL generation
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_generation",
                start_time=start_time,
                end_time=end_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                status="success",
                additional_data={
                    "sql_generated": True,
                    "sql_length": len(sql)
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(total_input_tokens, total_output_tokens, "sql_generation")

            return sql
            
        except Exception as e:
            end_time = time.time()
            log_info(self.logger, "SQLAgent._generate_sql_from_query", f"Error generating SQL: {str(e)}")
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_generation",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "error": str(e)
                }
            )
            
            return None
    

    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the agent pipeline with semantic caching."""
        
        log_info(self.logger, "SQLAgent.process_query", 
                f"Starting function, User Query:{user_query}, for tenant_id:{self.tenant_id}")
        
        import time
        start_time = time.time()
        
        try:
            # Step 0: Check Tier 1 cache (high similarity, no SQL validation)
            cached_result = self.cache.check_cache(user_query, self.tenant_id, generated_sql=None)
            
            if cached_result:
                end_time = time.time()
                log_info(self.logger, "SQLAgent.process_query", 
                        f"Tier 1 Cache HIT! Returning cached results (similarity: {cached_result.get('similarity_score', 0):.4f})")
                
                # Emit metrics for cache hit
                self.logger.emit_step_metrics(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    step_name="sql_query_cached",
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=0,
                    output_tokens=0,
                    status="success",
                    additional_data={
                        "cache_hit": True,
                        "cache_tier": cached_result.get('tier', 1),
                        "row_count": cached_result['row_count'],
                        "similarity_score": cached_result.get('similarity_score', 0),
                        "cached_sql":cached_result.get('sql', "")
                    }
                )
                
                return {
                    "type": "structured",
                    "status": "success",
                    "sql": cached_result['sql'],
                    "results": cached_result['results'],
                    "row_count": cached_result['row_count'],
                    "summary": "",
                    "chart_url": cached_result['chart_url'],
                    "cache_hit": True,
                    "cache_tier": cached_result.get('tier', 1)
                }
            
            log_info(self.logger, "SQLAgent.process_query", "Tier 1 cache miss - generating SQL")
            
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
            tier2_cached = self.cache.check_cache(user_query, self.tenant_id, generated_sql=sql)
            
            if tier2_cached:
                end_time = time.time()
                log_info(self.logger, "SQLAgent.process_query", 
                        f"Tier 2 Cache HIT! SQL match confirmed (similarity: {tier2_cached.get('similarity_score', 0):.4f})")
                
                # Emit metrics for tier 2 cache hit
                self.logger.emit_step_metrics(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    step_name="sql_query_cached",
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=0,
                    output_tokens=0,
                    status="success",
                    additional_data={
                        "cache_hit": True,
                        "cache_tier": 2,
                        "row_count": tier2_cached['row_count'],
                        "similarity_score": tier2_cached.get('similarity_score', 0),
                        "cached_sql":tier2_cached.get('sql', "")
                    }
                )
                
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

            # Step 2.5: Limit the SQL result going forward to a max of 100 rows for memory & optimization reasons
            if execution and "results" in execution:
                limited_sql_results = execution["results"][:100]
            
            # Step 3: Generate chart visualization
            chart_result = self.chart_agent.generate_chart(
                user_query, 
                sql, 
                limited_sql_results, 
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
                    limited_sql_results, # Only cache the top 100 rows
                    chart_url,
                    execution["row_count"]
                )
            
            end_time = time.time()
            
            # Emit metrics for successful query execution
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_agent_execution",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="success",
                additional_data={
                    "cache_hit": False,
                    "row_count": execution["row_count"],
                    "has_chart": chart_url is not None,
                    "sql_length": len(sql)
                }
            )
            
            return {
                "type": "structured",
                "status": "success",
                "sql": sql,
                "results": limited_sql_results,
                "row_count": execution["row_count"],
                "summary": "",
                "chart_url": chart_url,
                "cache_hit": False
            }

        except Exception as e:
            end_time = time.time()
            log_info(self.logger, "SQLAgent.process_query", 
                    f"Failed to query SQL database. Error Info: {str(e)}")
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_agent_execution",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "cache_hit": False,
                    "error": str(e)
                }
            )
            
            raise Exception(f"Failed to query SQL database. Error Info: {str(e)}")
