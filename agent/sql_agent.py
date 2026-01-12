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
        Normalize SQL query for string comparison.
        Preserves parameter values for strict matching.
        
        NOTE: This normalization is for STRUCTURE comparison only.
        It lowercases SQL keywords and identifiers, but preserves
        string literal values (in quotes) for strict matching.
        """
        if not sql:
            return ""
        
        import re
        
        # Step 1: Remove comments FIRST (before collapsing whitespace)
        # Remove -- style comments (must be done before removing newlines)
        normalized = re.sub(r'--[^\n]*', '', sql)
        
        # Remove /* */ style comments
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        # Step 2: Preserve string literals by temporarily replacing them
        # This ensures string literal case is preserved during normalization
        string_literals = []
        def preserve_string(match):
            string_literals.append(match.group(0))
            return f"__STRING_LITERAL_{len(string_literals)-1}__"
        
        # Match single and double quoted strings
        normalized = re.sub(r"'[^']*'", preserve_string, normalized)
        normalized = re.sub(r'"[^"]*"', preserve_string, normalized)
        
        # Step 3: Normalize whitespace and case (but not string literals)
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        # Remove trailing semicolon
        normalized = normalized.rstrip(';')
        
        # Step 4: Restore string literals (preserving their original case)
        for i, literal in enumerate(string_literals):
            normalized = normalized.replace(f"__string_literal_{i}__", literal)
        
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
    
    def check_cache(self, user_query: str, tenant_id: str, generated_sql: Optional[str] = None, visual_requested: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar query exists in cache with optional SQL validation.
        
        This unified method handles both Tier 1 (high similarity) and Tier 2 (SQL validation) caching:
        - Tier 1: When generated_sql is None, returns cache hit for very high similarity (≥ 0.95)
        - Tier 2: When generated_sql is provided, validates SQL match for medium+ similarity (≥ 0.75)
        
        Args:
            user_query: User's natural language question
            tenant_id: Tenant ID for isolation
            generated_sql: Optional SQL query for Tier 2 validation. If None, only Tier 1 check is performed.
            visual_requested: Optional. If the user has explicitoy asked for visuals
            
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
                    
                    # Extract cached SQL & if visual requested
                    cached_sql = cached_doc.get('sql_query', '')
                    cached_visual_requested = cached_doc.get('visual_requested', False)
                    
                    # Normalize and compare SQL
                    normalized_cached = self._normalize_sql(cached_sql)
                    normalized_generated = self._normalize_sql(generated_sql)
                    
                    # Log the comparison for debugging
                    self.logger.info(f"Tier 2 SQL Comparison:")
                    self.logger.info(f"  Cached SQL (normalized, len={len(normalized_cached)}): {normalized_cached[:500]}{'...' if len(normalized_cached) > 500 else ''}")
                    self.logger.info(f"  Generated SQL (normalized, len={len(normalized_generated)}): {normalized_generated[:500]}{'...' if len(normalized_generated) > 500 else ''}")
                    self.logger.info(f"  Match: {normalized_cached == normalized_generated}")
                    self.logger.info(f"  Cached Visual Requested: {cached_visual_requested}, Now Visual Requested: {visual_requested}")
                    
                    if normalized_cached == normalized_generated and cached_visual_requested == visual_requested:
                        # SQL & visual_requested matches- Tier 2 cache hit!
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
        visual_requested: bool,
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
                'visual_requested': visual_requested,
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
        
        # DIAGNOSTIC: Log system prompt token count for cache validation
        # AWS requires minimum 1,024 tokens for Claude Sonnet 4.5 (4,096 for Haiku)
        # If below minimum, cache checkpoint will be silently ignored
        prompt_length = len(system_prompt)
        estimated_tokens = prompt_length // 4  # Rough estimate: 1 token ≈ 4 characters
        log_info(logger, "SQLAgent.Init", 
                f"Step 4: Built system prompt (length: {prompt_length} chars, ~{estimated_tokens} tokens)")
        
        if estimated_tokens < 1024:
            logger.warning(
                f"SQL Agent system prompt may be too short for prompt caching. "
                f"Estimated ~{estimated_tokens} tokens, but minimum 1,024 tokens required for Claude Sonnet 4.5. "
                f"Cache checkpoint will be IGNORED by AWS if below minimum. "
            )
        
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
            <task_description>
            You are an elite SQL query specialist with deep expertise in PostgreSQL for AWS RDS environments. 
            Your mission is to transform natural language data requests into highly optimized, production-ready SQL queries that leverage the provided database schema effectively and accurately.
            </task_description>

            <database_schema>
            {schema_context}
            </database_schema>

            <instructions>
            Follow this systematic, step-by-step approach when crafting SQL queries:

            **Step 1: Request Analysis & Understanding**
            - Carefully read and analyze the user's natural language request to extract the core data requirements
            - Identify the key entities, attributes, relationships, filters, aggregations, and sorting criteria mentioned
            - Review the available database tables and their schemas provided above
            - Clarify any ambiguous terms or conditions in your understanding before proceeding
            - Determine whether the requested information can be retrieved from the available schema

            **Step 2: Schema Validation & Feasibility Assessment**
            - Cross-reference the user's request against the provided database schema
            - **If the query CAN be answered** with the available tables:
                * Identify all relevant tables and columns needed
                * Map out the necessary joins based on foreign key relationships
                * Determine required conditions, aggregations, and sorting logic
                * Proceed to construct the SQL query
            - **If the query CANNOT be answered** with the available tables:
                * Clearly explain why the query cannot be fulfilled
                * Specify what tables, columns, or information is missing
                * Do NOT attempt to generate a SQL query
                * Suggest what additional schema elements would be needed

            **Step 3: Query Planning & Decomposition**
            - For complex requests, break down the problem into logical sub-components
            - Identify whether subqueries, CTEs (Common Table Expressions), or window functions are needed
            - Plan the overall query structure and execution flow before writing SQL code
            - Consider performance implications and optimization opportunities

            **Step 4: SQL Query Construction**

            Construct your SQL query with meticulous attention to detail, following these guidelines:

            a) **SELECT Clause**:
            - Specify exactly the columns requested in the user's question
            - Use meaningful aliases for calculated fields and aggregations
            - Ensure column names are clear and descriptive

            b) **FROM & JOIN Clauses**:
            - **CRITICAL - Schema Prefix Requirement**: ALL table references MUST include the schema prefix
                * ALWAYS prefix table names with the schema name (e.g., `tenanta.customers`, `tenantb.products`)
                * Apply schema prefix to EVERY table in FROM, JOIN, subqueries, and CTEs
                * Example CORRECT: `FROM tenanta.customers c JOIN tenanta.orders o ON c.customer_id = o.customer_id`
                * Example INCORRECT: `FROM customers c` (missing schema prefix - will cause "relation does not exist" error)
            - Use descriptive table aliases for improved readability (e.g., `c` for customers, `o` for orders, `oi` for order_items)
            - **CRITICAL - Table Alias Consistency**: Once you define an alias, use it consistently throughout the entire query
                * If you alias a table as "ca", use "ca" everywhere (SELECT, WHERE, GROUP BY, ORDER BY, HAVING)
                * NEVER mix aliases (e.g., don't use "c" in one place and "ca" in another for the same table)
                * Example CORRECT: `FROM tenantb.categories ca ... SELECT ca.category_name ... WHERE ca.category_id = 5`
                * Example INCORRECT: `FROM tenantb.categories ca ... SELECT c.category_name` (inconsistent alias)
            - Implement appropriate JOIN types (INNER, LEFT, RIGHT, FULL OUTER) based on data relationships and requirements
            - **CRITICAL - JOIN Condition Data Type Matching**: Ensure columns in JOIN conditions have matching data types
                * ALWAYS join ID columns to ID columns (foreign key to primary key)
                * NEVER join ID columns to name/description columns
                * Example CORRECT: `JOIN tenantb.categories ca ON p.category_id = ca.category_id` (integer = integer)
                * Example INCORRECT: `JOIN tenantb.categories ca ON p.category_id = ca.category_name` (integer = varchar - TYPE MISMATCH!)
                * To retrieve names/descriptions, join on IDs first, then SELECT the name columns
                * Validation: For each JOIN, verify both sides are ID columns with matching data types

            c) **WHERE Clause**:
            - Apply precise filtering conditions that match the user's requirements
            - Use correct data types and formats for comparison values
            - For order status filtering, use ONLY these exact values: 'delivered', 'processing', or 'shipped'
            - **CRITICAL - String Literal Escaping Rules**:
                * ALWAYS use single quotes (') to delimit string literals in PostgreSQL
                * If a string value contains a single quote, escape it by doubling it ('')
                * Double quotes (") inside single-quoted strings do NOT need escaping
                * NEVER use double quotes (") to delimit string values - they are reserved for identifiers only
                * Examples:
                - Product with double quote: `WHERE product_name = 'Laptop Pro 15"'`
                - Product "Women's Jacket": `WHERE product_name = 'Women''s Jacket'`
                - Product with inch symbol: `WHERE product_name = '15" Monitor'`
            - Fully qualify column references with table aliases to prevent ambiguity

            d) **GROUP BY & Aggregation**:
            - Apply aggregation functions (COUNT, SUM, AVG, MAX, MIN) when summarization is required
            - Ensure all non-aggregated columns in SELECT appear in the GROUP BY clause
            - Use appropriate aggregation logic based on the user's request

            e) **HAVING Clause**:
            - Use HAVING to filter aggregated results when necessary
            - Apply conditions on aggregate functions (e.g., `HAVING COUNT(*) > 10`)

            f) **ORDER BY Clause**:
            - Sort results meaningfully based on context (chronological, alphabetical, by magnitude)
            - **CRITICAL**: When using SELECT DISTINCT, all ORDER BY expressions MUST appear in the SELECT list
            - Use DESC or ASC explicitly to clarify sorting direction

            g) **LIMIT Clause**:
            - Include reasonable limits to prevent overwhelming result sets
            - Use LIMIT for exploratory queries or when top-N results are requested

            **Step 5: PostgreSQL Optimization & Best Practices**
            - Leverage PostgreSQL-specific features and AWS RDS best practices
            - Design queries for efficiency and performance
            - Consider indexing strategies in your query design
            - Use CTEs (WITH clauses) for complex queries to improve readability
            - Add inline comments (-- or /* */) to explain complex logic, subqueries, or business rules
            - Ensure queries are maintainable and well-documented

            **Step 6: Query Validation & Quality Assurance**
            Before finalizing your SQL query, verify:
            - All table references include schema prefixes (e.g., `tenanta.customers`)
            - All column references are fully qualified with table aliases when multiple tables are involved
            - Table aliases are used consistently throughout the query
            - JOIN conditions use matching data types (ID to ID, not ID to name)
            - String literals use single quotes (') with proper escaping for embedded quotes
            - The query syntax is valid PostgreSQL
            - The query addresses all aspects of the user's request
            - All requested columns are included in the SELECT statement
            - Conditions and filters accurately reflect the user's requirements
            - The query follows best practices for readability and performance
            - Before returning the query, mentally simulate execution to ensure:
                a. All table and column names exist in the provided schema.
                b. All joins are on compatible types.
                c. No aggregation errors (all non-aggregates are in GROUP BY).
            - If you detect a likely error (missing column, invalid reference), correct the query before returning it and do not expose the incorrect version.
            </instructions>

            <critical_constraints>
            - **Query Type Restriction**: Generate SELECT statements ONLY. Never create data modification queries (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE)
            - **Single-statement rule**: Do not generate multiple SQL statements; never include ; in the middle of the query.
            - **Tenant-scoping guidance**: The SQL should always refer to tables that belong one database schema. No cross schema referencing allowed.
            - **Schema Fidelity**: Use exact table and column names as defined in the provided schema. Do not assume, invent, or fabricate schema elements
            - **Schema Prefix Mandate**: Every table reference must include the schema prefix (e.g., `tenanta.orders`, not just `orders`)
            - **Status Value Precision**: For order status filtering, use ONLY these exact values: 'delivered', 'processing', 'shipped'
            - **String Literal Safety**:
                * ALWAYS use single quotes (') for string literals in SQL
                * Escape single quotes within strings by doubling them ('')
                * Double quotes (") within single-quoted strings need NO escaping
                * NEVER use double quotes (") for string values - reserved for identifiers only
            - **Table Aliasing**: Always use table aliases when joining multiple tables for improved readability
            - **Alias Consistency**: Use the same alias consistently throughout the query for each table
            - **Column Qualification**: Fully qualify all column references with table aliases when multiple tables are involved
            - **JOIN Type Matching**: Ensure JOIN conditions match columns with compatible data types (ID to ID)
            - **Query Documentation**: Add explanatory comments for complex logic, business rules, or non-obvious query patterns
            - **Data Integrity**: Never fabricate, simulate, or assume data values not present in the schema
            - **PostgreSQL Compliance**: Ensure all SQL syntax is valid for PostgreSQL and compatible with AWS RDS environments
            - **Query Termination**: End all SQL queries with a semicolon (;)
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
            cache_read_tokens = 0
            cache_write_tokens = 0
            if summary and "accumulated_usage" in summary:
                total_input_tokens  = summary["accumulated_usage"].get("inputTokens",0)
                total_output_tokens = summary["accumulated_usage"].get("outputTokens",0)
                cache_read_tokens = summary["accumulated_usage"].get("cacheReadInputTokens", 0)
                cache_write_tokens = summary["accumulated_usage"].get("cacheWriteInputTokens", 0)

            # Emit step metrics
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="sql_sample_questions",
                start_time=start_time,
                end_time=end_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                status="success",
                additional_data={
                    "questions_generated": len(question_list),
                    "requested_count": no_of_questions
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(total_input_tokens, total_output_tokens, "sql_sample_questions", cache_read_tokens, cache_write_tokens)

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
                cache_read_tokens=0,
                cache_write_tokens=0,
                status="error",
                additional_data={
                    "error": str(e)
                }
            )
            
            raise Exception(f"Failed to query SQL database for sample questions. Error Info: {str(e)}")
    
    def _generate_sql_from_query(self, user_query: str, user_email: str) -> Optional[str]:
        """Generate SQL query from natural language user query.
        
        Returns:
            SQL query string if successful, None if query cannot be answered
        """
        sql_prompt = f"""
        <task_description>
        - You are an expert SQL query generator tasked with analyzing natural language requests and converting them into precise SQL queries when possible. 
        - Your goal is to determine if the user's request can be answered using the available database tables and generate the appropriate SQL query only when feasible.
        - For queries about "my orders" "my products", "my shipments" or any query intent related to the current user JOIN the {self.tenant_id}.Customers TABLE add a WHERE clause: `{self.tenant_id}.customers.email='{user_email}'`
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
            cache_read_tokens = 0
            cache_write_tokens = 0
            if summary and "accumulated_usage" in summary:
                total_input_tokens  = summary["accumulated_usage"].get("inputTokens",0)
                total_output_tokens = summary["accumulated_usage"].get("outputTokens",0)
                cache_read_tokens = summary["accumulated_usage"].get("cacheReadInputTokens", 0)
                cache_write_tokens = summary["accumulated_usage"].get("cacheWriteInputTokens", 0)
            
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
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    status="success",
                    additional_data={
                        "sql_generated": False,
                        "reason": f"query_not_answerable: {response_text}"
                    }
                )
                
                # Report tokens to parent via callback
                if self.token_callback:
                    self.token_callback(total_input_tokens, total_output_tokens, "sql_generation", cache_read_tokens, cache_write_tokens)
                
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
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                status="success",
                additional_data={
                    "sql_generated": True,
                    "sql_length": len(sql)
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(total_input_tokens, total_output_tokens, "sql_generation", cache_read_tokens, cache_write_tokens)

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
                cache_read_tokens=0,
                cache_write_tokens=0,
                status="error",
                additional_data={
                    "error": str(e)
                }
            )
            
            return None

    # Keywords that indicate explicit chart request
    CHART_KEYWORDS = [
        'chart', 'graph', 'plot', 'visualize', 'visualization', 
        'draw', 'display as graph', 'show me a chart', 'create a graph',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter'
    ]

    def _has_explicit_chart_request(self, user_query: str) -> bool:
        """
        Check if the user query explicitly requests a chart/visualization.
        
        Args:
            user_query: The user's original query
            
        Returns:
            True if user explicitly requested a chart, False otherwise
        """
        query_lower = user_query.lower()

        visual_requested = any(keyword in query_lower for keyword in self.CHART_KEYWORDS)

        # Pre-filter: Only generate charts for explicit requests
        if not visual_requested:
            log_info(self.logger, "SQLAgent._has_explicit_chart_request", 
                    f"Skipping chart generation - no explicit chart request in query: '{user_query[:200]}...'")
            
        return visual_requested
        

    
    def process_query(self, user_query: str, user_email: str) -> Dict[str, Any]:
        """Process user query through the agent pipeline with semantic caching."""
        
        log_info(self.logger, "SQLAgent.process_query", 
                f"Starting function, User Query:{user_query}, for tenant_id:{self.tenant_id}")
        
        import time
        start_time = time.time()
        
        try:

            visual_requested = self._has_explicit_chart_request(user_query)

            # Step 0: Check Tier 1 cache (high similarity, no SQL validation)
            cached_result = self.cache.check_cache(user_query, self.tenant_id, generated_sql=None, visual_requested = visual_requested)
            
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
                    cache_read_tokens=0,
                    cache_write_tokens=0,
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
                    "cache_tier": cached_result.get('tier', 1),
                    "cache_hit_similarity_score": cached_result.get('similarity_score', 0)
                }
            
            log_info(self.logger, "SQLAgent.process_query", "Tier 1 cache miss - generating SQL")
            
            # Step 1: Generate SQL from user query
            sql = self._generate_sql_from_query(user_query, user_email)
            
            if not sql:
                # No SQL generated - query cannot be answered or needs clarification
                return {
                    "type": "structured",
                    "status": "no_sql",
                    "summary": "Query cannot be answered with available tables",
                    "sql": "",
                    "results": [],
                    "row_count": 0,
                    "chart_url": "",
                    "cache_hit": False
                }
            
            # Step 1.5: Check Tier 2 cache with SQL validation
            tier2_cached = self.cache.check_cache(user_query, self.tenant_id, generated_sql=sql, visual_requested = visual_requested)
            
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
                    cache_read_tokens=0,
                    cache_write_tokens=0,
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
                    "cache_tier": 2,
                    "cache_hit_similarity_score": tier2_cached.get('similarity_score', 0)
                }
            
            log_info(self.logger, "SQLAgent.process_query", "Tier 2 cache miss - executing query")
            
            # Step 2: Execute SQL
            execution = self.executor_tool.execute(sql)
            
            log_info(self.logger, "SQLAgent.process_query", 
                    f"Step 2: Executed SQL, Row Count: {execution['row_count']}")

            # Step 2.5: Limit the SQL result going forward to a max of 100 rows for memory & optimization reasons
            if execution and "results" in execution:
                limited_sql_results = execution["results"][:100]
            
            chart_url = None # Initialize
            # Step 3: Generate chart visualization if asked
            if visual_requested:
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
                    visual_requested,
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
                cache_read_tokens=0,
                cache_write_tokens=0,
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
                cache_read_tokens=0,
                cache_write_tokens=0,
                status="error",
                additional_data={
                    "cache_hit": False,
                    "error": str(e)
                }
            )
            
            raise Exception(f"Failed to query SQL database. Error Info: {str(e)}")


class SQLAgentOptimized(SQLAgent):
    """Optimized SQL Agent with compressed schema and basic prompt for improved performance."""
    
    # Override the base class functionality
    def _get_schema_summary(self) -> str:
        """Get an optimized, compacted summary of available tables using OptimizeTableSchema.
        
        This method overrides the parent class to provide a compressed schema representation
        that reduces token usage while maintaining all essential information for SQL generation.
        """
        optimized_schemas = []
        for table, stmt in self.create_statements.items():
            try:
                # Use OptimizeTableSchema with compression for maximum compaction
                optimized_stmt = SQLAgentOptimized._optimizeTableSchema(stmt, self.logger, compress=True)
                optimized_schemas.append(f"-- Table: {table}\n{optimized_stmt}")
            except Exception as e:
                self.logger.warning(f"Failed to optimize schema for table {table}: {str(e)}")
                # Fallback to original statement if optimization fails
                optimized_schemas.append(f"-- Table: {table}\n{stmt}")
        
        return "\n\n".join(optimized_schemas)
    
    # Override the base class functionality
    def _build_system_prompt(self) -> str:
        """Build a basic, optimized system prompt using compressed schema context.
        
        This method overrides the parent class to provide a more concise prompt
        that focuses on essential SQL generation rules while using the optimized schema.
        """
        # This should use the optimized schema from _get_schema_summary() and provide
        # a streamlined set of instructions focused on core SQL generation requirements
        """Build a basic system prompt using optimized schema context."""
        optimized_schema_context = self._get_schema_summary()  # Use optimized version

        return f"""
        You are an expert SQL query generator for PostgreSQL databases. Convert natural language questions into precise SQL queries.

        DATABASE SCHEMA:
        {optimized_schema_context}

        RULES:
        1. Generate SELECT queries ONLY
        2. ALL table references MUST include schema prefix: {self.tenant_id}.table_name
        3. Use proper JOIN conditions with matching data types (ID to ID)
        4. Use single quotes for string literals
        5. Ensure all non-aggregated columns in SELECT appear in the GROUP BY clause
        6. For order status filtering, use ONLY these exact values: 'delivered', 'processing', or 'shipped'
        """

    ## ======= Helper Static functions for to optimize the generated sql ==================
    @staticmethod
    def _optimizeTableSchema(create_statement: str, logger: logging.Logger, compress: bool = False) -> str:
        """
        Optimizes a CREATE TABLE statement for LLM SQL generation by:
        1. Removing non-essential constraints (NOT NULL, DEFAULT)
        2. Converting to "Schema on a String" format
        3. Optionally compressing the output for maximum brevity
        
        Args:
            create_statement (str): The original CREATE TABLE statement
            logger (logging.Logger): Logger reference for debugging
            compress (bool): Whether to apply additional compression
            
        Returns:
            str: Optimized schema in "Schema on a String" format
        """
        try:
            logger.info(f"Starting schema optimization for SQL: {create_statement}")
            
            # Step 1: Remove non-essential constraints
            optimized_statement = SQLAgentOptimized._remove_non_essential_constraints(create_statement, logger)
            
            # Step 2: Convert to "Schema on a String" format
            schema_string = SQLAgentOptimized._convert_to_schema_string(optimized_statement, logger)
            
            # Step 3: Apply compression if requested
            if compress:
                schema_string = SQLAgentOptimized._compress_schema_string(schema_string, logger)
            
            logger.info(f"Schema optimization completed successfully, optimized SQL: {schema_string}")
            return schema_string
            
        except Exception as e:
            logger.error(f"Error optimizing schema: {str(e)}")
            return create_statement  # Return original if optimization fails

    @staticmethod
    def _remove_non_essential_constraints(create_statement: str, logger: logging.Logger) -> str:
        """
        Remove NOT NULL and DEFAULT constraints from CREATE TABLE statement.
        Optimized version with single-pass regex and compiled patterns.
        """
        logger.debug("Removing non-essential constraints")
        
        # Remove NOT NULL constraints first
        statement = re.sub(r'\s+NOT\s+NULL', '', create_statement, flags=re.IGNORECASE)
        
        # Remove DEFAULT constraints with different patterns
        # 1. DEFAULT with function calls like nextval('seq'::regclass)
        statement = re.sub(r'\s+DEFAULT\s+[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)(?:::[a-zA-Z_][a-zA-Z0-9_]*)?(?=\s*[,)])', '', statement, flags=re.IGNORECASE)
        
        # 2. DEFAULT with quoted strings
        statement = re.sub(r"\s+DEFAULT\s+'[^']*'(?=\s*[,)])", '', statement, flags=re.IGNORECASE)
        statement = re.sub(r'\s+DEFAULT\s+"[^"]*"(?=\s*[,)])', '', statement, flags=re.IGNORECASE)
        
        # 3. DEFAULT with numbers (including decimals)
        statement = re.sub(r'\s+DEFAULT\s+\d+(?:\.\d+)?(?=\s*[,)])', '', statement, flags=re.IGNORECASE)
        
        # 4. DEFAULT with keywords like CURRENT_TIMESTAMP
        statement = re.sub(r'\s+DEFAULT\s+[A-Z_]+(?=\s*[,)])', '', statement, flags=re.IGNORECASE)
        
        # Single cleanup pass for whitespace and commas
        statement = re.sub(r'\s+', ' ', statement)  # Normalize whitespace
        statement = re.sub(r'\s*,\s*', ', ', statement)  # Normalize comma spacing
        statement = re.sub(r',\s*,', ',', statement)  # Remove double commas
        statement = re.sub(r',\s*\)', ')', statement)  # Remove trailing commas
        
        return statement.strip()

    @staticmethod
    def _convert_to_schema_string(create_statement: str, logger: logging.Logger) -> str:
        """
        Convert CREATE TABLE statement to "Schema on a String" format.
        Optimized version with compiled regex patterns and single-pass processing.
        """
        logger.debug("Converting to schema string format")
        
        # Pre-compiled regex patterns for better performance
        table_pattern = re.compile(r'CREATE\s+TABLE\s+([^\s\(]+)', re.IGNORECASE)
        paren_pattern = re.compile(r'\((.*)\)', re.DOTALL)
        pk_pattern = re.compile(r'PRIMARY\s+KEY\s*\(\s*([^)]+)\s*\)', re.IGNORECASE)
        constraint_pattern = re.compile(r'^(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|CONSTRAINT)', re.IGNORECASE)
        
        # Extract table name
        table_match = table_pattern.search(create_statement)
        if not table_match:
            logger.warning("Could not extract table name")
            return create_statement
        
        table_name = table_match.group(1)
        
        # Extract table content
        paren_match = paren_pattern.search(create_statement)
        if not paren_match:
            logger.warning("Could not extract table definition")
            return create_statement
        
        table_content = paren_match.group(1)
        
        # Split by commas, respecting nested parentheses
        parts = SQLAgentOptimized._split_table_definition(table_content)
        
        # Single-pass processing: identify constraints and columns simultaneously
        columns = []
        constraints = []
        primary_key_columns = set()
        
        # First, collect all PRIMARY KEY columns
        for part in parts:
            part = part.strip()
            if part:
                pk_match = pk_pattern.search(part)
                if pk_match:
                    pk_cols = [col.strip() for col in pk_match.group(1).split(',')]
                    primary_key_columns.update(pk_cols)
        
        # Second pass: categorize and format
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if constraint_pattern.match(part):
                constraints.append(part)
            else:
                # This is a column definition
                formatted_col = SQLAgentOptimized._format_column_definition(part)
                if formatted_col:
                    columns.append(formatted_col)
        
        # Build final schema string efficiently
        all_parts = columns + constraints
        schema_string = f"{table_name}({', '.join(all_parts)})"
        
        return schema_string

    @staticmethod
    def _split_table_definition(content: str) -> list:
        """
        Split table definition by commas, respecting nested parentheses.
        Optimized version with reduced string operations.
        """
        if not content:
            return []
        
        parts = []
        current_part = []
        paren_depth = 0
        
        for char in content:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                part = ''.join(current_part).strip()
                if part:
                    parts.append(part)
                current_part = []
                continue
                
            current_part.append(char)
        
        # Add the last part
        part = ''.join(current_part).strip()
        if part:
            parts.append(part)
        
        return parts

    @staticmethod
    def _compress_schema_string(schema_string: str, logger: logging.Logger) -> str:
        """
        Apply additional compression to the schema string while preserving meaning.
        Optimized version with compiled regex patterns and single-pass replacements.
        """
        logger.debug("Applying compression to schema string")
        
        # Pre-compiled regex patterns for better performance
        type_patterns = [
            (re.compile(r'\binteger\b', re.IGNORECASE), 'int'),
            (re.compile(r'\bvarchar\b', re.IGNORECASE), 'str'),
            (re.compile(r'\btimestamp\b', re.IGNORECASE), 'ts'),
            (re.compile(r'\bdecimal\b', re.IGNORECASE), 'dec'),
            (re.compile(r'\bnumeric\b', re.IGNORECASE), 'num'),
            (re.compile(r'\bboolean\b', re.IGNORECASE), 'bool'),
            (re.compile(r'\bcharacter\b', re.IGNORECASE), 'char')
        ]
        
        constraint_patterns = [
            (re.compile(r'\bPRIMARY KEY\b', re.IGNORECASE), 'PK'),
            (re.compile(r'\bFOREIGN KEY\b', re.IGNORECASE), 'FK'),
            (re.compile(r'\bREFERENCES\b', re.IGNORECASE), 'REF'),
            (re.compile(r'\bUNIQUE\b', re.IGNORECASE), 'UQ'),
            (re.compile(r'\bCHECK\b', re.IGNORECASE), 'CK')
        ]
        
        # Apply all type abbreviations
        compressed = schema_string
        for pattern, replacement in type_patterns:
            compressed = pattern.sub(replacement, compressed)
        
        # Apply all constraint abbreviations
        for pattern, replacement in constraint_patterns:
            compressed = pattern.sub(replacement, compressed)
        
        # Single-pass spacing optimization
        spacing_patterns = [
            (re.compile(r'\s*\(\s*'), '('),
            (re.compile(r'\s*\)\s*'), ')'),
            (re.compile(r'\s*,\s*'), ','),
            (re.compile(r'\s*:\s*'), ':'),
            (re.compile(r'(FK|PK|UQ|CK)\s+\('), r'\1(')
        ]
        
        for pattern, replacement in spacing_patterns:
            compressed = pattern.sub(replacement, compressed)
        
        return compressed

    @staticmethod
    def _format_column_definition(column_def: str) -> str:
        """
        Format a column definition for the schema string.
        Optimized version with compiled regex pattern.
        """
        # Pre-compiled regex for better performance
        column_pattern = re.compile(r'^(\w+)\s+(\w+(?:\([^)]+\))?)')
        
        # Normalize whitespace once
        column_def = re.sub(r'\s+', ' ', column_def.strip())
        
        # Extract column name and data type
        match = column_pattern.match(column_def)
        if match:
            return f"{match.group(1)}: {match.group(2)}"
        
        # Fallback for edge cases
        parts = column_def.split(None, 1)  # Split on first whitespace only
        if len(parts) >= 2:
            return f"{parts[0]}: {parts[1]}"
        
        return column_def
