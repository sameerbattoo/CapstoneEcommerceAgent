"""AWS Strands SQL Agent for RDS query validation and execution."""

import json
import re
import os
from typing import Any, Dict, List, Optional
import psycopg2
import boto3
import logging
from datetime import datetime

# Import Strands SDK
from strands import Agent
from strands.models.bedrock import BedrockModel

# Define constants
MAX_ROWS_FOR_SQL_RESULT_DISPLAY = 20
MAX_SAMPLE_QUESTIONS = 5

def log_info(logger: logging.Logger, function: str, content: str, log_complete_content: bool = False) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not log_complete_content:
        logger.info(f"[{timestamp}] {function}: {content}..." if len(content) > 200 else f"[{timestamp}] {function}: {content}")
    else:
        logger.info(f"[{timestamp}] {function}: {content}...")

class SchemaExtractor:
    """Extracts CREATE TABLE statements from RDS database."""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.create_statements = {}
    
    def extract_schema(self) -> Dict[str, str]:

        """Extract CREATE TABLE statements from database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get all tables
            sql = """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """
            cursor.execute(sql)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Generate CREATE TABLE statement for each table
            for table in tables:
                create_stmt = self._generate_create_statement(cursor, table)
                self.create_statements[table] = create_stmt
            
            cursor.close()
            conn.close()
            
            return self.create_statements
        
        except Exception as e:
            raise Exception(f"Failed to extract schema: {str(e)}")
    
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
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = cursor.fetchall()
            
            # Get primary key
            cursor.execute("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass AND i.indisprimary
            """, (table_name,))
            
            pk_columns = [row[0] for row in cursor.fetchall()]
            
            # Get foreign keys
            cursor.execute("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s
            """, (table_name,))
            
            fk_info = cursor.fetchall()
            
            # Build CREATE TABLE statement
            create_stmt = f"CREATE TABLE {table_name} (\n"
            
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
                create_stmt += f",\n    FOREIGN KEY ({fk_col}) REFERENCES {fk_table}({fk_ref_col})"
            
            create_stmt += "\n);"
            
            return create_stmt

        except Exception as e:
            raise Exception(f"Failed to create table: {table_name}. Error Info:{str(e)}")

class SQLExecutorTool:
    """Executes SQL queries against RDS database."""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.name = "execute_sql"
        self.description = "Executes SQL query against the RDS database and returns results"
    
    def execute(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query safely."""

        # Basic SQL injection prevention
        if any(keyword in sql.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']):
            raise Exception("Only SELECT queries are allowed to be executed")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute(sql)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
            else:
                results = []
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results)
            }
        except Exception as e:
            raise Exception(f"Failed to execute SQL: {sql}. Error Info: {str(e)}")


class SQLAgent:
    """Main SQL Agent using AWS Strands with Bedrock."""
    
    def __init__(self,logger: logging.Logger, db_config: Dict[str, str], aws_region: str, model_id: str):

        # Get AWS region & other config value from environment or parameter
        self.db_config = db_config
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID")
        self.region = aws_region or os.getenv("AWS_REGION")
        self.logger = logger
        log_info(logger, "SQLAgent.Init", f"Starting function, Database:{db_config["database"]}, Region:{aws_region}, Model ID:{model_id}")

        # Step 1: Extract CREATE TABLE statements during initialization
        schema_extractor = SchemaExtractor(db_config)
        self.create_statements = schema_extractor.extract_schema()
        log_info(logger, "SQLAgent.Init", f"Step 1: Extracted schema for {len(self.create_statements)} tables")

        # Prepare schema context
        schema_context = "\n\n".join([
            f"-- Table: {table}\n{stmt}" 
            for table, stmt in self.create_statements.items()
        ])
        
        # Step 2: Initialize SQL executor tool
        self.executor_tool = SQLExecutorTool(db_config)
        log_info(logger, "SQLAgent.Init", f"Step 2: SQLExecutorTool configured")
               
        # Step 3: Initialize Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            region_name=self.region
        )
        log_info(logger, "SQLAgent.Init", f"Step 3: Bedrock model initialized")
        
        # Step 4: Create Strands agent with Bedrock model
        self.agent = Agent(
            name="SQL Query Agent",
            system_prompt=f"""You are a SQL query assistant with access to database CREATE TABLE statements.

You are a SQL developer creating queries that will be run against AWS RDS for PostgreSQL.
Here is how I want you to think step by step:
1. Accept user queries about data
2. Analyze the user request to understand the main objective. - Break down reqeusts into sub-queries that can each address a part of the user request, using the provided CREATE TABLE statements
3. For each sub-query, use the relevant tables and fields from the provided schema. - Construct SQL queries that are precise and tailored to retrieve the exact data required by the user query
4. Validate if the database schema (provided as CREATE TABLE statements) can answer the query, if not, please don't create the SQL
5. Generate accurate SQL SELECT statements for valid queries using the CREATE TABLE definitions

Important guidelines:
- Use the CREATE TABLE statements to understand table structure, columns, data types, and relationships
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
- Consider foreign key relationships when joining tables
- Always optimize SQL queries for performance and clarity.
- Keep in mind When constructing the WHERE clause for order.status, the values in the table are - delivered, processing, shipped.

Database Schema (CREATE TABLE statements):
{schema_context}

""",
            model=self.model
        )
        log_info(logger, "SQLAgent.Init", f"Step 4: Created Strands agent with Bedrock model")
        log_info(logger, "SQLAgent.Init", f"Ending function")

    def get_schema_summary(self) -> str:
        """Get a summary of available tables."""
        summary = "Available Tables:\n"
        for table_name in self.create_statements.keys():
            summary += f"  - {table_name}\n"
        return summary

    def get_sample_questions(self, no_of_questions: int = MAX_SAMPLE_QUESTIONS) -> List[str]:
        """Get a list of sample questions that can be answered from the configured database."""

        log_info(self.logger, "SQLAgent.get_sample_questions", f"Starting function")
        # Use agent to generate the sample questions
        prompt = f"""
User Query: List {no_of_questions} user questions that can be answered by the database tables.

Format your response as a JSON array of strings (no extra text) like:
```json
[]
```
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
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the agent pipeline."""
        
        log_info(self.logger, "SQLAgent.process_query", f"Starting function, User Query:{user_query}")

        # Define the prompt, remember the agent has already been initialized and knows the Table Schema
        prompt = f"""
User Query: {user_query}

Please:
1. Analyze if this query can be answered with the available tables
2. If YES: Generate the appropriate SQL SELECT query
3. If NO: Don't generate any SQL query

Format your SQL query in a code block like:
```sql
SELECT ...
```
""" 
        try:
           # Step 1: Use the agent to get the SQL for the user query
            response = self.agent(prompt)
            
            # Extract SQL from response - handle different response formats
            response_text = response.output if hasattr(response, 'output') else str(response)
            sql_match = re.search(r'```sql\n(.*?)\n```', response_text, re.DOTALL)
            
            if not sql_match:
                log_info(self.logger, "SQLAgent.process_query", f"No SQL was generated for user query, Reasons:{response_text}")
                # No SQL generated - query cannot be answered or needs clarification
                return {
                    "type": "structured",
                    "status": "no_sql",
                    "summary": response_text,
                    "sql": "",
                    "results": [],
                    "row_count": 0
                }
            
            sql = sql_match.group(1).strip()
            log_info(self.logger, "SQLAgent.process_query", f"Step 1:Generated the SQL based on user query, SQL:{sql}", True)
            
            # Step 2: Execute SQL
            execution = self.executor_tool.execute(sql)
            log_info(self.logger, "SQLAgent.process_query", f"Step 2:Executed the SQL, Row Count:{str(execution['row_count'])}")
                       
            # Step 3: Summarize results
            results_preview = execution["results"][:MAX_ROWS_FOR_SQL_RESULT_DISPLAY] if execution["results"] else []
            
            summary_prompt = f"""
SQL Query: {sql}
Results (first {MAX_ROWS_FOR_SQL_RESULT_DISPLAY}] rows): {json.dumps(results_preview, indent=2, default=str)}
Total Rows: {execution['row_count']}

Please provide a concise summary of the key insights from this data.
Focus on:
- Main findings and patterns
- Notable statistics or trends
- Any interesting observations
"""
            
            summary = self.agent(summary_prompt)
            summary_text = summary.output if hasattr(summary, 'output') else str(summary)
            log_info(self.logger, "SQLAgent.process_query", f"Step 3:Generated the key summary.")
            
            return {
                "type": "structured",
                "status": "success",
                "sql": sql,
                "results": execution["results"],
                "row_count": execution["row_count"],
                "summary": summary_text,
            }

        except Exception as e:
            log_info(self.logger, "SQLAgent.process_query", f"Failed to query SQL database. Error Info: {str(e)}")
            raise Exception(f"Failed to query SQL database. Error Info: {str(e)}")