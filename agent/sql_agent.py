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

from strands.types.content import SystemContentBlock

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
        self.model_id = model_id
        self.region = aws_region
        self.logger = logger
        log_info(logger, "SQLAgent.Init", f"Starting function, Database:{db_config["database"]}, Region:{aws_region}, Model ID:{model_id}")

        # Step 1: Extract CREATE TABLE statements during initialization
        schema_extractor = SchemaExtractor(db_config)
        self.create_statements = schema_extractor.extract_schema()
        log_info(logger, "SQLAgent.Init", f"Step 1: Extracted schema for {len(self.create_statements)} tables")

        # Step 2: Initialize SQL executor tool
        self.executor_tool = SQLExecutorTool(db_config)
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
        
        # Step 5: Create Strands agent with Bedrock model
        self.agent = Agent(
            name="SQL Query Agent",
            system_prompt=[ # Define system prompt with cache points
                SystemContentBlock(
                    text=system_prompt
                ),
                SystemContentBlock(cachePoint={"type": "default"})
            ],
            model=self.model
        )
        log_info(logger, "SQLAgent.Init", f"Step 5: Created Strands agent with Bedrock model")
        log_info(logger, "SQLAgent.Init", f"Ending function")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the SQL agent with database schema context."""
        
        # Prepare schema context from extracted CREATE TABLE statements
        schema_context = "\n\n".join([
            f"-- Table: {table}\n{stmt}" 
            for table, stmt in self.create_statements.items()
        ])
        
        return f"""
            # SQL Generation Agent

            <task_description>
            You are an advanced SQL query assistant specializing in PostgreSQL for AWS RDS environments. Your role is to translate natural language data requests into optimized SQL queries based on the provided database schema.
            </task_description>

            <database_context>
            Database Schema:
            {schema_context}
            </database_context>

            <instructions>
            When generating SQL queries, follow this systematic approach:

            1. **Analyze Request**: Carefully examine the user's data request to identify the core information needs and requirements.

            2. **Schema Validation**: Verify that the provided database schema contains all necessary tables and fields to fulfill the request. If the schema is insufficient to answer the query, indicate this limitation rather than creating an inaccurate query.

            3. **Query Decomposition**: Break complex requests into logical sub-queries when appropriate, ensuring each component addresses a specific part of the overall request.

            4. **SQL Construction**: 
            - Create precise SELECT statements that retrieve exactly the data requested
            - Utilize appropriate JOINs based on foreign key relationships
            - Implement efficient filtering conditions in WHERE clauses
            - Apply proper aggregation functions and GROUP BY clauses when needed
            - Include ORDER BY for meaningful result sequencing when appropriate
            - Consider LIMIT clauses to prevent excessive result sets

            5. **Query Optimization**: Ensure all queries are optimized for performance in a PostgreSQL environment, following AWS RDS best practices.
            </instructions>

            <constraints>
            - Generate SELECT statements ONLY (no data modification queries like INSERT, UPDATE, DELETE)
            - Respect the exact table and column names as defined in the schema
            - For order status filtering, use the exact values present in the database: 'delivered', 'processing', or 'shipped'
            - Include proper table aliases when joining multiple tables
            - Ensure all column references are fully qualified with table names/aliases when multiple tables are involved
            - Add comments to explain complex query logic when necessary
            </constraints>

            """
    
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
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the agent pipeline."""
        
        log_info(self.logger, "SQLAgent.process_query", f"Starting function, User Query:{user_query}")

        # Define the prompt, remember the agent has already been initialized and knows the Table Schema
        prompt = f"""
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

        3. **Response Format**:
        When providing a SQL query, format it in a code block as shown:
        ```sql
        SELECT ...
        ```
        </instructions>
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
            <task_description>
            You are a data analyst tasked with interpreting SQL query results and extracting meaningful insights. Your expertise in data analysis will help transform raw query results into actionable business intelligence.
            </task_description>

            <context>
            SQL Query: 
            {sql}

            Results (first {MAX_ROWS_FOR_SQL_RESULT_DISPLAY} rows): 
            {json.dumps(results_preview, indent=2, default=str)}

            Total Rows: {execution['row_count']}
            </context>

            <instructions>
            Please analyze the SQL query results above and provide a concise, insightful summary of the data. Your analysis should be clear, data-driven, and immediately useful for business decision-making.

            Structure your response to include:

            1. **Key Findings**: Identify and explain the most significant patterns or insights revealed by the data
            
            2. **Statistical Highlights**: Mention notable metrics, averages, outliers, or distributions that stand out

            3. **Trend Analysis**: Describe any temporal patterns, growth trends, or cyclical behaviors if applicable
            
            4. **Comparative Insights**: Highlight meaningful differences or similarities between categories, segments, or time periods
            
            5. **Business Implications**: Briefly suggest what these findings might mean for business decisions or strategy

            Keep your analysis concise, focused, and directly supported by the data presented. Avoid speculation beyond what the data clearly indicates.
            </instructions>
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
