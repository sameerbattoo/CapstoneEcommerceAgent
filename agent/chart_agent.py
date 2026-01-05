"""AWS Chart Agent for generating visualizations using Code Interpreter."""

import logging
from typing import Any, Dict, Optional, List
import boto3
import json
import time
import re
from datetime import datetime

# Import Code Interpreter tool
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

# Strands Agent imports
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.types.content import SystemContentBlock

def log_info(logger: logging.Logger, function: str, content: str) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {function}: {content[:500]}..." if len(content) > 500 else f"[{timestamp}] {function}: {content}")

class ChartAgent:
    """Generates charts and visualizations using AWS Code Interpreter."""
    
    def __init__(
        self, 
        logger: logging.Logger, 
        aws_region: str, 
        model_id: str, 
        tenant_id: str,
        chart_s3_bucket: str, 
        cloudfront_domain: Optional[str] = None,
        session_id: str = "unknown",
        token_callback: Optional[callable] = None
    ):
        """
        Initialize Chart Agent.
        
        Args:
            logger: Logger instance
            aws_region: AWS region for Code Interpreter and S3
            chart_s3_bucket: S3 bucket name for storing generated charts
            cloudfront_domain: Optional CloudFront domain for chart URLs (e.g., "d1234567890abc.cloudfront.net")
        """
        self.logger = logger
        self.name = "chart_agent"
        self.description = "Generates visual charts and graphs from data using Python code"
        self.code_interpreter = CodeInterpreter(region=aws_region)
        self.aws_region = aws_region
        self.model_id = model_id
        self.tenant_id = tenant_id # Store tenant_id for metrics
        self.session_id = session_id  # Store session_id for metrics
        self.token_callback = token_callback  # Store callback for token accumulation
        self.chart_s3_bucket = chart_s3_bucket
        self.cloudfront_domain = cloudfront_domain
        self.s3_client = boto3.client('s3', region_name=aws_region)

        # Step 1: Initialize Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            region_name=self.aws_region
        )
        log_info(self.logger, "ChartAgent.Init", "Step 1: Bedrock model initialized")
        
        # Step 2: Build system prompt
        system_prompt = self._build_system_prompt()
        log_info(self.logger, "ChartAgent.Init", "Step 2: Built system prompt")

        # Step 3: Create Strands agent with Bedrock model and tools
        self.agent = Agent(
            name="Chart Agent",
            system_prompt=[ # Define system prompt with cache points
                SystemContentBlock(
                    text=system_prompt
                ),
                SystemContentBlock(cachePoint={"type": "default"})
            ],
            model=self.model,
        )

        log_info(self.logger, "ChartAgent.Init", f"Step 3: Created Strands agent with Bedrock model and Code Interpreter tool")
        log_info(self.logger, "ChartAgent.Init", f"Ending function with region={aws_region}, bucket={chart_s3_bucket}, cloudfront={cloudfront_domain}")
           
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for chart generation with instructions and constraints.
        
        Returns:
            Complete system prompt string for the chart agent
        """
        return """
            # Chart Agent

            <task_description>
            **Data Visualization Code Generation**: You will generate the visualization code, which will be executed separately in a secure Python environment.
            When query results contain numeric data suitable for visualization, generate Python code using matplotlib to create professional charts. 
            </task_description>

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
            from datetime import datetime

            # Load data directly as Python literal (data is provided pre-escaped)
            # DO NOT use json.loads() - this avoids JSON escaping issues with quotes
            data = [PYTHON_DATA_HERE]
            df = pd.DataFrame(data)

            # [YOUR VISUALIZATION CODE HERE]

            # Save chart - CRITICAL: Use EXACTLY this path, do NOT use /tmp/ or any other directory!
            output_path = 'chart.png'
            plt.savefig(output_path, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved to '{{output_path}}'")
            ```

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

            ## Styling Requirements - Modern Dark Theme with Rounded Borders:
            
            **CRITICAL: Apply these styling rules to ALL charts for modern dark appearance:**
            
            ```python
            # 1. Define vibrant color palette for dark backgrounds
            COLORS = ['#5DADE2', '#EC7063', '#58D68D', '#F39C12', '#AF7AC5', '#48C9B0', '#F8B739', '#E74C3C']
            
            # 2. Figure setup with dark background and rounded corners
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a2332')
            ax.set_facecolor('#1a2332')  # Dark navy background for plot area
            
            # 3. For line charts - use vibrant colors with glow effect:
            ax.plot(x, y, color=COLORS[0], linewidth=2.5, marker='o', markersize=6, alpha=0.9)
            
            # 4. For bar charts - use vibrant colors:
            ax.bar(x, y, color=COLORS[0], alpha=0.85, edgecolor='#2c3e50', linewidth=1.5)
            
            # 5. Grid styling - subtle on dark background
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8, color='#7f8c8d')
            ax.set_axisbelow(True)  # Grid behind bars/lines
            
            # 6. Spine (border) styling - ROUNDED RAISED EFFECT
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#34495e')
                spine.set_linewidth(2)
            
            # 7. Labels and title styling - light colors for dark background
            ax.set_xlabel('X Label', fontsize=12, fontweight='600', color='#ecf0f1', labelpad=10)
            ax.set_ylabel('Y Label', fontsize=12, fontweight='600', color='#ecf0f1', labelpad=10)
            ax.set_title('Chart Title', fontsize=14, fontweight='bold', color='#ffffff', pad=20)
            
            # 8. Tick styling - light colors for visibility
            ax.tick_params(axis='both', labelsize=10, colors='#bdc3c7', length=6, width=1.5)
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if needed
            
            # 9. Legend styling - dark theme with rounded corners
            if ax.get_legend_handles_labels()[0]:  # Only if legend exists
                legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
                         framealpha=0.95, fontsize=10, edgecolor='#34495e', facecolor='#2c3e50')
                for text in legend.get_texts():
                    text.set_color('#ecf0f1')
            
            # 10. Add value labels on bars/points (optional, light colored)
            for i, v in enumerate(values):
                ax.text(i, v, f'{{v:,.0f}}', ha='center', va='bottom', 
                       fontsize=9, fontweight='500', color='#ecf0f1')
            
            # 11. ROUNDED CORNERS EFFECT - Add padding and adjust layout
            plt.tight_layout(pad=2.0)
            
            # 12. Save with transparent edges to show rounded effect
            # The rounded border effect is achieved through the figure facecolor and spine styling
            ```
            
            **Color Palette for Dark Theme:**
            - COLORS[0] (#5DADE2 - Bright Blue) for primary data
            - COLORS[1] (#EC7063 - Coral Red) for secondary comparisons
            - COLORS[2] (#58D68D - Emerald Green) for positive trends
            - COLORS[3] (#F39C12 - Orange) for highlights
            - COLORS[4] (#AF7AC5 - Purple) for additional categories
            - For multiple categories, cycle through COLORS array
            
            **Dark Theme Color Guidelines:**
            - Background: #1a2332 (Dark navy)
            - Text/Labels: #ecf0f1 (Light gray)
            - Title: #ffffff (White)
            - Borders: #34495e (Medium gray)
            - Grid: #7f8c8d with low alpha (0.15)
            
            **CRITICAL - No Duplicate Parameters:**
            - NEVER specify the same parameter twice in a function call
            - Example WRONG: `ax.text(..., fontweight='500', color='#ecf0f1', fontweight='bold')`  # fontweight appears twice!
            - Example CORRECT: `ax.text(..., fontweight='500', color='#ecf0f1')`  # Each parameter appears once
            - If you need bold text, use `fontweight='bold'` OR `fontweight='600'`, not both
            
            **Font Guidelines:**
            - Title: 14pt, bold, #ffffff
            - Axis labels: 12pt, semibold (600), #ecf0f1
            - Tick labels: 10pt, #555555
            - Value labels: 9pt, medium (500), #333333

            </chart_generation_instructions>

            <critical_constraints>
            ## For Python Code Generation
            - **Visualization Code Quality**: 
            - Generate only valid, executable Python code
            - Use only standard libraries: pandas, matplotlib, numpy, json, base64, datetime
            - **File Path Requirement**: ALWAYS save charts to 'chart.png' (NOT '/tmp/chart.png' or any other path)
            - The output file MUST be saved as: output_path = 'chart.png'
            - Include all required imports
            - Handle data type conversions properly
            - Test logic mentally before generating

            - **Code Safety**:
            - No file system operations except saving to 'chart.png' (in the current working directory)
            - No network operations
            - No system calls or subprocess execution
            - No eval() or exec() usage

            - **Data Handling**:
            - Embed data as JSON string in the code
            - Handle missing values and edge cases
            - Validate data structure before visualization

            </critical_constraints>
            """
    
    def _generate_visual_from_code(self, code: str) -> Dict[str, Any]:
        """
        Generate chart from Python visualization code using Code Interpreter.
        
        Args:
            code: Python code that generates a chart and saves it as 'chart.png'
            
        Returns:
            Dictionary with:
                - success: bool - Whether chart generation succeeded
                - message: str - Success or error message
                - chart_url: str - URL to access the generated chart (S3 or CloudFront)
                - output: str - Any output from code execution
        """
        try:
            log_info(self.logger, "ChartAgent.generate_visual", f"Starting chart generation via Code Interpreter")
            
            # Start Code Interpreter session
            self.code_interpreter.start(session_timeout_seconds=1200)
            
            # Execute the chart generation code
            log_info(self.logger, "ChartAgent.generate_visual", f"[ChartAgent] Executing chart generation code")
            exec_response = self.code_interpreter.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": False
            })

            if exec_response is None:
                raise Exception("Code Interpreter returned None")

            # Consume the execution response and check for errors
            execution_error = None
            for event in exec_response["stream"]:
                # Check if there was an error during execution
                if isinstance(event, dict):
                    if 'error' in event:
                        execution_error = event['error']
                        self.logger.error(f"[ChartAgent] Code execution error: {execution_error}")
                    elif 'result' in event:
                        result = event['result']
                        if isinstance(result, dict) and 'error' in result:
                            execution_error = result['error']
                            self.logger.error(f"[ChartAgent] Code execution error in result: {execution_error}")
            
            if execution_error:
                raise Exception(f"Code execution failed: {execution_error}")
            
            log_info(self.logger, "ChartAgent.generate_visual", f"Code execution completed, retrieving chart file")
            
            # Use readFiles to retrieve the generated chart
            file_response = self.code_interpreter.invoke("readFiles", {
                "paths": ["chart.png"]
            })
            
            # Extract chart bytes from response with defensive checks
            chart_bytes = None
            for event in file_response["stream"]:
                log_info(self.logger, "ChartAgent.generate_visual", f"readFiles event type: {type(event)}, keys: {list(event.keys()) if isinstance(event, dict) else 'not a dict'}")
                
                if 'result' in event:
                    result = event['result']
                    log_info(self.logger, "ChartAgent.generate_visual", f"result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                    
                    # Defensive extraction: result['content'][0]['resource']['blob']
                    if 'content' in result and isinstance(result['content'], list) and len(result['content']) > 0:
                        content_item = result['content'][0]
                        log_info(self.logger, "ChartAgent.generate_visual",f"content[0] type: {type(content_item)}, keys: {list(content_item.keys()) if isinstance(content_item, dict) else 'not a dict'}")
                        
                        if isinstance(content_item, dict) and 'resource' in content_item:
                            resource = content_item['resource']
                            log_info(self.logger, "ChartAgent.generate_visual",f"resource type: {type(resource)}, keys: {list(resource.keys()) if isinstance(resource, dict) else 'not a dict'}")
                           
                            if isinstance(resource, dict) and 'blob' in resource:
                                chart_bytes = resource['blob']
                                log_info(self.logger, "ChartAgent.generate_visual",f"Successfully extracted blob, size: {len(chart_bytes)} bytes")
                                break
                        else:
                            log_info(self.logger, "ChartAgent.generate_visual",f"'resource' key not found in content[0]")
                    else:
                        log_info(self.logger, "ChartAgent.generate_visual",f"'content' not found or empty in result")
                        # Log the full result structure for debugging
                        log_info(self.logger, "ChartAgent.generate_visual",f"Full result structure: {json.dumps(result, default=str, indent=2)[:2000]}")
           
            if not chart_bytes:
                raise Exception("Failed to retrieve chart data from readFiles - check logs for response structure details")
            
            log_info(self.logger, "ChartAgent.generate_visual",f"Chart retrieved: {len(chart_bytes)} bytes")
            
            # Upload to S3 and get URL
            chart_url = self._upload_to_s3_from_bytes(chart_bytes)
            if chart_url:
                log_info(self.logger, "ChartAgent.generate_visual",f"Chart uploaded to S3 successfully")
            else:
                self.logger.warning(f"[ChartAgent] Failed to upload chart to S3")
            
            # Stop Code Interpreter session
            self.code_interpreter.stop()
            
            return {
                "success": True,
                "message": "Chart generated successfully",
                "chart_url": chart_url,
                "output": '',
            }
            
        except Exception as e:
            error_msg = f"Failed to generate chart: {str(e)}"
            self.logger.error(f"[ChartAgent] {error_msg}")
            import traceback
            self.logger.error(f"[ChartAgent] Traceback: {traceback.format_exc()}")
            
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
    
    def _upload_to_s3_from_bytes(self, chart_bytes: bytes) -> Optional[str]:
        """
        Upload chart bytes directly to S3 and return CloudFront or pre-signed URL.
        
        Args:
            chart_bytes: Binary image data
            
        Returns:
            CloudFront URL (if configured) or pre-signed S3 URL, None if upload fails
        """
        try:
            import uuid
            from datetime import datetime
            
            # Use bucket name from constructor
            bucket_name = self.chart_s3_bucket
            if not bucket_name:
                self.logger.error("[ChartAgent] chart_s3_bucket not provided")
                return None
            
            log_info(self.logger, "ChartAgent.upload_to_s3_from_bytes",f"Uploading {len(chart_bytes)} bytes to S3")
            
            # Generate unique S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            s3_key = f"charts/{timestamp}_{unique_id}.png"
            
            # Upload to S3
            log_info(self.logger, "ChartAgent.upload_to_s3_from_bytes",f"Uploading to s3://{bucket_name}/{s3_key}")
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
                log_info(self.logger, "ChartAgent.upload_to_s3_from_bytes",f"Generated CloudFront URL: {chart_url}")
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
                log_info(self.logger, "ChartAgent.upload_to_s3_from_bytes",f"Generated pre-signed URL: {chart_url[:100]}...")
            
            return chart_url
            
        except Exception as e:
            self.logger.error(f"[ChartAgent] Failed to upload to S3: {str(e)}")
            import traceback
            self.logger.error(f"[ChartAgent] Traceback: {traceback.format_exc()}")
            return None

    def _generate_visualization_code(self, user_query: str, sql: str, 
                                     results: List[Dict[str, Any]], row_count: int) -> Optional[str]:
        """Generate Python visualization code from query results.
        
        Returns:
            Python code string if successful, None if no code generated
        """
        results_preview = results[:100] if results else [] #Limit the data for charts to a max of 100 rows
        
        # Convert datetime objects to ISO format strings for safe serialization
        # This prevents issues with repr() generating datetime.date() calls
        import datetime as dt
        def convert_dates(obj):
            """Recursively convert datetime objects to ISO format strings."""
            if isinstance(obj, (dt.date, dt.datetime)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            else:
                return obj
        
        results_preview_converted = convert_dates(results_preview)
        
        # Convert results to Python literal string using repr() for safe embedding
        # This handles all special characters correctly including quotes
        results_repr = repr(results_preview_converted)
        
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
            - Results (first 100 rows): {results_repr}
            - Total Rows: {row_count}
            </context>
            
            <important_note>
            The results data provided in the context above is a Python literal representation.
            When embedding this data in your generated code, copy it EXACTLY from the context.
            Do NOT use json.loads() or any JSON parsing - the data is already in Python format.
            
            **Date/Time Handling:**
            - Date and datetime values are provided as ISO format strings (e.g., '2025-11-01', '2025-11-01T10:30:00')
            - Convert them to datetime objects using pd.to_datetime() after creating the DataFrame
            - Example:
            ```python
            data = [... copy from Results field above ...]
            df = pd.DataFrame(data)
            
            # Convert date columns to datetime
            if 'month' in df.columns:
                df['month'] = pd.to_datetime(df['month'])
            if 'order_date' in df.columns:
                df['order_date'] = pd.to_datetime(df['order_date'])
            ```
            
            Example structure:
            ```python
            # CORRECT - Copy the data from context above
            data = [... copy from Results field above ...]
            df = pd.DataFrame(data)
            
            # WRONG - Don't use JSON parsing
            # data_json = '''...'''
            # data = json.loads(data_json)
            ```
            </important_note>

            <response_format>
            Provide your response as ython code.
            No additional text, explanations, or formatting should be included outside the python code.
            When providing Python code, format it in a code block as shown:

            ```python
            import pandas as pd
            import matplotlib.pyplot as plt
            ...
            ```
            </response_format>
            """
        
        #Note the start time
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            #Execute the agent to get response
            code_response = self.agent(code_prompt)
            code_response_text = code_response.output if hasattr(code_response, 'output') else str(code_response)
            python_match = re.search(r'```python\n(.*?)\n```', code_response_text, re.DOTALL)
            
            #Note the end time along with the token usage
            end_time = time.time()
            processing_duration_in_secs =  abs(end_time - start_time)
            summary = code_response.metrics.get_summary()
            if summary and "accumulated_usage" in summary:
                total_input_tokens  = summary["accumulated_usage"].get("inputTokens",0)
                total_output_tokens = summary["accumulated_usage"].get("outputTokens",0)
           
            if not python_match:
                log_info(self.logger, "SQLAgent._generate_visualization_code", 
                        "No Python code generated by agent")
                
                # Emit metrics for no code generated
                self.logger.emit_step_metrics(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    step_name="chart_code_generation",
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    status="success",
                    additional_data={
                        "code_generated": False,
                        "row_count": row_count
                    }
                )
                
                # Report tokens to parent via callback
                if self.token_callback:
                    self.token_callback(total_input_tokens, total_output_tokens, "chart_code_generation")
                
                return None
            
            code = python_match.group(1).strip()
            
            # Emit metrics for successful code generation
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="chart_code_generation",
                start_time=start_time,
                end_time=end_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                status="success",
                additional_data={
                    "code_generated": True,
                    "code_length": len(code),
                    "row_count": row_count,
                    "results_preview_rows": len(results_preview)
                }
            )
            
            # Report tokens to parent via callback
            if self.token_callback:
                self.token_callback(total_input_tokens, total_output_tokens, "chart_code_generation")

            return code
            
        except Exception as e:
            end_time = time.time()
            self.logger.error(f"[SQLAgent._generate_visualization_code]: Error generating visualization code: {str(e)}")
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="chart_code_generation",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "error": str(e),
                    "row_count": row_count
                }
            )
            
            return None

    def generate_chart(self, user_query: str, sql: str, 
                       results: List[Dict[str, Any]], row_count: int) -> Optional[Dict[str, Any]]:
        """Generate chart visualization from query results.
        
        Returns:
            Chart result dictionary if successful, None otherwise
        """
        start_time = time.time()
        
        if not results or len(results) == 0:
            log_info(self.logger, "ChartAgent.generate_chart", "No results to visualize")
            
            end_time = time.time()
            # Emit metrics for no results from the sql statement execution
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="chart_generation",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="success",
                additional_data={
                    "chart_generated": False,
                    "reason": "no_results",
                    "row_count": row_count
                }
            )
            
            return None
        
        log_info(self.logger, "ChartAgent.generate_chart", 
                "Generating Python code for chart visualization...")
        
        try:
            # Generate visualization code
            code = self._generate_visualization_code(user_query, sql, results, row_count)
            if not code:
                return None
            
            # Execute code in Code Interpreter
            chart_result = self._generate_visual_from_code(code)
            
            end_time = time.time()
            
            if chart_result.get("success"):
                chart_url = chart_result.get('chart_url')
                log_info(self.logger, "ChartAgent.generate_chart", 
                        f"Chart generated successfully. Chart URL: {chart_url}")
                
                # Emit metrics for successful chart generation
                self.logger.emit_step_metrics(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    step_name="chart_generation",
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=0,
                    output_tokens=0,
                    status="success",
                    additional_data={
                        "chart_generated": True,
                        "chart_url": chart_url,
                        "row_count": row_count,
                        "code_length": len(code)
                    }
                )
            else:
                self.logger.error(f"[ChartAgent.generate_chart]: Chart generation failed: {chart_result.get('message')}")
                
                # Emit metrics for chart generation failure
                self.logger.emit_step_metrics(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    step_name="chart_generation",
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=0,
                    output_tokens=0,
                    status="error",
                    additional_data={
                        "chart_generated": False,
                        "reason": "execution_failed",
                        "error": chart_result.get('message'),
                        "row_count": row_count
                    }
                )
            
            return chart_result
            
        except Exception as e:
            end_time = time.time()
            self.logger.error(f"[ChartAgent.generate_chart]: Error in chart generation: {str(e)}")
            
            # Emit metrics for error
            self.logger.emit_step_metrics(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                step_name="chart_generation",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={
                    "error": str(e),
                    "row_count": row_count
                }
            )
            
            return None