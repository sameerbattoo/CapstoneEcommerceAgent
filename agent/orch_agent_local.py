"""
Interactive eCommerceAgent using Strands

This agent queries the Bedrock Knowledgebased for unstractured data
and a RDS Database for stractured data to answer user's queries 
in an interactive chat format with logging.
"""
# Standard library imports
import logging
from typing import Dict, Any, List, Optional
import time
import json
from datetime import datetime
import os
import base64

import pandas as pd
import plotly.express as px

from strands import Agent, tool
from strands_tools import current_time

# Import the 2 sub agents
from agent import sql_agent
from agent import kb_agent

# Import dotenv for loading environment variables
from dotenv import load_dotenv

#=====================================================================================
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("eCommerceAgent")

#=====================================================================================
# Generic Function to validate environment variables and return the value
def validate_and_set_env_variable(env_variable: str) -> str:
    env_variable_val = os.getenv(env_variable)
    
    if env_variable_val is None:
        logger.error(f"{env_variable} not found in environment variables.")
        raise ValueError(f"{env_variable} environment variable is required")

    return env_variable_val

#=====================================================================================
def log_conversation(role: str, content: str, tool_calls: Optional[List] = None) -> None:
    """Log each conversation turn with timestamp and optional tool calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {role}: {content[:100]}..." if len(content) > 100 else f"[{timestamp}] {role}: {content}")
    
    if tool_calls:
        for call in tool_calls:
            logger.info(f"  Tool used: {call['name']} with args: {json.dumps(call['args'])}")

#=====================================================================================
# Initialize the agent
#=====================================================================================
# Load environment variables from .env file
load_dotenv()

# Set - AWS Region, Model ID and ARN, KB ID from environment variables
AWS_REGION = validate_and_set_env_variable("AWS_REGION")
BEDROCK_MODEL_ID = validate_and_set_env_variable("BEDROCK_MODEL_ID")
BEDROCK_MODEL_ARN = validate_and_set_env_variable("BEDROCK_MODEL_ARN")
KB_ID = validate_and_set_env_variable("KB_ID")
DB_CONFIG = {
            "host": validate_and_set_env_variable("DB_HOST"),
            "port": validate_and_set_env_variable("DB_PORT"),
            "database": validate_and_set_env_variable("DB_NAME"),
            "user": validate_and_set_env_variable("DB_USER"),
            "password": validate_and_set_env_variable("DB_PASSWORD")
        }

# Create an Sub Agent client objects
sql_agent_client = sql_agent.SQLAgent(logger, DB_CONFIG,AWS_REGION, BEDROCK_MODEL_ID)
kb_agent_client = kb_agent.KnowledgeBaseAgent(logger, KB_ID, AWS_REGION, BEDROCK_MODEL_ARN)

#=====================================================================================
# Now defined the tools that the agent has access to
#=====================================================================================
@tool
def get_default_questions(user_query: str) -> Dict[str, Any]:
    """
    Lists sample of questions that the user can ask related to products and orders. These questions can be based on the un-strutured data from the Bedrock knowledge-base or based on the structured data from the RDS Postgres database.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """

    log_conversation("User", user_query)
    
    try:
        
        start_time = time.time()

        response = (
            "Questions related to Order and Products transactional data in natural language \n"
            "Examples:\n"
        )

        # Call the agent and return its response
        for question in sql_agent_client.get_sample_questions():
            response += f"  • {question}\n"

        response += (
            "\n\n, Questions related to Products based on the product specifications \n"
            "Examples:\n"
        )
        for question in kb_agent_client.get_sample_questions():
            response += f"  • {question}\n"
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the default questions assistant")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the assistant: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the assistant: {str(e)}. Please try again later."}}
#=====================================================================================


#=====================================================================================
# Worker Agent A: Data retrieval agent for Structured data sored in RDS
@tool
def get_answers_for_structured_data(user_query: str) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products and orders transactional data stored in the RDS Postgres database.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """

    log_conversation("User", user_query)

    try:
        start_time = time.time()
        # Call the agent and return its response
        response = sql_agent_client.process_query(user_query)
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the structured assistant")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the structured assistant: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the structured assistant: {str(e)}. Please try again later."}}
#=====================================================================================

#=====================================================================================
# Worker Agent B: Data retrieval agent for Un-Structured data sored in Bedrock KB
@tool
def get_answers_for_unstructured_data(user_query: str) -> Dict[str, Any]:
    """
    This tool can answer all questions related to products specifications based on the unstructured data from the Bedrock knowledge-base.

    Args:
        user_query: the user query based on product or orders.

    Returns:
        Detailed answers which is formatted to the user question
    """

    log_conversation("User", user_query)

    try:
        start_time = time.time()
        # Call the agent and return its response
        response = kb_agent_client.process_query(user_query)
        
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the unstructured assistant")

        return response

    except Exception as e:
        logger.error(f"Error processing request in the unstructured assistant: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the unstructured assistant: {str(e)}. Please try again later."}}
#=====================================================================================

#=====================================================================================
@tool
def show_state_fact_map_tool(
    rows: List[Dict],
    state_col: str = "shipping_state", # State Code
    value_col: str = "value", # Fact Value
    title: Optional[str] = "Statewise facts map",
    width: int = 900,
    height: int = 600,
) -> Dict:
    """
    Display state-wise fact data on a US map using Plotly Express and return a base64 PNG.

    Args:
        rows: List of dicts, each with a US state code (e.g., 'CA') and a numeric value.
              Example: [{"state_code": "CA", "value": 10}, {"state_code": "TX", "value": 20}]
        state_col: Key in each row containing the two-letter US state code.
        value_col: Key in each row containing the numeric fact/value to visualize.
        title: Title for the map.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        Dict with:
            - "format": "png"
            - "b64_data": base64-encoded PNG image (no data URI prefix)
    """

    start_time = time.time()
    try:
        if not rows:
            return {"error": "No data rows provided for map."}

        df = pd.DataFrame(rows)

        if state_col not in df.columns or value_col not in df.columns:
            return {"error": "No state-wise data column provided for map."}

        fig = px.choropleth(
            df,
            locations=state_col,
            locationmode="USA-states",
            color=value_col,
            hover_name=state_col,
            hover_data={value_col: True},
            color_continuous_scale="Viridis",
            scope="usa",
            title=title or "Statewise facts map",
            width=width,
            height=height,
        )

        # Requires: pip install "plotly[kaleido]"
        png_bytes = fig.to_image(format="png", width=width, height=height)  # uses Kaleido [web:52][web:56]

        b64_data = base64.b64encode(png_bytes).decode("utf-8")  # [web:58][web:67]

        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds in the mapping assistant")

        return {
            "format": "png",
            "b64_data": b64_data,
        }
    except Exception as e:
        logger.error(f"Error processing request in the convert to map tool: {str(e)}")
        # Return a graceful error response
        return {"message": {"content": f"I'm sorry, I encountered an error in the convert to map tool: {str(e)}. Please try again later."}}

#=====================================================================================
# System prompt for the eCommerce orchestrator agent
AGENT_SYSTEM_PROMPT = """
You are an intelligent E-commerce Assistant Agent designed to help users with queries related to products and orders. 

You are a orchestrator agent which has access to the following tools:
- get_default_questions: for listing the sample questions that the user can ask related to products and orders.
- get_answers_for_structured_data: for any queries related to Transactional Data, this tool already has the capability to generate SQL based on user query and execute and get the formatted results.
- get_answers_for_unstructured_data: for any query related to Product Knowledge Base, this toll already has the capability to query the knowledgebase based on user query and get the formatted results.
- current_time: for current time infomation.
- show_state_fact_map_tool: for displaying state-wise data on a geographical map iamge.

Your responsibilities:
- Categorize the user query and find out which of the above listed tools can answer the question.
- If the user question is unrelated then gracefully let the user with reasons of which the question can be answered.
- Else, call the specific tool 
- Remember the tools produce formatted results, please convert it to markdown so that it can be displayed on the frontend.
- When the tool - get_answers_for_structured_data is called,
    1) Show the Tabular data first, in a nicely formated tabular grid. Merge the column values while displaying if the data is grouped by a column. 
    2) Based on the tabular data, if possible, show a graphical representation of the data. 
    3) If there is state-wise geographical data, call the tool - show_state_fact_map_tool to show data on a map.
    4) Then Show the Key Insights, and please make it brief and concise
    5) Then if there is a SQL statement, show the generate SQL at the end within <details> tag.
- When the tool - get_answers_for_unstructured_data is called, show the sources and the preview at the end within <details> tag.

Remember previous context from the conversation when responding.
"""

# Create an Strands agent with our defined tools
agent = Agent(
    model=BEDROCK_MODEL_ID,
    tools=[get_default_questions, get_answers_for_structured_data, get_answers_for_unstructured_data, current_time],
    system_prompt=AGENT_SYSTEM_PROMPT
)

#=====================================================================================
async def process_user_input(user_query: str) -> str:
    """
    Creates a Strands agent that answers questions about order and products for a eCommerce solution
    using the 2 tools, one for Structured data and the second one for Unstructed data
    There is another additional tool, which help users by showing the sample questions
    
    Args:
        question: The customer's question or request
        
    Returns:
        The agent's response
    """

    log_conversation("User", user_query)

    try:

        start_time = time.time()

        # Process the question and stream the response back
        async for event in agent.stream_async(user_query):
            if "data" in event:
                yield event["data"]
        #response = agent(user_query)

        end_time = time.time()
        
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds")
        
        ## Format the response for display
        #if isinstance(response, dict):
        #    if "content" in response:
        #        return response["content"]
        #    elif "message" in response and "content" in response["message"]:
        #        return response["message"]["content"]
        
        ## Default return the full response
        #return str(response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Yield a graceful error response
        yield f"I'm sorry, I encountered an error: {str(e)}. Please try again later."

#=====================================================================================

