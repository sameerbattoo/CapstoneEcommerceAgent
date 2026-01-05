"""
Metrics Logger Extension for AWS Bedrock AgentCore
Tracks execution steps with timing, tokens, and cost per session/tenant.

This module extends Python's logging.Logger class with emit_step_metrics() method
that automatically formats metrics as CloudWatch EMF (Embedded Metric Format).

Usage:
    import metrics_logger  # Auto-initializes on import
    
    logger.emit_step_metrics(
        session_id="session_123",
        tenant_id="tenant_abc",
        step_name="kb_retrieval",
        start_time=start_time,
        end_time=end_time,
        input_tokens=150,
        output_tokens=300,
        additional_data={"retrieved_docs": 5}
    )

Environment Variables:
    METRICS_NAMESPACE: CloudWatch namespace (default: ECommerceAgent/Metrics)
    ENABLE_METRICS_LOGGING: Enable/disable metrics (default: true)
    BEDROCK_MODEL_ID: Model ID for pricing calculations
"""

import logging
import json
import os
from typing import Dict, Any, Optional, Union
from datetime import datetime
import time

# Custom log level for metrics (between INFO and WARNING)
METRICS_LEVEL = 25
logging.addLevelName(METRICS_LEVEL, "METRICS")

# Pricing configuration (per 1K tokens)
# Source: https://aws.amazon.com/bedrock/pricing/
# Update these based on your model pricing
PRICING_CONFIG = {
    ## ======== (Global Cross-region Inference) ==================
    # Source: https://aws.amazon.com/bedrock/pricing/ (Global Cross-region Inference table)
    # Claude Sonnet 4.5 
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input_price_per_1k": 0.003,
        "output_price_per_1k": 0.015
    },
    # Claude Sonnet 4.5 - Long Context 
    "global.anthropic.claude-sonnet-4-5-long-context-20250929-v1:0": {
        "input_price_per_1k": 0.006,
        "output_price_per_1k": 0.0225
    },
    # Claude Haiku 4.5 
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {
        "input_price_per_1k": 0.001,
        "output_price_per_1k": 0.005
    },
    # Claude Opus 4.5 
    "global.anthropic.claude-opus-4-5-20250514-v1:0": {
        "input_price_per_1k": 0.005,
        "output_price_per_1k": 0.025
    },
    # Nova 2 Sonic  (Standard Tier)
    "global.amazon.nova-2-sonic-v1:0" : {
        "input_price_per_1k": 0.0003,
        "output_price_per_1k": 0.0025
    },
    ## ======== (US regional In-region Inference) ==================
    # Source: https://aws.amazon.com/bedrock/pricing/ (US In-region Inference table)
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input_price_per_1k": 0.0033,
        "output_price_per_1k": 0.0165
    },
    # Claude Sonnet 4.5 - Long Context 
    "us.anthropic.claude-sonnet-4-5-long-context-20250929-v1:0": {
        "input_price_per_1k": 0.0066,
        "output_price_per_1k": 0.02475
    },
    # Claude Haiku 4.5 
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {
        "input_price_per_1k": 0.0011,
        "output_price_per_1k": 0.0055
    },
    # Claude Opus 4.5 
    "us.anthropic.claude-opus-4-5-20250514-v1:0": {
        "input_price_per_1k": 0.0055,
        "output_price_per_1k": 0.0275
    },
    # Nova 2 Sonic  (Standard Tier)
    "us.amazon.nova-2-sonic-v1:0" : {
        "input_price_per_1k": 0.00033,
        "output_price_per_1k": 0.00275
    },
    # Default fallback pricing
    "default": {
        "input_price_per_1k": 0.003,
        "output_price_per_1k": 0.015
    }
}


def calculate_cost(input_tokens: int, output_tokens: int, model_id: Optional[str] = None) -> float:
    """
    Calculate cost based on token usage and model pricing.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Bedrock model ID (defaults to env var BEDROCK_MODEL_ID)
    
    Returns:
        Total cost in USD
    """
    if model_id is None:
        model_id = os.getenv("BEDROCK_MODEL_ID", "default")
    
    # Get pricing for model (fallback to default if not found)
    pricing = PRICING_CONFIG.get(model_id, PRICING_CONFIG["default"])
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * pricing["input_price_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_price_per_1k"]
    
    return round(input_cost + output_cost, 6)


def emit_step_metrics(
    self,
    session_id: str,
    tenant_id: str,
    step_name: str,
    start_time: float,
    end_time: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model_id: Optional[str] = None,
    status: str = "success",
    additional_data: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None
):
    """
    Emit step execution metrics with timing, tokens, and cost.
    
    Args:
        session_id: Unique session identifier
        tenant_id: Tenant identifier for multi-tenancy
        step_name: Name of the execution step (e.g., "kb_retrieval", "sql_query", "orchestration")
        start_time: Step start time (from time.time())
        end_time: Step end time (from time.time())
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        model_id: Bedrock model ID (defaults to env var BEDROCK_MODEL_ID)
        status: Execution status - "success" or "error" (default: "success")
        additional_data: Additional metadata (e.g., {"retrieved_docs": 5, "query_type": "semantic"})
        namespace: CloudWatch namespace (defaults to env var METRICS_NAMESPACE)
    
    Example:
        start_time = time.time()
        try:
            # ... do work ...
            end_time = time.time()
            
            logger.emit_step_metrics(
                session_id="session_123",
                tenant_id="tenant_abc",
                step_name="kb_retrieval",
                start_time=start_time,
                end_time=end_time,
                input_tokens=150,
                output_tokens=300,
                status="success",
                additional_data={"retrieved_docs": 5}
            )
        except Exception as e:
            end_time = time.time()
            logger.emit_step_metrics(
                session_id="session_123",
                tenant_id="tenant_abc",
                step_name="kb_retrieval",
                start_time=start_time,
                end_time=end_time,
                input_tokens=0,
                output_tokens=0,
                status="error",
                additional_data={"error": str(e)}
            )
    """
    # Note: We don't check isEnabledFor(METRICS_LEVEL) because metrics should
    # always be emitted regardless of the logger's level setting. The
    # StepMetricsHandler has its own level filtering.
    
    # Calculate duration
    duration_seconds = abs(end_time - start_time)
    
    # Calculate cost
    cost_usd = calculate_cost(input_tokens, output_tokens, model_id)
    
    # Build step metrics payload
    step_metrics = {
        "session_id": session_id,
        "tenant_id": tenant_id,
        "step_name": step_name,
        "status": status,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd,
        "model_id": model_id or os.getenv("BEDROCK_MODEL_ID", "unknown"),
        "namespace": namespace or os.getenv("METRICS_NAMESPACE", "ECommerceAgent/Metrics"),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add additional data if provided
    if additional_data:
        step_metrics["additional_data"] = additional_data
    
    # Log at METRICS level with structured data
    # Note: Only the StepMetricsHandler will process this (creates EMF + trace logs)
    # We don't call self._log() here to avoid duplicate formatted METRICS logs
    self._log(
        METRICS_LEVEL,
        "",  # Empty message - handler will create EMF and trace logs
        args=(),
        extra={"step_metrics": step_metrics}
    )


# Add emit_step_metrics() method to Logger class
logging.Logger.emit_step_metrics = emit_step_metrics


class StepMetricsHandler(logging.Handler):
    """
    Handler that converts emit_step_metrics() calls to CloudWatch EMF format.
    
    Creates metrics for:
    - Duration per step/tenant/session
    - Token usage per step/tenant/session
    - Cost per step/tenant/session
    """
    
    def __init__(self, namespace: str = "ECommerceAgent/Metrics"):
        super().__init__()
        self.namespace = namespace
        self.setLevel(METRICS_LEVEL)
    
    def emit(self, record):
        """
        Convert step metrics to EMF format and log to CloudWatch.
        
        For AgentCore: We write to stderr to avoid mixing with streaming response.
        AgentCore captures stderr separately from the streaming response.
        """
        try:
            # Check if this is a step metrics record
            if not hasattr(record, 'step_metrics'):
                return
            
            step_data = record.step_metrics
            
            # Create EMF formatted log for CloudWatch Metrics
            emf_log = self._create_emf_log(step_data)
            
            # Write to stderr so it doesn't mix with streaming response
            import sys
            sys.stderr.write(json.dumps(emf_log) + '\n')
            sys.stderr.flush()
            
            # Also create a detailed trace log for CloudWatch Logs Insights
            trace_log = self._create_trace_log(step_data)
            sys.stderr.write(json.dumps(trace_log) + '\n')
            sys.stderr.flush()
            
        except Exception as e:
            # Don't break the application if metrics fail
            self.handleError(record)
    
    def _create_emf_log(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create EMF (Embedded Metric Format) log for CloudWatch Metrics.
        
        This creates time-series metrics that can be graphed and alarmed on.
        """
        # Define metrics to emit
        metrics = {
            "Duration": step_data["duration_seconds"],
            "InputTokens": step_data["input_tokens"],
            "OutputTokens": step_data["output_tokens"],
            "TotalTokens": step_data["total_tokens"],
            "Cost": step_data["cost_usd"]
        }
        
        # Define dimensions for grouping
        dimensions = {
            "TenantId": step_data["tenant_id"],
            "StepName": step_data["step_name"],
            "SessionId": step_data["session_id"],
            "Status": step_data["status"]  # Add status as a dimension
        }
        
        # Build EMF structure
        emf_log = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": step_data.get("namespace", self.namespace),
                        "Dimensions": [
                            ["TenantId", "StepName", "Status"],  # Aggregate by tenant, step, and status
                            ["TenantId", "SessionId", "StepName", "Status"]  # Detailed per session with status
                        ],
                        "Metrics": [
                            {"Name": "Duration", "Unit": "Seconds"},
                            {"Name": "InputTokens", "Unit": "Count"},
                            {"Name": "OutputTokens", "Unit": "Count"},
                            {"Name": "TotalTokens", "Unit": "Count"},
                            {"Name": "Cost", "Unit": "Count"}  # Cost in USD (CloudWatch doesn't have a currency unit)
                        ]
                    }
                ]
            }
        }
        
        # Add dimension values
        emf_log.update(dimensions)
        
        # Add metric values
        emf_log.update(metrics)
        
        # Add metadata
        emf_log["_type"] = "step_metrics_emf"
        
        return emf_log
    
    def _create_trace_log(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed trace log for CloudWatch Logs Insights queries.
        
        This allows querying execution traces by session, tenant, or step.
        """
        trace_log = {
            "_type": "step_trace",
            "session_id": step_data["session_id"],
            "tenant_id": step_data["tenant_id"],
            "step_name": step_data["step_name"],
            "status": step_data["status"],
            "start_time": step_data["start_time"],
            "end_time": step_data["end_time"],
            "duration_seconds": step_data["duration_seconds"],
            "input_tokens": step_data["input_tokens"],
            "output_tokens": step_data["output_tokens"],
            "total_tokens": step_data["total_tokens"],
            "cost_usd": step_data["cost_usd"],
            "model_id": step_data["model_id"],
            "timestamp": step_data["timestamp"]
        }
        
        # Add additional data if present
        if "additional_data" in step_data:
            trace_log["additional_data"] = step_data["additional_data"]
        
        return trace_log


def initialize_metrics_logging(namespace: str = "CapstoneECommerceAgent/Metrics", enable_metrics: bool = True):
    """
    Initialize the metrics logging extension.
    Must be called explicitly after logging configuration is set up.
    
    Args:
        namespace: CloudWatch namespace for metrics
        enable_metrics: Whether to enable metrics logging
    """
    if not enable_metrics:
        return
    
    # Create and configure step metrics handler
    step_handler = StepMetricsHandler(namespace=namespace)
    
    # Add to root logger so all loggers inherit it
    root_logger = logging.getLogger()
    root_logger.addHandler(step_handler)


# DO NOT auto-initialize - will be called explicitly after configuration is loaded
# This prevents creating log streams during module import
