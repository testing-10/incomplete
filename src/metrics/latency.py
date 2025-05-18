
"""
Latency Metrics Module

This module provides functions for evaluating the latency of model responses.
"""

from typing import Dict, Any, List, Optional, Union, Tuple

def calculate_tokens_per_second(
    input_tokens: int,
    output_tokens: int,
    execution_time: float
) -> float:
    """
    Calculate tokens per second for a model response.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        execution_time: Execution time in seconds

    Returns:
        Tokens per second
    """
    total_tokens = input_tokens + output_tokens

    if execution_time <= 0:
        return 0.0

    return total_tokens / execution_time

def calculate_output_tokens_per_second(
    output_tokens: int,
    execution_time: float
) -> float:
    """
    Calculate output tokens per second for a model response.

    Args:
        output_tokens: Number of output tokens
        execution_time: Execution time in seconds

    Returns:
        Output tokens per second
    """
    if execution_time <= 0:
        return 0.0

    return output_tokens / execution_time

def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_token: float,
    output_cost_per_token: float
) -> float:
    """
    Calculate the cost of a model response.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost_per_token: Cost per input token
        output_cost_per_token: Cost per output token

    Returns:
        Total cost
    """
    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token

    return input_cost + output_cost

def calculate_cost_per_1k_tokens(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_token: float,
    output_cost_per_token: float
) -> float:
    """
    Calculate the cost per 1,000 tokens.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost_per_token: Cost per input token
        output_cost_per_token: Cost per output token

    Returns:
        Cost per 1,000 tokens
    """
    total_tokens = input_tokens + output_tokens

    if total_tokens <= 0:
        return 0.0

    total_cost = calculate_cost(
        input_tokens,
        output_tokens,
        input_cost_per_token,
        output_cost_per_token
    )

    return (total_cost / total_tokens) * 1000

def evaluate_latency(
    execution_time: float,
    input_tokens: int,
    output_tokens: int,
    model_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate the latency metrics of a model response.

    Args:
        execution_time: Execution time in seconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_config: Model configuration dictionary

    Returns:
        Dictionary of latency metrics
    """
    metrics = {}

    # Calculate tokens per second
    metrics["tokens_per_second"] = calculate_tokens_per_second(
        input_tokens,
        output_tokens,
        execution_time
    )

    # Calculate output tokens per second
    metrics["output_tokens_per_second"] = calculate_output_tokens_per_second(
        output_tokens,
        execution_time
    )

    # Get pricing information from model config
    pricing = model_config.get("pricing", {})
    input_cost_per_token = pricing.get("input_tokens", 0.0)
    output_cost_per_token = pricing.get("output_tokens", 0.0)

    # Calculate cost
    metrics["cost"] = calculate_cost(
        input_tokens,
        output_tokens,
        input_cost_per_token,
        output_cost_per_token
    )

    # Calculate cost per 1k tokens
    metrics["cost_per_1k_tokens"] = calculate_cost_per_1k_tokens(
        input_tokens,
        output_tokens,
        input_cost_per_token,
        output_cost_per_token
    )

    return metrics