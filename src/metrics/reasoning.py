
"""
Reasoning Metrics Module

This module provides functions for evaluating the reasoning capabilities of model responses.
"""

import re
from typing import Dict, Any, List, Optional, Union, Tuple

def evaluate_step_by_step(response: str) -> float:
    """
    Evaluate whether the response contains step-by-step reasoning.

    Args:
        response: Model response

    Returns:
        Score between 0.0 and 1.0
    """
    # Check for step indicators
    step_indicators = [
        r"step\s*\d+",  # "Step 1", "Step 2", etc.
        r"first\s*,",  # "First,"
        r"second\s*,",  # "Second,"
        r"third\s*,",  # "Third,"
        r"finally\s*,",  # "Finally,"
        r"^\d+\s*\.",  # "1.", "2.", etc. at the beginning of a line
        r"^\*\s",  # "* " bullet points at the beginning of a line
        r"^-\s",  # "- " bullet points at the beginning of a line
    ]

    # Count matches
    match_count = 0
    for pattern in step_indicators:
        matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
        match_count += len(matches)

    # Score based on number of step indicators
    if match_count >= 3:
        return 1.0
    elif match_count >= 1:
        return 0.5
    else:
        return 0.0

def evaluate_logical_consistency(response: str) -> float:
    """
    Evaluate the logical consistency of the response.

    Args:
        response: Model response

    Returns:
        Score between 0.0 and 1.0
    """
    # Check for logical connectors
    logical_connectors = [
        "therefore",
        "thus",
        "hence",
        "consequently",
        "because",
        "since",
        "as a result",
        "it follows that",
        "implies",
        "if",
        "then"
    ]

    # Count logical connectors
    connector_count = 0
    for connector in logical_connectors:
        connector_count += response.lower().count(connector)

    # Score based on number of logical connectors
    if connector_count >= 3:
        return 1.0
    elif connector_count >= 1:
        return 0.7
    else:
        return 0.3

def evaluate_reasoning_path(response: str, expected_answer: str) -> float:
    """
    Evaluate the validity of the reasoning path.

    Args:
        response: Model response
        expected_answer: Expected answer

    Returns:
        Score between 0.0 and 1.0
    """
    # Check if the response contains the expected answer
    contains_answer = expected_answer.lower() in response.lower()

    # Check for step-by-step reasoning
    step_score = evaluate_step_by_step(response)

    # Check for logical consistency
    logic_score = evaluate_logical_consistency(response)

    # Combine scores
    if contains_answer:
        # If the answer is correct, weight the reasoning process
        return 0.4 + (0.3 * step_score) + (0.3 * logic_score)
    else:
        # If the answer is wrong, the reasoning path is less valid
        return (0.5 * step_score) + (0.5 * logic_score)

def evaluate_reasoning(
    response: str,
    expected_answer: str,
    reasoning_type: str = "logical"
) -> Dict[str, float]:
    """
    Evaluate the reasoning capabilities of a response.

    Args:
        response: Model response
        expected_answer: Expected answer
        reasoning_type: Type of reasoning to evaluate

    Returns:
        Dictionary of reasoning metrics
    """
    metrics = {}

    # Evaluate step-by-step articulation
    metrics["step_by_step_articulation"] = evaluate_step_by_step(response)

    # Evaluate logical consistency
    metrics["logical_consistency"] = evaluate_logical_consistency(response)

    # Evaluate reasoning path validity
    metrics["reasoning_path_validity"] = evaluate_reasoning_path(response, expected_answer)

    # Calculate overall reasoning score
    if reasoning_type == "logical":
        # For logical reasoning, weight logical consistency more
        metrics["reasoning_score"] = (
            0.3 * metrics["step_by_step_articulation"] +
            0.4 * metrics["logical_consistency"] +
            0.3 * metrics["reasoning_path_validity"]
        )
    elif reasoning_type == "mathematical":
        # For mathematical reasoning, weight step-by-step more
        metrics["reasoning_score"] = (
            0.4 * metrics["step_by_step_articulation"] +
            0.2 * metrics["logical_consistency"] +
            0.4 * metrics["reasoning_path_validity"]
        )
    else:
        # Default weighting
        metrics["reasoning_score"] = (
            0.33 * metrics["step_by_step_articulation"] +
            0.33 * metrics["logical_consistency"] +
            0.34 * metrics["reasoning_path_validity"]
        )

    return metrics