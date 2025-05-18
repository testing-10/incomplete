
"""
Accuracy Metrics Module

This module provides functions for evaluating the accuracy of model responses.
"""

import re
import string
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

def exact_match(
    response: str,
    expected: str,
    case_sensitive: bool = False
) -> float:
    """
    Check if the response exactly matches the expected answer.

    Args:
        response: Model response
        expected: Expected answer
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not case_sensitive:
        response = response.lower()
        expected = expected.lower()

    return 1.0 if response.strip() == expected.strip() else 0.0

def substring_match(
    response: str,
    expected: str,
    case_sensitive: bool = False
) -> float:
    """
    Check if the expected answer is a substring of the response.

    Args:
        response: Model response
        expected: Expected answer
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        1.0 if substring match, 0.0 otherwise
    """
    if not case_sensitive:
        response = response.lower()
        expected = expected.lower()

    return 1.0 if expected.strip() in response.strip() else 0.0

def token_overlap(
    response: str,
    expected: str,
    case_sensitive: bool = False,
    ignore_punctuation: bool = True
) -> float:
    """
    Calculate the token overlap between response and expected answer.

    Args:
        response: Model response
        expected: Expected answer
        case_sensitive: Whether to perform case-sensitive matching
        ignore_punctuation: Whether to ignore punctuation

    Returns:
        Overlap score between 0.0 and 1.0
    """
    if not case_sensitive:
        response = response.lower()
        expected = expected.lower()

    if ignore_punctuation:
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        response = response.translate(translator)
        expected = expected.translate(translator)

    # Tokenize by whitespace
    response_tokens = set(response.split())
    expected_tokens = set(expected.split())

    # Calculate overlap
    if not expected_tokens:
        return 0.0

    overlap = len(response_tokens.intersection(expected_tokens))
    return overlap / len(expected_tokens)

def numerical_comparison(
    response: str,
    expected: str,
    tolerance: float = 0.01
) -> float:
    """
    Compare numerical values in the response and expected answer.

    Args:
        response: Model response
        expected: Expected answer
        tolerance: Relative tolerance for numerical comparison

    Returns:
        1.0 if numbers match within tolerance, 0.0 otherwise
    """
    # Extract numbers from strings
    response_numbers = re.findall(r'[-+]?\d*\.\d+|\d+', response)
    expected_numbers = re.findall(r'[-+]?\d*\.\d+|\d+', expected)

    if not expected_numbers:
        return 0.0

    if not response_numbers:
        return 0.0

    # Try to match each expected number
    matches = 0
    for exp_num in expected_numbers:
        exp_val = float(exp_num)
        for resp_num in response_numbers:
            resp_val = float(resp_num)
            if abs(exp_val - resp_val) <= tolerance * abs(exp_val):
                matches += 1
                break

    return matches / len(expected_numbers)

def semantic_similarity(
    response: str,
    expected: str,
    model = None  # Optional embedding model
) -> float:
    """
    Calculate semantic similarity between response and expected answer.

    Note: This is a placeholder. In a real implementation, you would use
    an embedding model to calculate semantic similarity.

    Args:
        response: Model response
        expected: Expected answer
        model: Optional embedding model

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Placeholder implementation
    # In a real implementation, you would use embeddings

    # Simple fallback to token overlap
    return token_overlap(response, expected)

def evaluate_factual_accuracy(
    response: str,
    expected_answer: str,
    evaluation_method: str = "substring_match"
) -> Dict[str, float]:
    """
    Evaluate the factual accuracy of a response.

    Args:
        response: Model response
        expected_answer: Expected answer
        evaluation_method: Method to use for evaluation

    Returns:
        Dictionary of accuracy metrics
    """
    metrics = {}

    if evaluation_method == "exact_match":
        metrics["accuracy"] = exact_match(response, expected_answer)
    elif evaluation_method == "substring_match":
        metrics["accuracy"] = substring_match(response, expected_answer)
    elif evaluation_method == "token_overlap":
        metrics["accuracy"] = token_overlap(response, expected_answer)
    elif evaluation_method == "numerical_comparison":
        metrics["accuracy"] = numerical_comparison(response, expected_answer)
    elif evaluation_method == "semantic_similarity":
        metrics["accuracy"] = semantic_similarity(response, expected_answer)
    else:
        # Default to substring match
        metrics["accuracy"] = substring_match(response, expected_answer)

    # Check for hallucination indicators
    hallucination_indicators = [
        "I'm not sure",
        "I don't know",
        "I'm uncertain",
        "It's unclear",
        "I cannot provide"
    ]

    if any(indicator in response.lower() for indicator in hallucination_indicators):
        # Model expressing uncertainty is good when it doesn't know
        metrics["hallucination_rate"] = 0.1
    else:
        # Default value, would need more sophisticated evaluation
        metrics["hallucination_rate"] = 0.2

    return metrics