
"""
Base Client Module

This module provides a base class for model API clients.
"""

from typing import Dict, List, Any, Optional, Union

class BaseClient:
    """Base class for model API clients."""

    def __init__(self, model_config: Dict[str, Any] = None):
        """
        Initialize the base client.

        Args:
            model_config: Model configuration dictionary
        """
        self.model_config = model_config or {}

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt
            system_prompt: System prompt (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text or response object
        """
        raise NotImplementedError("Subclasses must implement generate_text")

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a chat response using the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated response or response object
        """
        raise NotImplementedError("Subclasses must implement generate_chat_response")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return self.model_config