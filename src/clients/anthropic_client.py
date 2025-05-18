"""
Anthropic Client Module

This module provides a client for interacting with Anthropic's API.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union

import anthropic
from ..clients.base_client import BaseClient

logger = logging.getLogger(__name__)

class AnthropicClient(BaseClient):
    """Client for interacting with Anthropic's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Get model ID from config
        self.model_id = model_config.get("model_id", "claude-3-sonnet-20240229") if model_config else "claude-3-sonnet-20240229"

        logger.info(f"Initialized Anthropic client with model: {self.model_id}")

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text using the Anthropic model.

        Args:
            prompt: Input prompt
            system_prompt: System prompt (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            # Start timing
            start_time = time.time()

            # Get additional parameters from kwargs or use defaults
            top_p = kwargs.get("top_p", 1.0)

            # Create API call parameters
            api_params = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "top_p": top_p,
            }

            # Add system prompt if provided
            if system_prompt:
                api_params["system"] = system_prompt

            # Make API call
            response = self.client.messages.create(**api_params)

            # End timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Log response time
            logger.debug(f"Anthropic response time: {execution_time:.2f} seconds")

            return {
                "text": response.content[0].text,
                "model": self.model_id,
                "execution_time": execution_time,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "raw_response": response
            }

        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {e}")
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_id,
                "error": str(e)
            }

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate a chat response using the Anthropic model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        try:
            # Extract system prompt if present
            system_prompt = None
            chat_messages = []

            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                else:
                    # Convert OpenAI format to Anthropic format
                    role = "user" if message["role"] == "user" else "assistant"
                    chat_messages.append({"role": role, "content": message["content"]})

            # Start timing
            start_time = time.time()

            # Get additional parameters from kwargs or use defaults
            top_p = kwargs.get("top_p", 1.0)

            # Create API call parameters
            api_params = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": chat_messages,
                "top_p": top_p,
            }

            # Add system prompt if provided
            if system_prompt:
                api_params["system"] = system_prompt

            # Make API call
            response = self.client.messages.create(**api_params)

            # End timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Log response time
            logger.debug(f"Anthropic response time: {execution_time:.2f} seconds")

            return {
                "text": response.content[0].text,
                "model": self.model_id,
                "execution_time": execution_time,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "raw_response": response
            }

        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {e}")
            return {
                "text": f"Error: {str(e)}",
                "model": self.model_id,
                "error": str(e)
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        model_info = super().get_model_info()
        model_info.update({
            "provider": "Anthropic",
            "model_id": self.model_id
        })
        return model_info