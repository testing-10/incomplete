
"""
OpenAI Client Module

This module provides a client for interacting with OpenAI's API.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union

from openai import OpenAI
from ..clients.base_client import BaseClient

logger = logging.getLogger(__name__)

class OpenAIClient(BaseClient):
    """Client for interacting with OpenAI's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            organization: OpenAI organization ID
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.organization = organization or os.environ.get("OPENAI_ORG_ID")

        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )

        # Get model ID from config
        self.model_id = model_config.get("model_id", "gpt-4o") if model_config else "gpt-4o"

        logger.info(f"Initialized OpenAI client with model: {self.model_id}")

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text using the OpenAI model.

        Args:
            prompt: Input prompt
            system_prompt: System prompt (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        return self.generate_chat_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate a chat response using the OpenAI model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        try:
            # Get additional parameters from kwargs or use defaults
            top_p = kwargs.get("top_p", 1.0)
            presence_penalty = kwargs.get("presence_penalty", 0.0)
            frequency_penalty = kwargs.get("frequency_penalty", 0.0)

            # Start timing
            start_time = time.time()

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )

            # End timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Log response time
            logger.debug(f"OpenAI response time: {execution_time:.2f} seconds")

            # Extract and return the response text
            if response.choices and len(response.choices) > 0:
                return {
                    "text": response.choices[0].message.content,
                    "model": self.model_id,
                    "execution_time": execution_time,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "raw_response": response
                }
            else:
                logger.warning("Empty response from OpenAI API")
                return {
                    "text": "",
                    "model": self.model_id,
                    "execution_time": execution_time,
                    "error": "Empty response"
                }

        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
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
            "provider": "OpenAI",
            "model_id": self.model_id
        })
        return model_info