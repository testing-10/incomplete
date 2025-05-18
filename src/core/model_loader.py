
"""
Model Loader Module

This module handles loading model configurations and initializing clients.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..clients.openai_client import OpenAIClient
from ..clients.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading model configurations and initializing clients."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the model loader.

        Args:
            config_dir: Directory containing model configurations
        """
        self.config_dir = config_dir
        self.model_configs = {}
        self.clients = {}

    def load_model_configs(self, models_list: List[Dict[str, Any]]) -> None:
        """
        Load model configurations from the specified list.

        Args:
            models_list: List of model configurations with name and config_path
        """
        for model_entry in models_list:
            model_name = model_entry.get("name")
            config_path = model_entry.get("config_path")

            if not model_name or not config_path:
                logger.warning(f"Skipping invalid model entry: {model_entry}")
                continue

            full_path = os.path.join(self.config_dir, config_path)

            try:
                with open(full_path, 'r') as f:
                    model_config = yaml.safe_load(f)

                self.model_configs[model_name] = model_config
                logger.info(f"Loaded configuration for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load configuration for model {model_name}: {e}")

    def initialize_client(self, model_name: str) -> Any:
        """
        Initialize a client for the specified model.

        Args:
            model_name: Name of the model to initialize

        Returns:
            Initialized client
        """
        if model_name in self.clients:
            return self.clients[model_name]

        if model_name not in self.model_configs:
            logger.error(f"No configuration found for model: {model_name}")
            return None

        model_config = self.model_configs[model_name]
        provider = model_config.get("provider", "").lower()

        try:
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OPENAI_API_KEY not found in environment variables")
                    return None

                org_id = os.environ.get("OPENAI_ORG_ID")
                client = OpenAIClient(api_key=api_key, organization=org_id, model_config=model_config)

            elif provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.error("ANTHROPIC_API_KEY not found in environment variables")
                    return None

                client = AnthropicClient(api_key=api_key, model_config=model_config)

            else:
                logger.error(f"Unsupported provider: {provider}")
                return None

            self.clients[model_name] = client
            logger.info(f"Initialized client for model: {model_name}")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize client for model {model_name}: {e}")
            return None

    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get capabilities of the specified model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of model capabilities
        """
        if model_name not in self.model_configs:
            logger.error(f"No configuration found for model: {model_name}")
            return {}

        model_config = self.model_configs[model_name]
        return model_config.get("capabilities", {})

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for the specified model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of model parameters
        """
        if model_name not in self.model_configs:
            logger.error(f"No configuration found for model: {model_name}")
            return {}

        model_config = self.model_configs[model_name]
        return model_config.get("testing_parameters", {})