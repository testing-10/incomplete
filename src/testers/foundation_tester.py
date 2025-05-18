
"""
Foundation Model Tester Module

This module contains the tester for foundation language models.
"""

import os
import json
import logging
import time
import yaml
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ..core.model_loader import ModelLoader
from ..core.result_manager import TestResult, ResultManager

logger = logging.getLogger(__name__)

class FoundationModelTester:
    """Tester for foundation language models."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_loader: ModelLoader,
        result_manager: ResultManager
    ):
        """
        Initialize the foundation model tester.

        Args:
            config: Configuration dictionary
            model_loader: Model loader instance
            result_manager: Result manager instance
        """
        self.config = config
        self.model_loader = model_loader
        self.result_manager = result_manager

        # Extract models to test
        self.models_to_test = []
        for model_entry in config.get("models", []):
            self.models_to_test.append(model_entry.get("name"))

        # Load test suites
        self.test_suites = {}
        for suite_entry in config.get("test_suites", []):
            suite_name = suite_entry.get("name")
            suite_path = suite_entry.get("config_path")

            if suite_name and suite_path:
                full_path = os.path.join("config", suite_path)
                try:
                    with open(full_path, 'r') as f:
                        suite_config = yaml.safe_load(f)
                        self.test_suites[suite_name] = suite_config
                        logger.info(f"Loaded test suite: {suite_name}")
                except Exception as e:
                    logger.error(f"Failed to load test suite {suite_name}: {e}")

        # Load test data
        self.test_data = {}
        self._load_test_data()

        logger.info(f"FoundationModelTester initialized with {len(self.models_to_test)} models and {len(self.test_suites)} test suites")

    def _load_test_data(self) -> None:
        """Load test data from files."""
        data_files = {
            "factual_questions": "data/factual_questions.json",
            "reasoning_examples": "data/reasoning_examples.json",
            "instruction_examples": "data/instruction_examples.json",
            "context_examples": "data/context_examples.json"
        }

        for data_name, file_path in data_files.items():
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        self.test_data[data_name] = json.load(f)
                        logger.info(f"Loaded test data: {data_name}")
            except Exception as e:
                logger.error(f"Failed to load test data {data_name}: {e}")

    def run_tests(self, parallel: bool = True, max_workers: int = 4) -> None:
        """
        Run all tests for all models.

        Args:
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of parallel workers
        """
        logger.info(f"Starting tests for {len(self.models_to_test)} models")

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for model_name in self.models_to_test:
                    futures.append(executor.submit(self._test_model, model_name))

                for future in futures:
                    future.result()  # Wait for completion
        else:
            for model_name in self.models_to_test:
                self._test_model(model_name)

        # Save all results
        self.result_manager.save_results(include_raw_responses=True)
        logger.info("All tests completed")

    def _test_model(self, model_name: str) -> None:
        """
        Run all tests for a specific model.

        Args:
            model_name: Name of the model to test
        """
        logger.info(f"Testing model: {model_name}")

        # Initialize client
        client = self.model_loader.initialize_client(model_name)
        if not client:
            logger.error(f"Failed to initialize client for model: {model_name}")
            return

        # Run each test suite
        for suite_name, suite_config in self.test_suites.items():
            logger.info(f"Running test suite {suite_name} for model {model_name}")

            test_type = suite_config.get("name", suite_name)

            # Get test cases
            test_cases = suite_config.get("test_cases", [])

            # Run each test case
            for test_case in test_cases:
                case_id = test_case.get("id", "unknown")
                domain = test_case.get("domain", "general")

                logger.info(f"Running test case {case_id} for model {model_name}")

                # Get examples for this test case
                examples = test_case.get("examples", [])

                for i, example in enumerate(examples):
                    # Format prompt
                    prompt_template = test_case.get("prompt_template", "")
                    input_data = example.get("input", {})

                    # Replace placeholders in prompt template
                    prompt = prompt_template
                    for key, value in input_data.items():
                        placeholder = f"{{{key}}}"
                        prompt = prompt.replace(placeholder, str(value))

                    # Get expected output
                    expected_output = example.get("expected_output", "")

                    # Get model parameters
                    model_params = self.model_loader.get_model_parameters(model_name)

                    # Use system prompt if available
                    system_prompt = model_params.get("default_system_prompt", None)

                    try:
                        # Start timing
                        start_time = time.time()

                        # Generate response
                        response = client.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=0.0,  # Use 0 for deterministic results
                            max_tokens=1000
                        )

                        # End timing
                        end_time = time.time()
                        execution_time = end_time - start_time

                        # Extract response text
                        if isinstance(response, dict):
                            response_text = response.get("text", "")
                        else:
                            response_text = str(response)

                        # Evaluate response
                        metrics = self._evaluate_response(
                            response_text,
                            expected_output,
                            test_type,
                            domain
                        )

                        # Create test result
                        result = TestResult(
                            model_name=model_name,
                            test_type=test_type,
                            test_case_id=f"{case_id}_{i}",
                            domain=domain,
                            prompt=prompt,
                            response=response_text,
                            metrics=metrics,
                            execution_time=execution_time,
                            raw_response=response
                        )

                        # Add result
                        self.result_manager.add_result(result)

                    except Exception as e:
                        logger.error(f"Error in test case {case_id}_{i} for model {model_name}: {e}")

                        # Create error result
                        result = TestResult(
                            model_name=model_name,
                            test_type=test_type,
                            test_case_id=f"{case_id}_{i}",
                            domain=domain,
                            prompt=prompt,
                            response=None,
                            metrics={},
                            execution_time=0,
                            error=str(e)
                        )

                        # Add result
                        self.result_manager.add_result(result)

    def _evaluate_response(
        self,
        response: str,
        expected_output: str,
        test_type: str,
        domain: str
    ) -> Dict[str, Any]:
        """
        Evaluate a model response.

        Args:
            response: Model response
            expected_output: Expected output
            test_type: Type of test
            domain: Domain of the test

        Returns:
            Evaluation metrics
        """
        metrics = {}

        # Basic accuracy check
        if expected_output:
            # For exact match tests
            if expected_output.lower() in response.lower():
                metrics["accuracy"] = 1.0
            else:
                # Check for partial match
                words_expected = set(expected_output.lower().split())
                words_response = set(response.lower().split())
                overlap = len(words_expected.intersection(words_response))
                metrics["accuracy"] = overlap / len(words_expected) if words_expected else 0

        # Reasoning evaluation
        if test_type == "foundation_llm_reasoning_tests":
            # Check for step-by-step reasoning
            if "step by step" in response.lower() or "first" in response.lower():
                metrics["step_by_step_articulation"] = 1.0
            else:
                metrics["step_by_step_articulation"] = 0.5

            # Check for logical consistency
            metrics["logical_consistency"] = 0.8  # Default value, would need more sophisticated evaluation

            # Check for reasoning path validity
            metrics["reasoning_path_validity"] = 0.8  # Default value, would need more sophisticated evaluation

        # Factual accuracy evaluation
        if test_type == "foundation_llm_factual_accuracy_tests":
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

            # Check for source citation
            if "according to" in response.lower() or "source:" in response.lower():
                metrics["source_citation"] = 1.0
            else:
                metrics["source_citation"] = 0.0

        # Instruction following evaluation
        if test_type == "foundation_llm_instruction_following_tests":
            # Check if response follows instructions
            metrics["instruction_following"] = 0.9  # Default value, would need more sophisticated evaluation

        # Extended context evaluation
        if test_type == "foundation_llm_extended_context_tests":
            # Check for context utilization
            metrics["context_utilization"] = 0.8  # Default value, would need more sophisticated evaluation

        return metrics