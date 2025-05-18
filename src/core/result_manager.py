
"""
Result Manager Module

This module handles storing and processing test results.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TestResult:
    """Class representing a test result."""

    def __init__(
        self,
        model_name: str,
        test_type: str,
        test_case_id: str,
        domain: str,
        prompt: str,
        response: Optional[str],
        metrics: Dict[str, Any],
        execution_time: float,
        raw_response: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extracted_content: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a test result.

        Args:
            model_name: Name of the model
            test_type: Type of test
            test_case_id: ID of the test case
            domain: Domain of the test
            prompt: Input prompt
            response: Model response
            metrics: Evaluation metrics
            execution_time: Execution time in seconds
            raw_response: Raw response from the model
            error: Error message if any
            metadata: Additional metadata
            extracted_content: Extracted content from response
        """
        self.model_name = model_name
        self.test_type = test_type
        self.test_case_id = test_case_id
        self.domain = domain
        self.prompt = prompt
        self.response = response
        self.metrics = metrics
        self.execution_time = execution_time
        self.raw_response = raw_response
        self.error = error
        self.metadata = metadata or {}
        self.extracted_content = extracted_content or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self, include_raw_response: bool = False) -> Dict[str, Any]:
        """
        Convert the test result to a dictionary.

        Args:
            include_raw_response: Whether to include the raw response

        Returns:
            Dictionary representation of the test result
        """
        result_dict = {
            "model_name": self.model_name,
            "test_type": self.test_type,
            "test_case_id": self.test_case_id,
            "domain": self.domain,
            "prompt": self.prompt,
            "response": self.response,
            "metrics": self.metrics,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

        if self.error:
            result_dict["error"] = self.error

        if self.extracted_content:
            result_dict["extracted_content"] = self.extracted_content

        if include_raw_response and self.raw_response:
            # Try to convert raw response to a serializable format
            try:
                if hasattr(self.raw_response, "model_dump"):
                    # For Pydantic models
                    result_dict["raw_response"] = self.raw_response.model_dump()
                elif hasattr(self.raw_response, "to_dict"):
                    # For objects with to_dict method
                    result_dict["raw_response"] = self.raw_response.to_dict()
                else:
                    # Try direct serialization
                    json.dumps(self.raw_response)
                    result_dict["raw_response"] = self.raw_response
            except:
                # If serialization fails, store as string
                result_dict["raw_response"] = str(self.raw_response)

        return result_dict


class ResultManager:
    """Handles storing and processing test results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the result manager.

        Args:
            output_dir: Directory to store results
        """
        self.output_dir = output_dir
        self.results = []

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp for this test run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        logger.info(f"Results will be saved to {self.run_dir}")

    def add_result(self, result: TestResult) -> None:
        """
        Add a test result.

        Args:
            result: Test result to add
        """
        self.results.append(result)

    def save_results(self, include_raw_responses: bool = False) -> None:
        """
        Save all results to files.

        Args:
            include_raw_responses: Whether to include raw responses
        """
        # Group results by model
        model_results = {}
        for result in self.results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result.to_dict(include_raw_responses))

        # Save results for each model
        for model_name, results in model_results.items():
            model_dir = os.path.join(self.run_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save as JSON
            json_path = os.path.join(model_dir, "results.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Save as YAML
            yaml_path = os.path.join(model_dir, "results.yaml")
            with open(yaml_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)

            logger.info(f"Saved results for model {model_name}")

        # Save summary
        self._save_summary()

    def _save_summary(self) -> None:
        """Save a summary of all results."""
        summary = {
            "timestamp": self.timestamp,
            "total_tests": len(self.results),
            "models_tested": list(set(r.model_name for r in self.results)),
            "test_types": list(set(r.test_type for r in self.results)),
            "domains": list(set(r.domain for r in self.results)),
            "model_summaries": {}
        }

        # Create summaries for each model
        for model_name in summary["models_tested"]:
            model_results = [r for r in self.results if r.model_name == model_name]

            model_summary = {
                "total_tests": len(model_results),
                "errors": sum(1 for r in model_results if r.error),
                "avg_execution_time": sum(r.execution_time for r in model_results) / len(model_results),
                "metrics": {}
            }

            # Aggregate metrics
            all_metrics = {}
            for result in model_results:
                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)

            # Calculate average for each metric
            for metric_name, values in all_metrics.items():
                model_summary["metrics"][metric_name] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }

            summary["model_summaries"][model_name] = model_summary

        # Save summary as JSON
        json_path = os.path.join(self.run_dir, "summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save summary as YAML
        yaml_path = os.path.join(self.run_dir, "summary.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        logger.info(f"Saved results summary")