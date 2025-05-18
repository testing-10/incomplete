
#!/usr/bin/env python

"""
Main entry point for the Foundation Model Testing Framework.
"""

import os
import sys
import argparse
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.core.model_loader import ModelLoader
from src.core.result_manager import ResultManager
from src.testers.foundation_tester import FoundationModelTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("foundation-model-testing")

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Foundation Model Testing Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="config/main.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to test (e.g., 'gpt4o,claude3_sonnet')"
    )
    parser.add_argument(
        "--test-suites",
        type=str,
        default=None,
        help="Comma-separated list of test suites to run (e.g., 'factual_accuracy,reasoning')"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def filter_models(config: Dict[str, Any], models_arg: str) -> Dict[str, Any]:
    """Filter models based on command line argument."""
    if not models_arg:
        return config

    requested_models = models_arg.lower().split(',')
    filtered_config = config.copy()

    # Filter models in the foundation_llm category
    if "categories" in filtered_config and "foundation_llm" in filtered_config["categories"]:
        category = filtered_config["categories"]["foundation_llm"]
        if "models" in category:
            category["models"] = [
                model for model in category["models"]
                if model.get("name", "").lower() in requested_models
            ]

    return filtered_config

def filter_test_suites(config: Dict[str, Any], test_suites_arg: str) -> Dict[str, Any]:
    """Filter test suites based on command line argument."""
    if not test_suites_arg:
        return config

    requested_suites = test_suites_arg.lower().split(',')
    filtered_config = config.copy()

    # Filter test suites in the foundation_llm category
    if "categories" in filtered_config and "foundation_llm" in filtered_config["categories"]:
        category = filtered_config["categories"]["foundation_llm"]
        if "test_suites" in category:
            category["test_suites"] = [
                suite for suite in category["test_suites"]
                if suite.get("name", "").lower() in requested_suites
            ]

    return filtered_config

def display_summary(result_manager: ResultManager) -> None:
    """Display a summary of test results."""
    console = Console()
    console.print("\n[bold green]Foundation Model Testing - Results Summary[/bold green]\n")

    # Load summary from file
    summary_path = os.path.join(result_manager.run_dir, "summary.json")
    if not os.path.exists(summary_path):
        console.print("[yellow]No summary file found[/yellow]")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Display basic info
    console.print(f"[bold]Test Run:[/bold] {summary['timestamp']}")
    console.print(f"[bold]Total Tests:[/bold] {summary['total_tests']}")
    console.print(f"[bold]Models Tested:[/bold] {', '.join(summary['models_tested'])}")
    console.print()

    # Create table for model summaries
    table = Table(title="Model Performance Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Tests", style="green")
    table.add_column("Errors", style="red")
    table.add_column("Avg Time (s)", style="yellow")

    # Add metric columns
    metric_columns = set()
    for model_name, model_summary in summary["model_summaries"].items():
        if "metrics" in model_summary:
            metric_columns.update(model_summary["metrics"].keys())

    for metric in sorted(metric_columns):
        table.add_column(f"Avg {metric.replace('_', ' ').title()}", style="blue")

    # Add rows for each model
    for model_name, model_summary in summary["model_summaries"].items():
        row = [
            model_name,
            str(model_summary["total_tests"]),
            str(model_summary["errors"]),
            f"{model_summary['avg_execution_time']:.2f}"
        ]

        # Add metric values
        for metric in sorted(metric_columns):
            if metric in model_summary.get("metrics", {}):
                row.append(f"{model_summary['metrics'][metric]['avg']:.2f}")
            else:
                row.append("N/A")

        table.add_row(*row)

    console.print(table)
    console.print(f"\nDetailed results saved to: {result_manager.run_dir}")

def main():
    """Main entry point for the testing framework."""
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Filter models and test suites
    config = filter_models(config, args.models)
    config = filter_test_suites(config, args.test_suites)

    # Initialize components
    model_loader = ModelLoader()
    result_manager = ResultManager(args.output)

    # Get foundation_llm category configuration
    if "categories" not in config or "foundation_llm" not in config["categories"]:
        logger.error("foundation_llm category not found in configuration")
        sys.exit(1)

    foundation_config = config["categories"]["foundation_llm"]

    # Load model configurations
    model_loader.load_model_configs(foundation_config.get("models", []))

    # Initialize tester
    tester = FoundationModelTester(
        config=foundation_config,
        model_loader=model_loader,
        result_manager=result_manager
    )

    # Run tests
    tester.run_tests(
        parallel=args.parallel,
        max_workers=args.workers
    )

    # Display summary
    display_summary(result_manager)

if __name__ == "__main__":
    main()