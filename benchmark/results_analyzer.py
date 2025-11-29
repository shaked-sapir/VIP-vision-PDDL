"""
Results Analyzer for Benchmark System

Analyzes experiment results and generates:
1. CSV files with metric comparisons
2. Visualization plots
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_experiment_results(results_dir: Path) -> Dict:
    """
    Load experiment results from all algorithms.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Dictionary with all results
    """
    results_file = results_dir / "benchmark_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)


def generate_comparison_csv(results: Dict, output_file: Path) -> None:
    """
    Generate CSV file comparing all algorithms.

    Args:
        results: Dictionary with experiment results
        output_file: Path to save CSV file
    """
    print("Generating comparison CSV...")

    # TODO: Implement CSV generation
    # Columns: Algorithm, Learning Time, Test Problem, Success, Plan Length, etc.

    print(f"✓ CSV saved to: {output_file}")


def generate_plots(results: Dict, output_dir: Path) -> None:
    """
    Generate visualization plots.

    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save plots
    """
    print("Generating comparison plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement plot generation
    # 1. Learning time comparison (bar chart)
    # 2. Success rate comparison (bar chart)
    # 3. Plan length comparison (box plot)
    # 4. Per-problem comparison (grouped bar chart)

    print(f"✓ Plots saved to: {output_dir}")


def analyze_benchmark_results(results_dir: Path, output_dir: Path) -> None:
    """
    Analyze benchmark results and generate outputs.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis outputs
    """
    print("="*80)
    print("BENCHMARK RESULTS ANALYSIS")
    print("="*80)
    print()

    # Load results
    results = load_experiment_results(results_dir)

    print(f"Domain: {results['domain']}")
    print(f"Training problem: {results['training_problem']}")
    print(f"Test problems: {results['num_test_problems']}")
    print()

    # Generate CSV
    csv_file = output_dir / "comparison.csv"
    generate_comparison_csv(results, csv_file)
    print()

    # Generate plots
    plots_dir = output_dir / "plots"
    generate_plots(results, plots_dir)
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze benchmark experiment results"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="blocks",
        help="Domain to analyze results for (default: blocks)"
    )

    args = parser.parse_args()

    # Setup paths
    benchmark_dir = Path(__file__).parent
    results_dir = benchmark_dir / "results" / args.domain
    output_dir = results_dir

    # Verify results exist
    if not results_dir.exists():
        print(f"Error: Results not found at {results_dir}")
        print("Please run experiment_runner.py first")
        sys.exit(1)

    # Analyze results
    analyze_benchmark_results(results_dir, output_dir)
