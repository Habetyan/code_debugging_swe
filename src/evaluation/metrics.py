"""
Evaluation Metrics

Implements Pass@k and other metrics for evaluating code generation.
"""

from typing import Callable
import math


def calculate_pass_at_k(
    n: int,
    c: int, 
    k: int
) -> float:
    """
    Calculate Pass@k metric using the unbiased estimator.
    
    Formula: 1 - C(n-c, k) / C(n, k)
    
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k value for Pass@k
        
    Returns:
        Pass@k probability
    """
    if n - c < k:
        return 1.0
    
    return 1.0 - math.prod(range(n - c + 1, n + 1)) / math.prod(range(n - k + 1, n + 1))


def aggregate_pass_at_k(
    results: list[dict],
    k_values: list[int] = [1, 3, 5],
    is_correct_fn: Callable[[dict], bool] = lambda r: r.get("correct", False)
) -> dict[str, float]:
    """
    Aggregate Pass@k across multiple instances.
    
    Args:
        results: List of result dicts, each with attempts for one instance
        k_values: k values to compute
        is_correct_fn: Function to determine if a result is correct
        
    Returns:
        Dict mapping "pass@k" to aggregate score
    """
    scores = {f"pass@{k}": [] for k in k_values}
    
    for instance_results in results:
        attempts = instance_results.get("attempts", [])
        n = len(attempts)
        c = sum(1 for a in attempts if is_correct_fn(a))
        
        for k in k_values:
            if n >= k:
                score = calculate_pass_at_k(n, c, k)
                scores[f"pass@{k}"].append(score)
    
    # Average across instances
    return {
        metric: sum(vals) / len(vals) if vals else 0.0
        for metric, vals in scores.items()
    }


def compute_error_distribution(results: list[dict]) -> dict[str, int]:
    """
    Compute distribution of error types across results.
    
    Args:
        results: List of result dicts
        
    Returns:
        Dict mapping error type to count
    """
    error_counts = {
        "success": 0,
        "syntax_error": 0,
        "api_error": 0,
        "empty_patch": 0,
        "other": 0,
    }
    
    for instance_results in results:
        for attempt in instance_results.get("attempts", []):
            if attempt.get("success"):
                error_counts["success"] += 1
            elif not attempt.get("generated_patch"):
                error_counts["empty_patch"] += 1
            elif "SyntaxError" in str(attempt.get("error", "")):
                error_counts["syntax_error"] += 1
            elif "API" in str(attempt.get("error", "")):
                error_counts["api_error"] += 1
            else:
                error_counts["other"] += 1
    
    return error_counts
