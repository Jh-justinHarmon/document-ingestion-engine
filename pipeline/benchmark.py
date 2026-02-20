#!/usr/bin/env python3
"""
Benchmark Stage - Strategy Comparison

Deterministic accuracy scoring and strategy selection.
Compares baseline (raw) vs optimized strategies.
"""

from dataclasses import dataclass
from .metrics import get_canonical_token_count, estimate_cost


@dataclass
class BenchmarkResult:
    """Result from benchmark comparison."""
    strategy: str
    accuracy: float
    cost: float


def run_benchmark(raw_text: str, optimized_text: str, model: str = "gpt-4o-mini") -> BenchmarkResult:
    """
    Run deterministic benchmark comparing strategies.
    
    Args:
        raw_text: Original uncompressed text
        optimized_text: Compressed/optimized text
        model: Model to use for cost estimation
    
    Returns:
        BenchmarkResult with chosen strategy, accuracy, and cost
    """
    # Count tokens for both versions
    raw_tokens = get_canonical_token_count(raw_text, model)
    optimized_tokens = get_canonical_token_count(optimized_text, model)
    
    # Calculate token loss ratio
    token_loss_ratio = 1 - (optimized_tokens / raw_tokens) if raw_tokens > 0 else 0
    
    # Strategy 1: Baseline (raw text)
    baseline_accuracy = 1.0  # Full fidelity
    baseline_cost = estimate_cost(raw_tokens, 0, model)
    
    # Strategy 2: Optimized
    # Accuracy = 1 - token_loss_ratio (assumes proportional information retention)
    optimized_accuracy = round(1 - token_loss_ratio, 3)
    optimized_cost = estimate_cost(optimized_tokens, 0, model)
    
    # Choose strategy: optimized if accuracy >= 0.90, else baseline
    if optimized_accuracy >= 0.90:
        chosen_strategy = "optimized"
        chosen_accuracy = optimized_accuracy
        chosen_cost = optimized_cost
    else:
        chosen_strategy = "baseline"
        chosen_accuracy = baseline_accuracy
        chosen_cost = baseline_cost
    
    return BenchmarkResult(
        strategy=chosen_strategy,
        accuracy=chosen_accuracy,
        cost=chosen_cost
    )
