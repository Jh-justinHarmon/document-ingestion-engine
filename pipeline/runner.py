#!/usr/bin/env python3
"""
Pipeline Stage Runner

Executes document ingestion pipeline stages with timing and metrics.
Stages: Intake → Parse → Normalize → Optimize → Output
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

from .optimizer import optimize_content_sync, OptimizerConfig
from .metrics import get_canonical_token_count, estimate_cost


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    duration_ms: float
    metrics: Dict[str, Any]


def run_pipeline(file_path: str, doc_type: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Run complete ingestion pipeline with timing.
    
    Args:
        file_path: Path to input document
        doc_type: Document type (e.g., "resume", "transcript")
        model: Model to use for token counting and optimization
    
    Returns:
        Pipeline results dictionary with stage timings and metrics
    """
    from run_id import generate_run_id
    
    stage_results = []
    
    # Generate deterministic run ID
    run_id = generate_run_id(file_path, model)
    
    # =========================================================================
    # STAGE 1: INTAKE
    # =========================================================================
    start = time.perf_counter()
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    raw_bytes = file_path_obj.read_bytes()
    file_size = len(raw_bytes)
    
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if duration_ms == 0.0:
        duration_ms = 0.1
    stage_results.append(StageResult(
        stage_name="Intake",
        duration_ms=duration_ms,
        metrics={"file_size_bytes": file_size}
    ))
    
    # =========================================================================
    # STAGE 2: PARSE
    # =========================================================================
    start = time.perf_counter()
    
    # Simple text extraction (UTF-8 decode for now)
    # In production, this would use PDF parsers, etc.
    try:
        raw_text = raw_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        raise ValueError(f"Failed to parse file: {e}")
    
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if duration_ms == 0.0:
        duration_ms = 0.1
    stage_results.append(StageResult(
        stage_name="Parse",
        duration_ms=duration_ms,
        metrics={"chars_extracted": len(raw_text)}
    ))
    
    # =========================================================================
    # STAGE 3: NORMALIZE
    # =========================================================================
    start = time.perf_counter()
    
    # Strip excessive whitespace
    lines = raw_text.split('\n')
    normalized_lines = [line.strip() for line in lines if line.strip()]
    normalized_text = '\n'.join(normalized_lines)
    
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if duration_ms == 0.0:
        duration_ms = 0.1
    stage_results.append(StageResult(
        stage_name="Normalize",
        duration_ms=duration_ms,
        metrics={"chars_normalized": len(normalized_text)}
    ))
    
    # =========================================================================
    # STAGE 4: OPTIMIZE
    # =========================================================================
    start = time.perf_counter()
    
    # Get raw token count
    raw_tokens = get_canonical_token_count(normalized_text, model)
    
    # Run optimization
    config = OptimizerConfig()
    optimized = optimize_content_sync(
        content=normalized_text,
        filename=file_path_obj.name,
        config=config
    )
    
    optimized_tokens = optimized.stats.compressed_tokens
    
    # Calculate compression percentage from token counts
    compression_pct = round((1 - (optimized_tokens / raw_tokens)) * 100, 1) if raw_tokens > 0 else 0.0
    
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if duration_ms == 0.0:
        duration_ms = 0.1
    stage_results.append(StageResult(
        stage_name="Optimize",
        duration_ms=duration_ms,
        metrics={
            "raw_tokens": raw_tokens,
            "optimized_tokens": optimized_tokens,
            "compression_pct": compression_pct,
            "strategy": optimized.stats.strategy_used.value
        }
    ))
    
    # =========================================================================
    # STAGE 5: COST ANALYSIS
    # =========================================================================
    start = time.perf_counter()
    
    # Calculate costs
    cost_raw = estimate_cost(raw_tokens, 0, model)
    cost_optimized = estimate_cost(optimized_tokens, 0, model)
    cost_saved = cost_raw - cost_optimized
    
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if duration_ms == 0.0:
        duration_ms = 0.1
    stage_results.append(StageResult(
        stage_name="Cost Analysis",
        duration_ms=duration_ms,
        metrics={
            "cost_raw": cost_raw,
            "cost_optimized": cost_optimized,
            "cost_saved": cost_saved
        }
    ))
    
    # =========================================================================
    # STAGE 6: BENCHMARK
    # =========================================================================
    start = time.perf_counter()
    
    from .benchmark import run_benchmark
    
    # Run benchmark comparison
    benchmark_result = run_benchmark(normalized_text, optimized.compressed, model)
    
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if duration_ms == 0.0:
        duration_ms = 0.1
    stage_results.append(StageResult(
        stage_name="Benchmark",
        duration_ms=duration_ms,
        metrics={
            "accuracy": benchmark_result.accuracy,
            "strategy": benchmark_result.strategy
        }
    ))
    
    # =========================================================================
    # ASSEMBLE FINAL RESULTS
    # =========================================================================
    
    return {
        "run_id": run_id,
        "file_path": file_path,
        "doc_type": doc_type,
        "model": model,
        "stage_results": [
            {
                "stage_name": sr.stage_name,
                "duration_ms": sr.duration_ms,
                "metrics": sr.metrics
            }
            for sr in stage_results
        ],
        "summary": {
            "raw_tokens": raw_tokens,
            "optimized_tokens": optimized_tokens,
            "compression_pct": compression_pct,
            "cost_raw": cost_raw,
            "cost_optimized": cost_optimized,
            "cost_saved": cost_saved,
            "benchmark_accuracy": benchmark_result.accuracy,
            "benchmark_strategy": benchmark_result.strategy
        }
    }
