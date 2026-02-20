#!/usr/bin/env python3
"""
Document Ingestion CLI

Deterministic document ingestion with token optimization and cost measurement.

Usage:
    python ingest.py sample_docs/resume.pdf --type resume
    python ingest.py transcript.txt --type transcript --model gpt-4o
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline.runner import run_pipeline


def format_cli_card(results: dict) -> str:
    """
    Format pipeline results as a CLI card.
    
    Args:
        results: Pipeline results dictionary
    
    Returns:
        Formatted string for terminal output
    """
    lines = []
    
    # Header
    lines.append(f"\nRun ID: {results['run_id']}\n")
    
    # Stage results
    for i, stage in enumerate(results['stage_results'], 1):
        stage_name = stage['stage_name']
        duration = stage['duration_ms']
        
        # Format stage line with dotted spacing
        stage_line = f"[Stage {i}: {stage_name}]"
        dots = "." * (40 - len(stage_line))
        lines.append(f"{stage_line} {dots} {duration:.1f}ms")
        
        # Add metrics if present
        if stage['metrics']:
            for key, value in stage['metrics'].items():
                if key == 'compression_pct':
                    lines.append(f"    compression: {value:.1f}%")
                elif key.startswith('cost_'):
                    lines.append(f"    {key}: ${value:.6f}")
                elif 'tokens' in key:
                    lines.append(f"    {key}: {value:,}")
                else:
                    lines.append(f"    {key}: {value}")
    
    # Summary
    lines.append("\n" + "=" * 50)
    lines.append("SUMMARY")
    lines.append("=" * 50)
    
    summary = results['summary']
    lines.append(f"Raw tokens:        {summary['raw_tokens']:,}")
    lines.append(f"Optimized tokens:  {summary['optimized_tokens']:,}")
    lines.append(f"Compression:       {summary['compression_pct']:.1f}%")
    lines.append(f"Cost (raw):        ${summary['cost_raw']:.6f}")
    lines.append(f"Cost (optimized):  ${summary['cost_optimized']:.6f}")
    lines.append(f"Cost saved:        ${summary['cost_saved']:.6f}")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Document Ingestion Engine - Deterministic token optimization"
    )
    parser.add_argument(
        "file_path",
        help="Path to document to ingest"
    )
    parser.add_argument(
        "--type",
        default="document",
        help="Document type (e.g., resume, transcript, document)"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        help="Model to use for token counting (default: gpt-4o-mini)"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)
    
    # Run pipeline
    try:
        results = run_pipeline(
            file_path=args.file_path,
            doc_type=args.type,
            model=args.model
        )
    except Exception as e:
        print(f"Error: Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print CLI card
    print(format_cli_card(results))
    
    # Write JSON artifact
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / f"run_{results['run_id']}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
