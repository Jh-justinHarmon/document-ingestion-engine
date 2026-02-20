#!/usr/bin/env python3
"""
Deterministic Run ID Generation

Generates reproducible run identifiers based on file content and model.
Format: YYYYMMDD_HHMM-<hash>

Example: 20260219_2145-a3f8c2
"""

import hashlib
from datetime import datetime
from pathlib import Path


def generate_run_id(file_path: str, model: str) -> str:
    """
    Generate deterministic run ID from file content and model.
    
    Args:
        file_path: Path to input file
        model: Model name used for processing
    
    Returns:
        Run ID string in format: YYYYMMDD_HHMM-<hash>
    
    Example:
        >>> generate_run_id("resume.pdf", "gpt-4o-mini")
        '20260219_2145-a3f8c2'
    """
    # Read file contents
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_contents = file_path_obj.read_bytes()
    
    # Generate hash from file contents + model
    hash_input = file_contents + model.encode('utf-8')
    file_hash = hashlib.sha256(hash_input).hexdigest()[:6]
    
    # Generate timestamp (rounded to minute)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    
    # Combine timestamp + hash
    run_id = f"{timestamp}-{file_hash}"
    
    return run_id
