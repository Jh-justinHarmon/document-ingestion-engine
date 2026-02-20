#!/usr/bin/env python3
"""
Token Counting and Cost Measurement

Provides deterministic token counting and cost estimation for LLM operations.
Supports multiple model families with configurable pricing.

Version: 1.0.0
"""

import os
from functools import lru_cache
from typing import Optional, Tuple


# Model family -> encoder mapping
ENCODERS = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5": "cl100k_base",
    "claude": "cl100k_base",      # Approximate
    "gemini": "cl100k_base",      # Approximate
    "mistral": "cl100k_base",     # Approximate
}

# Cost per 1K tokens (input/output) by model
# Can be overridden via environment variable INPUT_COST_PER_1K
COSTS_PER_1K = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "gemini-pro": (0.00025, 0.0005),
    "mistral-large": (0.004, 0.012),
    "mistral-small": (0.001, 0.003),
}


@lru_cache(maxsize=1000)
def get_canonical_token_count(text: str, model: str = "gpt-4o") -> int:
    """
    Get canonical token count using the correct tokenizer for the model.
    Results are cached to avoid redundant computation.
    
    Args:
        text: Text to tokenize
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet")
    
    Returns:
        Token count
    """
    try:
        import tiktoken
        
        # Get encoder for model family
        model_family = model.split("-")[0]
        encoder_name = ENCODERS.get(model_family, "cl100k_base")
        encoder = tiktoken.get_encoding(encoder_name)
        return len(encoder.encode(text))
    except ImportError:
        # Fallback: approximate 4 chars per token
        return len(text) // 4
    except Exception:
        # Fallback: approximate 4 chars per token
        return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    """
    Estimate cost based on token counts and model.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
    
    Returns:
        Estimated cost in USD
    """
    # Find matching model or default
    costs = COSTS_PER_1K.get(model)
    if not costs:
        # Try partial match
        for key, val in COSTS_PER_1K.items():
            if key in model or model in key:
                costs = val
                break
    
    if not costs:
        # Default to gpt-4o-mini pricing
        costs = COSTS_PER_1K["gpt-4o-mini"]
    
    input_cost = (input_tokens / 1000) * costs[0]
    output_cost = (output_tokens / 1000) * costs[1]
    return input_cost + output_cost


def count_and_cost(text: str, model: str = "gpt-4o", 
                   is_input: bool = True) -> Tuple[int, float]:
    """
    Count tokens and estimate cost for a single text.
    
    Args:
        text: Text to analyze
        model: Model name
        is_input: True if input tokens, False if output
    
    Returns:
        Tuple of (token_count, cost_usd)
    """
    tokens = get_canonical_token_count(text, model)
    
    if is_input:
        cost = estimate_cost(tokens, 0, model)
    else:
        cost = estimate_cost(0, tokens, model)
    
    return (tokens, cost)
