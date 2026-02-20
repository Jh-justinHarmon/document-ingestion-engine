#!/usr/bin/env python3
"""
Token Optimizer - Context Compression for Cost Reduction

Implements preprocessing for long-context inputs:
- Transcripts, web pages, documents → compressed summaries
- Reduces token count by 60-80% while preserving key information
- Routes to appropriate compression strategy based on content type

Cost savings: ~$0.01-0.05 per long document by avoiding full-context LLM calls
"""

import asyncio
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .metrics import estimate_cost

# Rough token estimation (4 chars ≈ 1 token)
CHARS_PER_TOKEN = 4


class ContentType(Enum):
    """Types of content with different compression strategies."""
    TRANSCRIPT = "transcript"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    CODE = "code"
    EMAIL = "email"
    CHAT = "chat"
    UNKNOWN = "unknown"


class CompressionStrategy(Enum):
    """Compression strategies based on content type and size."""
    NONE = "none"                    # Small content, no compression needed
    TRUNCATE = "truncate"            # Simple truncation with ellipsis
    EXTRACTIVE = "extractive"        # Extract key sentences (no LLM)
    ABSTRACTIVE = "abstractive"      # LLM-based summarization
    HIERARCHICAL = "hierarchical"    # Chunk → summarize → merge


@dataclass
class TokenStats:
    """Token statistics for a piece of content."""
    original_chars: int
    original_tokens: int
    compressed_chars: int
    compressed_tokens: int
    compression_ratio: float  # 0.0 = no compression, 1.0 = fully compressed
    strategy_used: CompressionStrategy
    cost_saved: float  # Estimated $ saved
    
    def to_dict(self) -> Dict:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": f"{self.compression_ratio:.0%}",
            "strategy": self.strategy_used.value,
            "cost_saved": f"${self.cost_saved:.4f}"
        }


@dataclass
class OptimizedContent:
    """Result of token optimization."""
    original: str
    compressed: str
    content_type: ContentType
    stats: TokenStats
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "content_type": self.content_type.value,
            "compressed_preview": self.compressed[:500] + "..." if len(self.compressed) > 500 else self.compressed,
            "stats": self.stats.to_dict(),
            "metadata": self.metadata
        }


@dataclass
class OptimizerConfig:
    """Configuration for token optimizer."""
    # Thresholds (in tokens)
    no_compress_threshold: int = 500      # Below this, don't compress
    truncate_threshold: int = 2000        # Below this, simple truncate
    extractive_threshold: int = 8000      # Below this, extractive
    # Above extractive_threshold → abstractive or hierarchical
    
    # Target sizes
    target_ratio: float = 0.3             # Target 30% of original
    max_output_tokens: int = 2000         # Never exceed this
    
    # Cost estimation (per 1K tokens, gpt-4o-mini input)
    input_cost_per_1k: float = 0.00015
    
    # LLM settings
    summarization_model: str = "gpt-4o-mini"
    chunk_size: int = 3000                # Tokens per chunk for hierarchical


# =============================================================================
# Content Type Detection
# =============================================================================

def detect_content_type(content: str, filename: Optional[str] = None) -> ContentType:
    """Detect content type from content and optional filename."""
    content_lower = content.lower()[:2000]  # Check first 2000 chars
    
    # Check filename first
    if filename:
        filename_lower = filename.lower()
        if any(ext in filename_lower for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.go']):
            return ContentType.CODE
        if any(ext in filename_lower for ext in ['.md', '.txt', '.doc', '.pdf']):
            return ContentType.DOCUMENT
        if 'transcript' in filename_lower or 'otter' in filename_lower:
            return ContentType.TRANSCRIPT
    
    # Transcript indicators
    transcript_signals = [
        r'\d{1,2}:\d{2}',           # Timestamps like 0:00 or 12:34
        r'speaker \d',               # Speaker labels
        r'\[.*?\]:',                 # [Name]: format
        r'um+|uh+|like,',           # Filler words
        r'yeah|okay|right\?',       # Conversational markers
    ]
    transcript_score = sum(1 for pattern in transcript_signals 
                          if re.search(pattern, content_lower))
    if transcript_score >= 3:
        return ContentType.TRANSCRIPT
    
    # Email indicators
    email_signals = ['from:', 'to:', 'subject:', 'sent:', 'cc:', 'bcc:']
    if sum(1 for s in email_signals if s in content_lower) >= 2:
        return ContentType.EMAIL
    
    # Web page indicators
    web_signals = ['<html', '<div', '<script', 'http://', 'https://', 'cookie', 'privacy policy']
    if sum(1 for s in web_signals if s in content_lower) >= 2:
        return ContentType.WEB_PAGE
    
    # Code indicators
    code_signals = ['def ', 'function ', 'class ', 'import ', 'const ', 'let ', 'var ']
    if sum(1 for s in code_signals if s in content_lower) >= 3:
        return ContentType.CODE
    
    # Chat indicators
    chat_signals = [r'^\w+:', r'^\[\d{1,2}:\d{2}\]', 'lol', 'brb', 'omg']
    if sum(1 for pattern in chat_signals if re.search(pattern, content_lower)) >= 2:
        return ContentType.CHAT
    
    # Default to document
    return ContentType.DOCUMENT


def estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // CHARS_PER_TOKEN


# =============================================================================
# Compression Strategies
# =============================================================================

def compress_truncate(content: str, target_tokens: int) -> str:
    """Simple truncation with smart boundary detection."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    
    if len(content) <= target_chars:
        return content
    
    # Try to truncate at paragraph boundary
    truncated = content[:target_chars]
    
    # Find last paragraph break
    last_para = truncated.rfind('\n\n')
    if last_para > target_chars * 0.7:  # At least 70% of target
        truncated = truncated[:last_para]
    else:
        # Find last sentence
        last_sentence = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )
        if last_sentence > target_chars * 0.8:
            truncated = truncated[:last_sentence + 1]
    
    return truncated.strip() + "\n\n[... content truncated ...]"


def compress_extractive(content: str, target_tokens: int, content_type: ContentType) -> str:
    """Extract key sentences without LLM."""
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    if not sentences:
        return compress_truncate(content, target_tokens)
    
    # Score sentences based on content type
    scored = []
    for i, sentence in enumerate(sentences):
        score = 0
        sentence_lower = sentence.lower()
        
        # Position bonus (first and last sentences often important)
        if i < 3:
            score += 3 - i
        if i >= len(sentences) - 2:
            score += 1
        
        # Length penalty (very short sentences less useful)
        if len(sentence) < 20:
            score -= 2
        elif len(sentence) > 200:
            score -= 1
        
        # Content-type specific scoring
        if content_type == ContentType.TRANSCRIPT:
            # Prioritize action items, decisions, questions
            if any(w in sentence_lower for w in ['decide', 'agreed', 'will do', 'action', 'next step']):
                score += 3
            if '?' in sentence:
                score += 1
            # Deprioritize filler
            if any(w in sentence_lower for w in ['um', 'uh', 'like,', 'you know']):
                score -= 2
                
        elif content_type == ContentType.DOCUMENT:
            # Prioritize headers, key terms
            if sentence.isupper() or sentence.startswith('#'):
                score += 2
            if any(w in sentence_lower for w in ['important', 'key', 'critical', 'must', 'should']):
                score += 2
                
        elif content_type == ContentType.EMAIL:
            # Prioritize requests, deadlines
            if any(w in sentence_lower for w in ['please', 'request', 'deadline', 'by', 'asap']):
                score += 2
            if any(w in sentence_lower for w in ['fyi', 'regards', 'thanks']):
                score -= 1
        
        scored.append((score, i, sentence))
    
    # Sort by score, then by position
    scored.sort(key=lambda x: (-x[0], x[1]))
    
    # Select top sentences up to target
    selected = []
    current_tokens = 0
    
    for score, idx, sentence in scored:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens <= target_tokens:
            selected.append((idx, sentence))
            current_tokens += sentence_tokens
    
    # Re-order by original position
    selected.sort(key=lambda x: x[0])
    
    # Join with context markers
    result = " [...] ".join(s for _, s in selected)
    
    return result


async def compress_abstractive(
    content: str,
    target_tokens: int,
    content_type: ContentType,
    model: str = "gpt-4o-mini"
) -> str:
    """LLM-based summarization."""
    import openai
    
    # Build prompt based on content type
    type_instructions = {
        ContentType.TRANSCRIPT: "Extract key decisions, action items, and main discussion points. Preserve speaker attributions for important statements.",
        ContentType.DOCUMENT: "Summarize the main points, preserving structure and key details.",
        ContentType.WEB_PAGE: "Extract the main content, ignoring navigation, ads, and boilerplate.",
        ContentType.EMAIL: "Summarize the request/information and any action items or deadlines.",
        ContentType.CODE: "Describe what the code does, key functions, and any important patterns.",
        ContentType.CHAT: "Summarize the conversation, key decisions, and any action items.",
        ContentType.UNKNOWN: "Provide a concise summary of the main content."
    }
    
    prompt = f"""Summarize the following {content_type.value} in approximately {target_tokens} tokens.

Instructions: {type_instructions.get(content_type, type_instructions[ContentType.UNKNOWN])}

Content:
{content[:12000]}

Summary:"""

    try:
        client = openai.OpenAI()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise summarizer. Be concise but preserve key information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=min(target_tokens + 100, 2000),
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to extractive on error
        print(f"Abstractive compression failed: {e}, falling back to extractive", file=sys.stderr)
        return compress_extractive(content, target_tokens, content_type)


async def compress_hierarchical(
    content: str,
    target_tokens: int,
    content_type: ContentType,
    chunk_size: int = 3000,
    model: str = "gpt-4o-mini"
) -> str:
    """Chunk → summarize each → merge summaries."""
    # Split into chunks
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if current_tokens + para_tokens > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    if len(chunks) <= 1:
        # Single chunk, just use abstractive
        return await compress_abstractive(content, target_tokens, content_type, model)
    
    # Summarize each chunk
    chunk_target = target_tokens // len(chunks)
    summaries = []
    
    for i, chunk in enumerate(chunks):
        summary = await compress_abstractive(
            chunk,
            chunk_target,
            content_type,
            model
        )
        summaries.append(f"[Section {i+1}]\n{summary}")
    
    # Merge summaries
    merged = "\n\n".join(summaries)
    
    # If still too long, do final compression
    if estimate_tokens(merged) > target_tokens * 1.2:
        merged = await compress_abstractive(merged, target_tokens, content_type, model)
    
    return merged


# =============================================================================
# Main Optimizer Class
# =============================================================================

class TokenOptimizer:
    """
    Token Optimizer - reduces context size while preserving information.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
    
    def select_strategy(self, token_count: int) -> CompressionStrategy:
        """Select compression strategy based on token count."""
        if token_count <= self.config.no_compress_threshold:
            return CompressionStrategy.NONE
        elif token_count <= self.config.truncate_threshold:
            return CompressionStrategy.TRUNCATE
        elif token_count <= self.config.extractive_threshold:
            return CompressionStrategy.EXTRACTIVE
        else:
            return CompressionStrategy.ABSTRACTIVE
    
    def calculate_target(self, original_tokens: int) -> int:
        """Calculate target token count."""
        target = int(original_tokens * self.config.target_ratio)
        return min(target, self.config.max_output_tokens)
    
    def estimate_cost_saved(self, original_tokens: int, compressed_tokens: int) -> float:
        """Estimate cost saved by compression."""
        tokens_saved = original_tokens - compressed_tokens
        return (tokens_saved / 1000) * self.config.input_cost_per_1k
    
    async def optimize(
        self,
        content: str,
        filename: Optional[str] = None,
        force_strategy: Optional[CompressionStrategy] = None,
        content_type: Optional[ContentType] = None
    ) -> OptimizedContent:
        """
        Optimize content for token efficiency.
        
        Args:
            content: The content to optimize
            filename: Optional filename for type detection
            force_strategy: Override automatic strategy selection
            content_type: Override automatic type detection
        
        Returns:
            OptimizedContent with compressed text and stats
        """
        original_tokens = estimate_tokens(content)
        
        # Detect content type
        detected_type = content_type or detect_content_type(content, filename)
        
        # Select strategy
        strategy = force_strategy or self.select_strategy(original_tokens)
        
        # Calculate target
        target_tokens = self.calculate_target(original_tokens)
        
        # Apply compression
        if strategy == CompressionStrategy.NONE:
            compressed = content
        elif strategy == CompressionStrategy.TRUNCATE:
            compressed = compress_truncate(content, target_tokens)
        elif strategy == CompressionStrategy.EXTRACTIVE:
            compressed = compress_extractive(content, target_tokens, detected_type)
        elif strategy == CompressionStrategy.ABSTRACTIVE:
            compressed = await compress_abstractive(
                content, target_tokens, detected_type, self.config.summarization_model
            )
        elif strategy == CompressionStrategy.HIERARCHICAL:
            compressed = await compress_hierarchical(
                content, target_tokens, detected_type,
                self.config.chunk_size, self.config.summarization_model
            )
        else:
            compressed = content
        
        # Calculate stats
        compressed_tokens = estimate_tokens(compressed)
        compression_ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0
        cost_saved = self.estimate_cost_saved(original_tokens, compressed_tokens)
        
        stats = TokenStats(
            original_chars=len(content),
            original_tokens=original_tokens,
            compressed_chars=len(compressed),
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            strategy_used=strategy,
            cost_saved=cost_saved
        )
        
        return OptimizedContent(
            original=content,
            compressed=compressed,
            content_type=detected_type,
            stats=stats,
            metadata={
                "filename": filename,
                "target_tokens": target_tokens
            }
        )


# =============================================================================
# Public API
# =============================================================================

async def optimize_content(
    content: str,
    filename: Optional[str] = None,
    config: Optional[OptimizerConfig] = None
) -> OptimizedContent:
    """
    Main entry point for token optimization.
    
    Args:
        content: Text content to optimize
        filename: Optional filename for type detection
        config: Optional configuration override
    
    Returns:
        OptimizedContent with compressed text and stats
    
    Example:
        result = await optimize_content(
            content=long_transcript,
            filename="meeting_notes.txt"
        )
        print(f"Reduced from {result.stats.original_tokens} to {result.stats.compressed_tokens}")
        print(f"Saved: {result.stats.cost_saved}")
    """
    optimizer = TokenOptimizer(config)
    return await optimizer.optimize(content, filename)


def optimize_content_sync(
    content: str,
    filename: Optional[str] = None,
    config: Optional[OptimizerConfig] = None
) -> OptimizedContent:
    """Synchronous wrapper for optimize_content."""
    return asyncio.run(optimize_content(content, filename, config))
