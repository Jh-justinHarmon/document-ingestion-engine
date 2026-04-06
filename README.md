# Document Ingestion Engine

## Why I Built This

Most LLM demos show that something works once. I wanted to build a pipeline that measures whether it works consistently, cheaply, and reproducibly. This project is a deterministic document ingestion pipeline designed to measure token cost, compression, and accuracy across runs, not just produce output.

The goal was to treat LLM ingestion like an engineering system, not a prompt.

---

## What This Is

A deterministic, multi-stage document ingestion pipeline that:

- Reduces tokens before LLM processing  
- Measures cost deltas from optimization  
- Benchmarks extraction accuracy  
- Produces reproducible run artifacts  
- Generates a deterministic run ID for traceability  

This CLI demo represents a simplified 5-stage subset of a larger 8-stage ingestion architecture.

---

## What It Demonstrates

This repo shows how I think about LLM systems:

- Token reduction and cost control  
- Deterministic processing  
- Measurement over intuition  
- Accuracy benchmarking  
- Reproducible artifacts  
- Pipeline architecture (not just API calls)  

The model is interchangeable. The system around it is the important part.

---

## How to Run the Demo

```bash
python ingest.py sample_docs/resume_sample.pdf
```

---

## Example Output

The pipeline prints:

- Stage execution timing  
- Raw vs optimized token counts  
- Compression %  
- Cost delta  
- Accuracy score  
- Deterministic run ID  
- Output artifact path  

The point is to see whether optimization actually reduces cost and whether accuracy holds after compression.

---

## Architecture (High Level)

The full system is an 8-stage deterministic ingestion pipeline designed for determinism, measurement, and reproducibility.

The CLI demo runs a simplified 5-stage version, but the architecture supports full pipeline benchmarking and artifact generation for each stage.

---

## Repo Structure

If you're reviewing the code, start here:

- `ingest.py` — CLI entry point and pipeline runner  
- `stages/` — Individual pipeline stages  
- `metrics/` — Token, cost, and accuracy measurement  
- `artifacts/` — Reproducible run outputs  
- `sample_docs/` — Example input documents  

---

## What This Project Is (In One Sentence)

I don’t treat LLM calls as the system. The system is everything around the model: cost, accuracy, determinism, and reproducibility. This project is a small version of that idea.
