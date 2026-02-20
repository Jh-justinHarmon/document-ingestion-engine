# Document Ingestion Engine

Deterministic 8-stage document ingestion pipeline with:

- Token optimization (60–80% compression)
- Cost delta measurement
- Accuracy benchmarking
- Reproducible run artifacts

Designed as a falsifiable engineering artifact for interview demonstration.

## Demo

```bash
python ingest.py sample_docs/resume_sample.pdf
```

Displays:

- Stage execution timing
- Raw vs optimized token counts
- Compression %
- Cost delta
- Accuracy score
- Deterministic run ID
- Output artifact path