# Redrob AI-Recruiter — Intelligent Candidate Discovery & Ranking

Multi-stage retrieval and ranking pipeline for the Redrob India Runs Data & AI Challenge. Ranks 100,000+ candidates for a Senior AI Engineer role using dense semantic search, knowledge graphs, lexical expansion, and cross-encoder re-ranking — all on CPU, under 60 seconds.

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure you have Python 3.11+ installed. All models are bundled in the `./models/` directory for offline execution.

## How to Reproduce the Submission

### Step 1: Pre-compute Embeddings (one-time, ~80 minutes)

This step embeds all 100,000 candidate profiles using BGE-Small and saves them to disk. It only needs to be run once.

```bash
python precompute.py --candidates ./candidates.jsonl --out ./embeddings_full.npz
```

### Step 2: Run the Ranker (under 60 seconds)

Once embeddings are pre-computed, run the ranking pipeline to generate the final `submission.csv`:

```bash
python rank.py --candidates ./candidates.jsonl --embeddings ./embeddings_full.npz --out ./submission.csv
```

This produces a CSV with the top 100 candidates, their scores, and grounded reasoning.

## Pipeline Architecture

1. **Stage 1 — Dense Retrieval (BGE-Small):** Cosine similarity narrows 100K candidates to the top 3,000.
2. **Stage 2 — SkillRank (PageRank):** A bipartite candidate-skill graph with Personalized PageRank weighted toward JD-required skills.
3. **Stage 3 — Lexical Expansion (BM25):** Pseudo-relevance feedback expands the query with terms from top-ranked candidates, then re-scores via BM25.
4. **Stage 4 — Semantic Re-ranking (Cross-Encoder):** The top 300 candidates from the blended Stage 1–3 score are re-ranked by a Cross-Encoder that reads the JD and candidate profile simultaneously.
5. **Stage 5 — Feature Engineering:** Career evidence bonuses, domain penalties, honeypot detection, and recruiter response rate multipliers produce the final score.

## Validation

```bash
python val5.py              # Verifies reasoning is grounded (company/title match)
python val4.py              # Checks format, honeypots, tie-breaking, Ela Singh
python validate_submission.py submission.csv  # Official organizer format check
```

## Key Files

| File | Purpose |
|---|---|
| `rank.py` | Main ranking pipeline — produces `submission.csv` |
| `precompute.py` | Pre-computes BGE embeddings for all candidates |
| `features.py` | Feature extraction, honeypot detection, domain penalties |
| `requirements.txt` | Python dependencies |
| `submission.csv` | Final output (top 100 ranked candidates) |
| `submission_metadata.yaml` | Metadata for Stage 3 reproducibility |
| `models/` | Local model weights (BGE-Small, MiniLM Cross-Encoder) |

## Web Dashboard (Optional)

A visual dashboard to explore the top 100 candidates:

```bash
python server.py
# Open http://localhost:5000
```
