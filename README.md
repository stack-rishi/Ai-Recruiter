<div align="center">

# 🎯 AI-Recruiter — Intelligent Candidate Discovery & Ranking

**Multi-Stage Retrieval Pipeline for the Redrob India Runs Data & AI Challenge**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CPU Only](https://img.shields.io/badge/Compute-CPU%20Only-orange?style=for-the-badge)]()
[![Runtime](https://img.shields.io/badge/Runtime-~14s-brightgreen?style=for-the-badge)]()

*Ranks 100,000+ candidates for a Senior AI Engineer role using dense semantic search, knowledge graphs, lexical expansion, and cross-encoder re-ranking — all on CPU, in under 60 seconds.*

<br>

[![🚀 Live Demo — Try It Now](https://img.shields.io/badge/🚀_LIVE_DEMO_—_Try_It_Now-FF6F00?style=for-the-badge&logoColor=white&logo=streamlit)](https://huggingface.co/spaces/rishi-sharma/ai-recruiter)

<br>

[Architecture](#-architecture) · [Quick Start](#-quick-start) · [Results](#-results)

</div>

---

## 📋 Overview

This project implements a **production-grade, multi-stage candidate ranking system** that processes 100,000+ candidate profiles against a detailed job description for a Senior AI Engineer role. The system is designed to:

- **Run entirely offline** — no external API calls (OpenAI, Anthropic, etc.)
- **Execute in under 60 seconds** on a standard CPU machine
- **Produce grounded, non-hallucinated reasoning** for every ranked candidate
- **Detect and filter honeypot candidates** with impossible profile data

> **Challenge Constraint:** 5-minute wall-clock, 16 GB RAM, CPU-only, no network. Our pipeline finishes in ~14 seconds with comfortable margin.

---

## 🏗 Architecture

The pipeline uses a **four-stage funnel** that progressively narrows and refines the candidate pool:

```
100,000 candidates
       │
       ▼
┌──────────────────────────────────┐
│  Stage 1 — Dense Retrieval       │  BGE-Small embeddings + cosine similarity
│  100K → 3,000 candidates         │  Pre-computed offline (~80 min)
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Stage 2 — SkillRank Graph       │  Bipartite candidate-skill graph
│  Personalized PageRank           │  Weighted toward JD-required skills
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Stage 3 — Lexical Expansion     │  BM25 with pseudo-relevance feedback
│  Two-pass query expansion        │  Captures keyword-level signals
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Stage 4 — Cross-Encoder         │  ms-marco-MiniLM-L-6-v2
│  3,000 → 300 → Top 100          │  Deep semantic interaction scoring
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Stage 5 — Feature Engineering   │  Career evidence bonuses
│  + Honeypot Detection            │  Domain penalties, response rate
└──────────────┬───────────────────┘
               │
               ▼
        submission.csv
      (Top 100 ranked)
```

### Why Multi-Stage?

Pure embedding similarity misses candidates whose profiles use different vocabulary than the JD. A candidate who "built a recommendation system at a product company" is a strong fit even without mentioning "RAG" or "Pinecone." Each stage captures a different signal:

| Stage | Signal Type | What It Catches |
|-------|------------|-----------------|
| Dense Retrieval | Semantic similarity | Candidates whose overall profile aligns with the JD |
| PageRank | Skill graph topology | Candidates connected to JD-critical skills through adjacent skills |
| BM25 | Lexical overlap | Exact keyword matches the embedding model might normalize away |
| Cross-Encoder | Deep interaction | Nuanced JD↔candidate alignment that bi-encoders miss |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

> Python 3.11+ required. All ML models are bundled in `./models/` for fully offline execution.

### Step 1 — Pre-compute Embeddings *(one-time, ~80 minutes)*

```bash
python precompute.py --candidates ./candidates.jsonl --out ./embeddings_full.npz
```

This embeds all 100K candidate profiles using [BGE-Small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) and saves them to disk. Only needs to run once.

### Step 2 — Run the Ranker *(~14 seconds)*

```bash
python rank.py --candidates ./candidates.jsonl --embeddings ./embeddings_full.npz --out ./submission.csv
```

Produces `submission.csv` with the top 100 candidates, scores, and grounded reasoning.

### Docker *(Alternative)*

```bash
docker build -t ai-recruiter .
docker run -v $(pwd):/app ai-recruiter
```

---

## 📊 Results

### Validation Summary

```
=== VALIDATION SUMMARY ===
  [PASS] | row_count             — Exactly 100 data rows
  [PASS] | ranks_valid           — Ranks 1–100, each exactly once
  [PASS] | no_duplicate_ids      — 100 unique candidate IDs
  [PASS] | monotonic_scores      — Scores non-increasing with rank
  [PASS] | tie_breaking_valid    — Ties broken by candidate_id ascending
  [PASS] | no_h1_honeypots       — 0 impossible-date honeypots
  [PASS] | no_template_phrase    — No templated reasoning
  [PASS] | ela_singh_top15       — Star candidate ranked #6
  [PASS] | reasoning_has_concerns — 80/100 entries cite honest concerns

Overall: READY TO SUBMIT
```

### Reasoning Quality

Every reasoning string references **actual profile data** — real titles, real companies, real skill durations. No hallucination.

| Rank | Candidate | Reasoning (excerpt) |
|------|-----------|-------------------|
| 1 | Machine Learning Engineer @ Unacademy | *"Top-tier match... Production background supported by Milvus (94 months, expert). Notice period is 120 days..."* |
| 6 | Recommendation Systems Engineer @ Swiggy | *"Top-tier match... Production background supported by Pinecone (88 months, expert). 60-day notice period..."* |
| 100 | — | *"Potential fit... Limited direct retrieval/ranking evidence."* |

### Key Design Decisions

- **Rank-banded tone**: Top 10 get "Top-tier match", ranks 11–30 get "Strong candidate", 31–70 get "Solid fit", 71–100 get "Potential fit"
- **Honest concerns**: Notice periods, inactivity, relocation status are surfaced, not hidden
- **Career evidence bonus**: Candidates with production retrieval/ranking work described in career history get score boosts — not from the skills section
- **Domain penalty**: Candidates whose primary expertise is computer vision/speech (which the JD explicitly excludes) are down-weighted

---

## 🔒 Honeypot Detection

The dataset contains ~80 honeypot candidates with subtly impossible profiles. Our system detects two patterns:

| Type | Detection Logic | Result |
|------|----------------|--------|
| **H1** | `last_active_date < signup_date` — impossible timeline | Score → 0 |
| **H2** | `salary_min - salary_max > 2 LPA` — inverted salary range | Score → 0 |

**0 honeypots** appear in our top 100.

---

## 📁 Repository Structure

```
.
├── rank.py                    # Main ranking pipeline — produces submission.csv
├── precompute.py              # Pre-computes BGE-Small embeddings (offline step)
├── features.py                # Feature extraction, honeypot detection, domain penalties
├── requirements.txt           # Python dependencies (pinned versions)
├── submission.csv             # Final output — top 100 ranked candidates
├── submission_metadata.yaml   # Metadata for Stage 3 reproducibility
├── Dockerfile                 # Containerized reproduction environment
├── val4.py                    # Format + content validation script
├── val5.py                    # Reasoning factual integrity checker
└── README.md                  # You are here
```

---

## ✅ Validation

```bash
# Official format checker (from hackathon bundle)
python validate_submission.py submission.csv

# Reasoning grounding check (company/title match)
python val5.py

# Full format + honeypot + tie-breaking + Ela Singh check  
python val4.py
```

---

## 🛠 Technical Details

### Models Used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| `BAAI/bge-small-en-v1.5` | Dense candidate embeddings | 33M params | [HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Semantic re-ranking | 22M params | [HuggingFace](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |

### Score Composition

The final score for each candidate is:

```
base_score = 0.50 × cross_encoder + 0.25 × bm25 + 0.25 × pagerank
score = (base_score + career_evidence_bonus) × response_rate_mult × domain_penalty
```

### Runtime Breakdown

| Component | Time |
|-----------|------|
| Load embeddings + model init | ~3s |
| Dense retrieval (100K → 3K) | ~1s |
| PageRank graph | ~2s |
| BM25 (2 passes) | ~1s |
| Cross-encoder (300 candidates) | ~5s |
| Scoring + reasoning | ~2s |
| **Total** | **~14s** |

---

<div align="center">

**Built for the [Redrob India Runs Data & AI Challenge](https://redrob.ai)**

</div>
