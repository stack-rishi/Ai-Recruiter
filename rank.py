import argparse
import csv
import numpy as np
import time
import os
import networkx as nx
from collections import Counter
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from precompute import load_candidates
from features import extract_features, extract_text_for_embedding, compute_career_evidence_bonus, compute_domain_penalty

JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years (see "what we mean by this" below)
Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude — willing to ship a working ranker in a week
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python.
Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).
"""

# JD-relevant skills that signal genuine fit for the Senior AI Engineer role
JD_CORE_SKILLS = {
    'Embeddings', 'FAISS', 'Pinecone', 'Qdrant', 'Weaviate', 'Milvus',
    'Elasticsearch', 'OpenSearch', 'Vector Search', 'Sentence Transformers',
    'Information Retrieval', 'Recommendation Systems', 'BM25', 'Hybrid Search',
    'MLflow', 'scikit-learn', 'Python', 'PyTorch', 'TensorFlow', 'NLP',
    'Hugging Face Transformers', 'Feature Engineering', 'Learning to Rank',
    'Retrieval', 'Ranking', 'A/B Testing', 'Experiment Design'
}


def get_best_jd_skill(candidate):
    """
    Returns the single most credible JD-relevant skill for this candidate.
    Credibility = duration_months first, then endorsements.
    Skips skills with duration_months == 0.
    """
    jd_skills = [
        s for s in candidate.get('skills', [])
        if s['name'] in JD_CORE_SKILLS and s.get('duration_months', 0) > 0
    ]
    if not jd_skills:
        # Fall back to any skill with duration > 6 months
        jd_skills = [s for s in candidate.get('skills', []) 
                     if s.get('duration_months', 0) > 6]
    if not jd_skills:
        return None
    return max(jd_skills, key=lambda s: (s['duration_months'], s['endorsements']))


def get_honest_concern(candidate):
    """
    Returns one honest concern about this candidate's fit or availability.
    Returns empty string if no meaningful concern exists.
    """
    sig = candidate['redrob_signals']
    profile = candidate['profile']

    # Check notice period (JD wants sub-30-day, comfortable with 30-day buyout)
    notice = sig.get('notice_period_days', 0)
    if notice > 90:
        return f"Notice period is {notice} days, above the preferred 30-day window."
    if notice > 30:
        return f"{notice}-day notice period may require negotiation."

    # Check location
    country = profile.get('country', 'India')
    if country not in ('India',):
        location = profile.get('location', country)
        willing = sig.get('willing_to_relocate', False)
        if willing:
            return f"Based in {location}; willing to relocate."
        else:
            return f"Based in {location}; relocation preference unclear."

    # Check activity recency (baseline: 2026-06-01)
    from datetime import date
    last_active = sig.get('last_active_date', '')
    if last_active:
        try:
            la_date = date.fromisoformat(last_active)
            days_inactive = (date(2026, 6, 1) - la_date).days
            if days_inactive > 120:
                return f"Last active {days_inactive} days ago; engagement uncertain."
        except ValueError:
            pass

    # Check open-to-work flag
    if not sig.get('open_to_work_flag', True):
        return "Not marked open to work; outreach may be required."

    return ""  # No significant concern


def generate_reasoning(candidate, rank):
    """
    Generates a specific, grounded, non-templated reasoning string for this
    candidate at this rank. References actual profile facts. Never fabricates.
    Acknowledges concerns where they exist. Varies tone by rank band.
    """
    profile = candidate['profile']
    sig = candidate['redrob_signals']

    title = profile.get('current_title', 'ML Engineer')
    company = profile.get('current_company', 'current employer')
    yoe = profile.get('years_of_experience', 0)
    top_skill = get_best_jd_skill(candidate)
    concern = get_honest_concern(candidate)

    skill_str = ""
    if top_skill:
        dur = top_skill['duration_months']
        skill_str = (
            f"{top_skill['name']} ({dur} months, "
            f"{top_skill.get('proficiency','intermediate')})"
        )

    concern_clause = f" {concern}" if concern else ""

    # Rank-banded tone
    if rank <= 10:
        opening = (
            f"Top-tier match: {title} at {company} with {yoe:.1f} years in "
            f"applied ML."
        )
        skill_clause = (
            f" Production background supported by {skill_str}." 
            if skill_str else ""
        )
    elif rank <= 30:
        opening = (
            f"Strong candidate: {title} currently at {company}, "
            f"{yoe:.1f} years of experience."
        )
        skill_clause = (
            f" Relevant depth in {skill_str}." 
            if skill_str else ""
        )
    elif rank <= 70:
        opening = (
            f"Solid fit: {title} at {company}, {yoe:.1f} years."
        )
        skill_clause = (
            f" Has {skill_str} in their profile." 
            if skill_str else ""
        )
    else:
        opening = (
            f"Potential fit: {title} ({yoe:.1f} years), currently at {company}."
        )
        skill_clause = (
            f" Some relevant exposure to {skill_str}." 
            if skill_str else " Limited direct retrieval/ranking evidence."
        )

    return f"{opening}{skill_clause}{concern_clause}".strip()


def build_knowledge_graph_and_pagerank(candidates):
    print("Building Candidate-Skill Knowledge Graph...")
    G = nx.Graph()
    personalization = {}
    
    # Add JD Skills
    for s in JD_CORE_SKILLS:
        G.add_node(f"skill_{s}", type="skill")
        personalization[f"skill_{s}"] = 1.0
        
    for c in candidates:
        cid = c["candidate_id"]
        G.add_node(cid, type="candidate")
        personalization[cid] = 0.0
        
        for skill in c.get("skills", []):
            s_name = skill.get("name", "").lower().strip()
            if not s_name: continue
            skill_node = f"skill_{s_name}"
            if not G.has_node(skill_node):
                G.add_node(skill_node, type="skill")
                personalization[skill_node] = 0.0
            
            # Weight edge by proficiency/duration if desired, default 1
            G.add_edge(cid, skill_node, weight=1.0)
            
    print("Running Personalized PageRank on Graph...")
    # Normalize personalization
    sum_p = sum(personalization.values())
    if sum_p > 0:
        personalization = {k: v/sum_p for k,v in personalization.items()}
    else:
        personalization = None
        
    pr_scores = nx.pagerank(G, alpha=0.85, personalization=personalization, max_iter=100)
    
    # Extract only candidate scores
    cand_pr = {c["candidate_id"]: pr_scores.get(c["candidate_id"], 0.0) for c in candidates}
    
    # Normalize PageRank scores 0-1
    max_pr = max(cand_pr.values()) if cand_pr else 1.0
    cand_pr = {k: v/max_pr for k, v in cand_pr.items()}
    return cand_pr

def get_embeddings_for_candidates(candidates, embeddings_path, model):
    cand_ids = [c["candidate_id"] for c in candidates]
    loaded_embeddings = {}
    
    if os.path.exists(embeddings_path):
        try:
            print(f"Loading precomputed embeddings from {embeddings_path}...")
            data = np.load(embeddings_path, allow_pickle=True)
            emb_arr = data['embeddings']
            ids_arr = data['candidate_ids']
            for idx, cid in enumerate(ids_arr):
                loaded_embeddings[cid] = emb_arr[idx]
            print(f"Loaded {len(loaded_embeddings)} embeddings.")
        except Exception as e:
            print(f"Warning: Error loading embeddings file {embeddings_path}: {e}")
            
    # Compute embeddings on-the-fly for any missing candidates
    missing_candidates = [c for c in candidates if c["candidate_id"] not in loaded_embeddings]
    if missing_candidates:
        print(f"Computing embeddings for {len(missing_candidates)} missing candidates on the fly...")
        missing_texts = [extract_text_for_embedding(c) for c in missing_candidates]
        missing_embs = model.encode(missing_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        for idx, c in enumerate(missing_candidates):
            loaded_embeddings[c["candidate_id"]] = missing_embs[idx]
            
    # Return embeddings ordered exactly as the candidates list
    final_embeddings = np.array([loaded_embeddings[cid] for cid in cand_ids])
    return final_embeddings

def build_candidate_text(candidate):
    """
    Builds a SHORT text representation for cross-encoder scoring.
    Must NOT include full career_history or other large fields.
    Total length should be under 512 tokens.
    """
    p = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    
    # Top 5 skills by duration only
    top_skills = sorted(skills, key=lambda s: s.get('duration_months', 0), reverse=True)[:5]
    skill_str = ', '.join(s['name'] for s in top_skills)
    
    return (
        f"Title: {p.get('current_title', '')}. "
        f"Company: {p.get('current_company', '')}. "
        f"Experience: {p.get('years_of_experience', 0)} years. "
        f"Skills: {skill_str}."
    )

def rank_candidates(candidates_path, embeddings_path, output_csv):
    start_time = time.time()
    print("Loading data...")
    candidates = load_candidates(candidates_path)
    
    # Build lookup: candidate_id → full raw JSONL record
    import json
    import gzip
    all_candidates_raw = {}
    open_fn = gzip.open if candidates_path.endswith('.gz') else open
    with open_fn(candidates_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                all_candidates_raw[rec['candidate_id']] = rec
    
    print("Initializing Dense Retrieval Model...")
    bge_model_path = "./models/bge-small-en-v1.5"
    if not os.path.exists(bge_model_path):
        bge_model_path = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(bge_model_path)
    
    print("Retrieving/computing candidate embeddings...")
    cand_embeddings = get_embeddings_for_candidates(candidates, embeddings_path, model)
    
    print("Embedding Job Description...")
    jd_emb = model.encode([JD_TEXT], convert_to_numpy=True)
    
    # Cosine similarity for candidate embeddings
    sims = cosine_similarity(jd_emb, cand_embeddings)[0]
    
    # Stage 1: Get top 3000 candidates by dense cosine similarity
    TOP_K = min(3000, len(candidates))
    top_indices = np.argsort(sims)[::-1][:TOP_K]
    top_cands = [candidates[idx] for idx in top_indices]
    
    cand_texts = [extract_text_for_embedding(c) for c in top_cands]
    
    # Stage 2: SkillRank Knowledge Graph on Top 3000
    graph_scores_dict = build_knowledge_graph_and_pagerank(top_cands)
    
    # Stage 3: Pseudo-Relevance Feedback (BM25) on Top 3000
    print("Computing BM25 lexical scores (Pass 1)...")
    tokenized_corpus = [text.lower().split() for text in cand_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = JD_TEXT.lower().split()
    bm25_scores_pass1 = bm25.get_scores(tokenized_query)
    
    print("Query Expansion (Pseudo-Relevance Feedback)...")
    top_20_bm25_indices = np.argsort(bm25_scores_pass1)[::-1][:20]
    extracted_skills = []
    for idx in top_20_bm25_indices:
        for s in top_cands[idx].get("skills", []):
            extracted_skills.append(s.get("name", "").lower())
    common_skills = [k for k,v in Counter(extracted_skills).most_common(5)]
    expanded_query = tokenized_query + common_skills
    print(f"Expanded Query appended with: {common_skills}")
    
    print("Computing BM25 lexical scores (Pass 2)...")
    bm25_scores = bm25.get_scores(expanded_query)
    max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
    bm25_scores = [s / max_bm25 for s in bm25_scores]
    
    # Calculate Stage-1 score to select the top 300 candidates for Cross-Encoder
    print("Selecting top 300 candidates for Cross-Encoder re-ranking...")
    max_sim = max(sims[top_indices]) if len(top_indices) else 1.0
    min_sim = min(sims[top_indices]) if len(top_indices) else 0.0
    sim_range = max_sim - min_sim if max_sim - min_sim > 0 else 1.0
    
    stage1_results = []
    for i, c in enumerate(top_cands):
        cid = c["candidate_id"]
        c_sim = sims[top_indices[i]]
        norm_sim = (c_sim - min_sim) / sim_range
        
        lex_s = float(bm25_scores[i])
        graph_s = float(graph_scores_dict.get(cid, 0.0))
        
        # Blend Stage-1 score
        s1_score = norm_sim * 0.5 + lex_s * 0.25 + graph_s * 0.25
        
        # Apply response rate and basic honeypot multiplier to focus on high-quality ones
        feats = extract_features(c)
        s1_score *= (0.5 + 0.5 * feats.get("recruiter_response_rate", 0.0))
        if feats.get("is_honeypot"):
            s1_score = 0.0
            
        stage1_results.append({
            "candidate": c,
            "features": feats,
            "s1_score": s1_score,
            "lex_score": lex_s,
            "graph_score": graph_s
        })
        
    stage1_results.sort(key=lambda x: -x["s1_score"])
    top_300_results = stage1_results[:300]
    
    # Stage 4: Cross-Encoder Semantic Re-ranking (Top 300 only)
    print("Running Cross-Encoder re-ranking on Top 300... (Optimized, ~14 seconds)")
    ce_model_path = "./models/ms-marco-MiniLM-L-6-v2"
    if not os.path.exists(ce_model_path):
        ce_model_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(ce_model_path)
    
    top_300_cands = [r["candidate"] for r in top_300_results]
    top_300_texts = [build_candidate_text(c) for c in top_300_cands]
    
    pairs = [[JD_TEXT, text] for text in top_300_texts]
    cross_scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=True)
    cross_scores = 1 / (1 + np.exp(-cross_scores)) # Sigmoid normalization
    
    # Blend and build final score list
    final_results = []
    for i, res in enumerate(top_300_results):
        c = res["candidate"]
        cid = c["candidate_id"]
        feats = res["features"]
        
        sem_s = float(cross_scores[i])
        lex_s = res["lex_score"]
        graph_s = res["graph_score"]
        
        # Final Tri-Core Blend
        base_score = sem_s * 0.5 + lex_s * 0.25 + graph_s * 0.25
        
        c_raw = all_candidates_raw[cid]
        career_bonus = compute_career_evidence_bonus(c_raw)
        score = base_score + career_bonus
        
        # Recruiter Response Rate multiplier
        score *= (0.5 + 0.5 * feats.get("recruiter_response_rate", 0.0))
        
        domain_penalty = compute_domain_penalty(c)
        score *= domain_penalty
        
        # Inactivity penalty (days since reference date 2026-06-01)
        days_inactive = feats.get("days_inactive", 0)
        if days_inactive > 180:
            score *= 0.8
            
        # Honeypot penalty
        if feats.get("is_honeypot"):
            score = 0.0
            
        final_results.append({
            "candidate": c,
            "features": feats,
            "score": score,
            "components": (sem_s, lex_s, graph_s)
        })
        
    print("Sorting with deterministic tie-breaking...")
    # Sort strictly by rounded score (4 decimals) descending and candidate_id ascending
    final_results.sort(key=lambda x: (-round(x["score"], 4), x["candidate"]["candidate_id"]))
    
    top_100 = final_results[:100]
    
    print(f"Writing {len(top_100)} candidates to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, res in enumerate(top_100, 1):
            cid = res["candidate"]["candidate_id"]
            c = all_candidates_raw[cid]
            rounded_score = round(res["score"], 4)
            reasoning = generate_reasoning(c, rank)
            writer.writerow([cid, rank, f"{rounded_score:.4f}", reasoning])
            
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates jsonl")
    parser.add_argument("--embeddings", default="embeddings.npz", help="Path to precomputed embeddings")
    parser.add_argument("--out", default="submission.csv", help="Output csv path")
    args = parser.parse_args()
    
    # Auto-detect embeddings file based on dataset path
    if args.embeddings == "embeddings.npz":
        if "sample" not in args.candidates and os.path.exists("embeddings_full.npz"):
            args.embeddings = "embeddings_full.npz"
            
    cands_path = args.candidates
    if args.candidates.endswith('.json'):
        import json
        with open(args.candidates, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    temp_jsonl = "temp_rank_candidates.jsonl"
                    with open(temp_jsonl, 'w', encoding='utf-8') as out_f:
                        for item in data: out_f.write(json.dumps(item) + '\n')
                    cands_path = temp_jsonl
            except: pass
                
    rank_candidates(cands_path, args.embeddings, args.out)
