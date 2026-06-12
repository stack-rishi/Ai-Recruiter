import json
import datetime

def extract_text_for_embedding(candidate: dict) -> str:
    """
    Builds a single string representing the candidate's career, skills, and summary.
    This string will be embedded by the dense retrieval model.
    """
    parts = []
    
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    parts.append(f"Headline: {headline}")
    if summary:
        parts.append(f"Summary: {summary}")
        
    career = candidate.get("career_history", [])
    if career:
        parts.append("Career History:")
        for job in career:
            parts.append(f"- Title: {job.get('title')} at {job.get('company')} ({job.get('industry')})")
            if job.get('description'):
                parts.append(f"  Description: {job.get('description')}")
                
    skills = candidate.get("skills", [])
    if skills:
        parts.append("Skills:")
        skill_strs = []
        for s in skills:
            duration = s.get('duration_months', 0)
            prof = s.get('proficiency', '')
            skill_strs.append(f"{s.get('name')} ({prof}, {duration} months)")
        parts.append(", ".join(skill_strs))
        
    return "\n".join(parts)

def check_h1(candidate):
    """Returns True if candidate is a confirmed honeypot via impossible dates."""
    signals = candidate.get('redrob_signals', {})
    signup = signals.get('signup_date', '')
    last_active = signals.get('last_active_date', '')
    if signup and last_active:
        return last_active < signup  # impossible: active before signing up
    return False

def check_h2(candidate):
    """Returns True if salary min > max by more than 2 LPA."""
    sal = candidate.get('redrob_signals', {}).get('expected_salary_range_inr_lpa', {})
    if not isinstance(sal, dict):
        sal = {}
    mn = sal.get('min', 0)
    mx = sal.get('max', 0)
    return mn > 0 and mx > 0 and (mn - mx) > 2.0

def extract_features(candidate: dict) -> dict:
    """
    Extracts numerical and categorical features for rule-based scoring and ranking.
    """
    feats = {}
    
    profile = candidate.get("profile", {})
    feats["years_of_experience"] = profile.get("years_of_experience", 0)
    
    signals = candidate.get("redrob_signals", {})
    
    # Behavioral signals
    feats["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0.0)
    feats["profile_completeness_score"] = signals.get("profile_completeness_score", 0.0)
    
    # Inactivity penalty: calculate days since last active relative to dataset baseline (2026-06-01)
    feats["days_inactive"] = 0
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            # Assuming format YYYY-MM-DD
            last_dt = datetime.datetime.strptime(last_active, "%Y-%m-%d")
            ref_dt = datetime.datetime(2026, 6, 1) # Baseline date for the challenge
            delta = ref_dt - last_dt
            feats["days_inactive"] = max(0, delta.days)
        except ValueError:
            pass
            
    # Trust & Verify
    feats["verified_email"] = 1.0 if signals.get("verified_email") else 0.0
    feats["verified_phone"] = 1.0 if signals.get("verified_phone") else 0.0
    feats["linkedin_connected"] = 1.0 if signals.get("linkedin_connected") else 0.0
    
    # Honeypot detection
    feats["is_honeypot"] = check_h1(candidate) or check_h2(candidate)
    
    
    return feats

# Terms that indicate genuine production retrieval/ranking work
# Found in career history DESCRIPTION fields only — not skills section
CAREER_EVIDENCE_TERMS = {
    # Strong signals — direct JD match (0.06 each, max one per term)
    'learning-to-rank':     0.06,
    'learning to rank':     0.06,
    'ranking model':        0.06,
    'reranking':            0.05,
    'retrieval system':     0.06,
    'information retrieval':0.06,
    'embedding-based':      0.06,
    'embedding based':      0.06,
    'dense retrieval':      0.06,
    'vector search':        0.05,
    'semantic search':      0.05,
    'recommendation system':0.05,
    'recommender system':   0.05,
    'search system':        0.04,
    # Evaluation signals — the JD explicitly requires this
    'offline-online':       0.06,
    'offline metric':       0.05,
    'a/b test':             0.04,
    'a/b experiment':       0.04,
    'ndcg':                 0.05,
    'relevance label':      0.05,
    'precision@':           0.04,
    'recall@':              0.04,
    # Production signals — JD requires production deployment
    'shipped':              0.03,
    'deployed to':          0.03,
    'in production':        0.03,
    'improved revenue':     0.04,
    'improved precision':   0.04,
    'reduced latency':      0.03,
}


def compute_career_evidence_bonus(candidate):
    """
    Scans ALL career history description fields for evidence of production
    retrieval/ranking/recommendation work.
    
    Returns a bonus score between 0.0 and 0.20.
    
    IMPORTANT: Only scans the 'description' field of career_history entries.
    Does NOT scan the skills section — that would defeat the purpose.
    """
    career_history = candidate.get('career_history', [])
    
    # Collect all description text, lowercased
    all_descriptions = ' '.join(
        job.get('description', '').lower() 
        for job in career_history
    )
    
    if not all_descriptions.strip():
        return 0.0
    
    total_bonus = 0.0
    for term, weight in CAREER_EVIDENCE_TERMS.items():
        if term in all_descriptions:
            total_bonus += weight
    
    # Cap at 0.20 to prevent extreme score inflation
    return min(total_bonus, 0.20)

CV_SPEECH_TERMS = {
    'computer vision', 'object detection', 'image classification', 
    'image segmentation', 'opencv', 'yolo', 'rcnn', 'convolutional',
    'speech recognition', 'asr', 'tts', 'text-to-speech', 'speech synthesis',
    'robotics', 'autonomous driving', 'lidar', 'slam'
}

NLP_RETRIEVAL_TERMS = {
    'nlp', 'natural language', 'retrieval', 'ranking', 'recommendation',
    'embedding', 'transformer', 'bert', 'information retrieval', 'search',
    'text classification', 'sentiment', 'named entity', 'question answering',
    'semantic', 'vector', 'language model', 'llm'
}


def classify_primary_domain(candidate):
    """
    Returns 'cv_speech' if primary expertise is vision/speech,
    'nlp_retrieval' if primary expertise is NLP/search/ranking,
    'mixed' otherwise.
    """
    all_desc = ' '.join(
        job.get('description', '').lower() 
        for job in candidate.get('career_history', [])
    )
    
    cv_score = sum(1 for term in CV_SPEECH_TERMS if term in all_desc)
    nlp_score = sum(1 for term in NLP_RETRIEVAL_TERMS if term in all_desc)
    
    if cv_score > nlp_score and cv_score >= 2:
        return 'cv_speech', cv_score, nlp_score
    elif nlp_score > cv_score:
        return 'nlp_retrieval', cv_score, nlp_score
    else:
        return 'mixed', cv_score, nlp_score


def compute_domain_penalty(candidate):
    """
    Returns a multiplier (0.4, 0.7, or 1.0) based on whether the candidate's
    primary expertise is in a domain the JD explicitly disqualifies.
    """
    domain, cv_score, nlp_score = classify_primary_domain(candidate)
    if domain == 'cv_speech':
        return 0.4
    elif domain == 'mixed':
        return 0.7
    return 1.0
