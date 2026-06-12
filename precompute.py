import json
import gzip
import numpy as np
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from features import extract_text_for_embedding

def load_candidates(filepath):
    candidates = []
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    return candidates

def precompute_embeddings(candidates_path, output_path, model_name="BAAI/bge-small-en-v1.5"):
    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    
    print("Extracting text for embedding...")
    texts = [extract_text_for_embedding(c) for c in candidates]
    candidate_ids = [c["candidate_id"] for c in candidates]
    
    # Use local model if available, fallback to HuggingFace download
    bge_local = "./models/bge-small-en-v1.5"
    if os.path.exists(bge_local):
        model_name = bge_local
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Pre-cache cross-encoder model too
    ce_local = "./models/ms-marco-MiniLM-L-6-v2"
    ce_name = ce_local if os.path.exists(ce_local) else 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    print(f"Loading cross-encoder model for caching...")
    cross_encoder = CrossEncoder(ce_name)
    
    print("Computing embeddings... (This may take a while)")
    # Encode in batches
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Saving embeddings to {output_path}...")
    # Save as .npz to include candidate IDs
    np.savez_compressed(output_path, embeddings=embeddings, candidate_ids=candidate_ids)
    print("Pre-computation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates jsonl or json")
    parser.add_argument("--out", default="embeddings.npz", help="Output npz file path")
    args = parser.parse_args()
    
    # If it's the sample JSON (not JSONL), we load it slightly differently
    # But for simplicity, let's assume it's formatted as jsonl or a json array
    if args.candidates.endswith('.json'):
        with open(args.candidates, 'r', encoding='utf-8') as f:
            try:
                # Try parsing as JSON array first
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                
                # Write to a temporary jsonl for the standard loader
                temp_jsonl = "temp_candidates.jsonl"
                with open(temp_jsonl, 'w', encoding='utf-8') as out_f:
                    for c in data:
                        out_f.write(json.dumps(c) + '\n')
                args.candidates = temp_jsonl
            except json.JSONDecodeError:
                # If it fails, it might already be JSONL, proceed
                pass

    precompute_embeddings(args.candidates, args.out)
