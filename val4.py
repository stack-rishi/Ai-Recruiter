import csv, json

with open('C:\\Users\\RISHI\\Desktop\\Ai-Recruiter\\[PUB] India_runs_data_and_ai_challenge\\[PUB] India_runs_data_and_ai_challenge\\India_runs_data_and_ai_challenge\\candidates.jsonl', encoding='utf-8') as f:
    all_candidates = {json.loads(l)['candidate_id']: json.loads(l) 
                      for l in f if l.strip()}

with open('submission.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

results = {}

# 1. Row count
results['row_count'] = len(rows) == 100

# 2. Ranks 1-100 each exactly once
ranks = [int(r['rank']) for r in rows]
results['ranks_valid'] = sorted(ranks) == list(range(1, 101))

# 3. No duplicate candidate IDs
ids = [r['candidate_id'] for r in rows]
results['no_duplicate_ids'] = len(set(ids)) == 100

# 4. Scores monotonically non-increasing
scores = [float(r['score']) for r in rows]
results['monotonic_scores'] = all(scores[i] >= scores[i+1] for i in range(99))

# 5. Tie-breaking correct
tie_ok = True
for i in range(99):
    if round(scores[i], 4) == round(scores[i+1], 4):
        if ids[i] >= ids[i+1]:
            tie_ok = False
            print(f"Tie-break violation: rank {i+1} {ids[i]} >= rank {i+2} {ids[i+1]}")
results['tie_breaking_valid'] = tie_ok

# 6. No H1 honeypots
h1_count = 0
for r in rows:
    c = all_candidates[r['candidate_id']]
    sig = c['redrob_signals']
    if sig['last_active_date'] < sig['signup_date']:
        h1_count += 1
        print(f"H1 still present: rank {r['rank']} {r['candidate_id']}")
results['no_h1_honeypots'] = h1_count == 0

# 7. No templated reasoning
template_count = sum(1 for r in rows if 'Score heavily driven by' in r['reasoning'])
results['no_template_phrase'] = template_count == 0

# 8. Star candidate present and highly ranked
ela = next((r for r in rows if r['candidate_id'] == 'CAND_0000031'), None)
ela_rank = int(ela['rank']) if ela else 999
results['ela_singh_top15'] = ela_rank <= 15
if ela:
    print(f"Ela Singh rank: {ela_rank} (must be <= 15)")
else:
    print("CRITICAL: Ela Singh not in top 100")

# 9. No reasoning contains concern-free language for all entries
concern_count = sum(1 for r in rows 
                   if any(w in r['reasoning'].lower() 
                         for w in ['notice', 'relocation', 'days ago', 'not marked',
                                   'may require', 'unclear', 'uncertain', 'limited',
                                   'some relevant', 'exposure']))
print(f"Entries with concern/qualifier language: {concern_count}/100 (target >= 25)")
results['reasoning_has_concerns'] = concern_count >= 25

# Print final summary
print("\n=== VALIDATION SUMMARY ===")
all_pass = True
for check, passed in results.items():
    status = "[PASS]" if passed else "[FAIL]"
    if not passed:
        all_pass = False
    print(f"  {status} | {check}")

print(f"\nOverall: {'READY TO SUBMIT' if all_pass else 'DO NOT SUBMIT — fix failing checks first'}")
