import csv, json

with open('[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl', encoding='utf-8') as f:
    all_candidates = {json.loads(l)['candidate_id']: json.loads(l) 
                      for l in f if l.strip()}

with open('submission.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

mismatches = []
for r in rows:
    cand = all_candidates[r['candidate_id']]
    actual_company = cand['profile'].get('current_company', '')
    reasoning = r['reasoning']
    # Many actual companies are short or abbreviations, check basic inclusion
    if actual_company and actual_company not in reasoning:
        mismatches.append((r['rank'], r['candidate_id'], actual_company, reasoning[:100]))

print(f"Company name mismatches: {len(mismatches)}/100")
for rank, cid, company, reason in mismatches[:10]:
    print(f"  Rank {rank}: {cid} — actual company '{company}' not found in reasoning")
    print(f"  Reasoning: {reason}")

# Specific check for Ela Singh
ela_row = next((r for r in rows if r['candidate_id'] == 'CAND_0000031'), None)
if ela_row:
    ela_actual = all_candidates['CAND_0000031']['profile']
    print(f"\nEla Singh (rank {ela_row['rank']}):")
    print(f"  Actual title:   {ela_actual.get('current_title')}")
    print(f"  Actual company: {ela_actual.get('current_company')}")
    print(f"  Actual YOE:     {ela_actual.get('years_of_experience')}")
    print(f"  Reasoning:      {ela_row['reasoning']}")
    company_match = str(ela_actual.get('current_company', '')) in ela_row['reasoning']
    title_match = str(ela_actual.get('current_title', '')) in ela_row['reasoning']
    print(f"  Company in reasoning: {company_match}")
    print(f"  Title in reasoning:   {title_match}")
else:
    print("\nEla Singh (CAND_0000031) not found in top 100")
