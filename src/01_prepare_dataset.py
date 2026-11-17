"""Step 1: Audit an existing job/resume matching dataset and prepare reproducible CSVs.

This script DOES NOT fine-tune models, alter demographic details, or use stored embeddings.
Input files must be downloaded separately from the dataset repository.
"""
from pathlib import Path
import json
import random
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
SEED = 42
TARGET_JOBS = 20
TARGET_RESUMES = 100


def require_columns(df, names, filename):
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f'{filename} is missing required columns: {missing}')


def valid_text(value):
    return isinstance(value, str) and len(value.strip()) >= 80


def main():
    paths = {name: RAW / name for name in ('cv.csv', 'job.csv', 'matches.csv')}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit('Download the source CSVs first. Missing:\n' + '\n'.join(missing))

    resumes = pd.read_csv(paths['cv.csv'], low_memory=False)
    jobs = pd.read_csv(paths['job.csv'], low_memory=False)
    matches = pd.read_csv(paths['matches.csv'], low_memory=False)

    require_columns(resumes, ['candidate_id', 'resume_text'], 'cv.csv')
    require_columns(jobs, ['job_id', 'job_description'], 'job.csv')
    require_columns(matches, ['candidate_id', 'job_id'], 'matches.csv')

    audit = {
        'source': 'https://huggingface.co/datasets/nonameee12233/job-resume-matching',
        'raw_rows': {'cv': len(resumes), 'job': len(jobs), 'matches': len(matches)},
        'columns': { 'cv': resumes.columns.tolist(), 'job': jobs.columns.tolist(), 'matches': matches.columns.tolist()},
        'blank_resume_texts': int((~resumes['resume_text'].apply(valid_text)).sum()),
        'blank_job_descriptions': int((~jobs['job_description'].apply(valid_text)).sum()),
        'duplicate_candidate_ids': int(resumes['candidate_id'].duplicated().sum()),
        'duplicate_job_ids': int(jobs['job_id'].duplicated().sum()),
        'seed': SEED,
        'sampling_target': {'jobs': TARGET_JOBS, 'resumes': TARGET_RESUMES},
        'note': 'Licensing, privacy, and truncation still require review before research use.'
    }

    resumes = resumes[resumes['resume_text'].apply(valid_text)].drop_duplicates('candidate_id')
    jobs = jobs[jobs['job_description'].apply(valid_text)].drop_duplicates('job_id')
    matches = matches[['candidate_id','job_id']].drop_duplicates()
    matches = matches.merge(resumes[['candidate_id']], on='candidate_id').merge(jobs[['job_id']], on='job_id')

    # Try to select 20 jobs with 5 distinct, source-linked candidates each.
    # These source matches are sampling links, NOT ground-truth fairness labels.
    rng = random.Random(SEED)
    available = matches.groupby('job_id')['candidate_id'].apply(list)
    job_ids = [j for j, candidates in available.items() if len(candidates) >= 5]
    rng.shuffle(job_ids)
    selected = []
    used_candidates = set()
    for job_id in job_ids:
        candidates = list(available[job_id]); rng.shuffle(candidates)
        eligible = [c for c in candidates if c not in used_candidates]
        if len(eligible) < 5:
            continue
        chosen = eligible[:5]
        selected.extend((job_id, c) for c in chosen)
        used_candidates.update(chosen)
        if len(selected) == TARGET_RESUMES:
            break
    if len(selected) != TARGET_RESUMES:
        audit['sampling_error'] = f'Only found {len(selected)} distinct matched candidates with this sampling rule.'
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'dataset_audit.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
        raise SystemExit('Could not obtain 20 jobs x 5 distinct linked candidates. See dataset_audit.json; revise the sampling plan.')

    sampled_job_ids = [j for j, _ in selected[::5]]
    sampled_candidate_ids = [c for _, c in selected]
    job_out = jobs.set_index('job_id').loc[sampled_job_ids].reset_index()[['job_id','job_description']]
    resume_out = resumes.set_index('candidate_id').loc[sampled_candidate_ids].reset_index()[['candidate_id','resume_text']]
    association_out = pd.DataFrame(selected, columns=['job_id','candidate_id'])
    # IDs allow independent identification and analysis of candidate groups by job.
    OUT.mkdir(parents=True, exist_ok=True)
    job_out.to_csv(OUT / 'selected_jobs.csv', index=False)
    resume_out.to_csv(OUT / 'selected_resumes.csv', index=False)
    association_out.to_csv(OUT / 'selected_job_resume_links.csv', index=False)
    audit['selected_rows'] = {'jobs': len(job_out), 'resumes': len(resume_out), 'links': len(association_out)}
    audit['sample_candidate_ids'] = sampled_candidate_ids[:5]
    (OUT / 'dataset_audit.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    print('Step 1 done. Created selected_jobs.csv, selected_resumes.csv, selected_job_resume_links.csv and dataset_audit.json')
    print('Next: inspect sensitive identifiers / license and check SBERT/E5 token lengths before generating controlled profiles.')

if __name__ == '__main__':
    main()
