"""Step 2: Inspect selected data, possible personal identifiers and model input lengths.

No training, augmentation, embedding generation, or deletion of source text.
Run: pip install transformers pandas; python src/02_check_dataset_and_token_lengths.py
"""
from pathlib import Path
import json
import re
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
REPORT = ROOT / 'results'
REPORT.mkdir(parents=True, exist_ok=True)

MODELS = {
    'SBERT_all-MiniLM-L6-v2': ('sentence-transformers/all-MiniLM-L6-v2', 256),
    'E5_base_v2': ('intfloat/e5-base-v2', 512),
}
# Screening patterns are deliberately conservative and imperfect; manually check flagged records.
PATTERNS = {
    'email': r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
    'phone_like': r'(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)',
    'url': r'(?i)\b(?:https?://|www\.)\S+',
    'linkedin': r'(?i)linkedin(?:\.com|\s*profile)',
}

def main():
    jobs = pd.read_csv(DATA / 'selected_jobs.csv', dtype={'job_id': str}).fillna('')
    cvs = pd.read_csv(DATA / 'selected_resumes.csv', dtype={'candidate_id': str}).fillna('')
    links = pd.read_csv(DATA / 'selected_job_resume_links.csv', dtype={'candidate_id': str, 'job_id': str})
    required = [(jobs, ['job_id','job_description']), (cvs, ['candidate_id','resume_text']), (links, ['candidate_id','job_id'])]
    for frame, cols in required:
        absent = set(cols)-set(frame.columns)
        if absent: raise ValueError(f'Missing columns: {sorted(absent)}')

    issues = {
        'missing_job_links': int((~links.job_id.isin(jobs.job_id)).sum()),
        'missing_candidate_links': int((~links.candidate_id.isin(cvs.candidate_id)).sum()),
        'duplicate_jobs': int(jobs.job_id.duplicated().sum()),
        'duplicate_resumes': int(cvs.candidate_id.duplicated().sum()),
    }
    report = {'selected_rows': {'jobs': len(jobs), 'resumes': len(cvs), 'links': len(links)}, 'integrity_issues': issues, 'token_check': {}, 'identifier_screen': {}}

    for kind, df, col in [('jobs',jobs,'job_description'),('resumes',cvs,'resume_text')]:
        report['identifier_screen'][kind] = {name: int(df[col].astype(str).str.contains(pattern, regex=True, na=False).sum()) for name,pattern in PATTERNS.items()}
        report['identifier_screen'][kind]['empty_text'] = int((df[col].astype(str).str.strip()=='').sum())

    for key, (model_id, max_len) in MODELS.items():
        print(f'Loading tokenizer: {model_id}')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        report['token_check'][key] = {'model':model_id, 'max_tokens':max_len}
        for kind, df, col in [('jobs',jobs,'job_description'),('resumes',cvs,'resume_text')]:
            texts = df[col].astype(str).tolist()
            if key.startswith('E5_'):
                texts = [f"{'query' if kind == 'jobs' else 'passage'}: {s}" for s in texts]
            lengths = [len(tokenizer.encode(txt, add_special_tokens=True, truncation=False)) for txt in texts]
            s = pd.Series(lengths)
            report['token_check'][key][kind] = {
                'min':int(s.min()), 'median':float(s.median()), 'p95':float(s.quantile(.95)),
                'max':int(s.max()), 'over_model_limit':int((s > max_len).sum()),
                'over_200_tokens':int((s>200).sum()),
            }
    path = REPORT / 'step2_dataset_report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSTEP 2 SUMMARY (counts only; no personal text printed)')
    print(json.dumps(report, indent=2))
    print(f'\nSaved: {path}')
    print('Important: Screening flags are NOT proof of personal data or proof that text is safe.')
    print('Inspect source permissions and review actual flagged resumes locally before sharing or using them.')

if __name__ == '__main__':
    main()
