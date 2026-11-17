
from pathlib import Path
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
EMB = ROOT / "results" / "embeddings"
SIM = ROOT / "results" / "similarity"
OUT = ROOT / "results"

checks = {}
warnings = []

def check(name, condition):
    checks[name] = bool(condition)
    print(f"{'PASS' if condition else 'FAIL'}: {name}")

# 1. Load datasets
original = pd.read_csv(
    DATA / "selected_resumes.csv", dtype={"candidate_id": str}
)
prepared = pd.read_csv(
    DATA / "model_ready_resumes.csv", dtype={"candidate_id": str}
)
jobs = pd.read_csv(DATA / "model_ready_jobs.csv", dtype={"job_id": str})
profiles = pd.read_csv(
    DATA / "augmented_resumes.csv", dtype={"candidate_id": str}
)
links = pd.read_csv(
    DATA / "selected_job_resume_links.csv",
    dtype={"job_id": str, "candidate_id": str}
)

expected_versions = {
    "original": ("Alex", 30),
    "female_name": ("Anna", 30),
    "male_name": ("Adam", 30),
    "name_james": ("James", 30),
    "name_ahmed": ("Ahmed", 30),
    "age_25": ("Alex", 25),
    "age_50": ("Alex", 50),
}

check("20 jobs", len(jobs) == 20)
check("100 original resumes", len(original) == 100)
check("700 profiles", len(profiles) == 700)
check("100 links", len(links) == 100)

# 2. Check demographic variations
valid_profiles = True

for candidate_id, group in profiles.groupby("candidate_id"):
    versions = set(group["version"])

    if len(group) != 7 or versions != set(expected_versions):
        valid_profiles = False
        continue

    baseline = group.loc[
        group["version"] == "original", "profile_text"
    ].iloc[0]

    baseline_body = baseline.split("Profile:\n", 1)[-1]

    for _, row in group.iterrows():
        name, age = expected_versions[row["version"]]

        expected_text = (
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Profile:\n{baseline_body}"
        )

        if row["profile_text"] != expected_text:
            valid_profiles = False

check("Controlled demographic variations", valid_profiles)

# 3. Check token lengths
models = {
    "sbert": (
        "sentence-transformers/all-MiniLM-L6-v2", 256, ""
    ),
    "e5": ("intfloat/e5-base-v2", 512, "passage: ")
}

for label, (model_name, limit, prefix) in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lengths = [
        len(tokenizer.encode(
            prefix + text,
            truncation=False
        ))
        for text in profiles["profile_text"].fillna("")
    ]

    check(
        f"{label.upper()} token limit",
        max(lengths) <= limit
    )

    print(f"  Maximum tokens: {max(lengths)}")

# 4. Check saved embeddings and row order
saved_jobs = pd.read_csv(EMB / "job_ids.csv", dtype=str)
saved_profiles = pd.read_csv(EMB / "resume_ids.csv", dtype=str)

check(
    "Job embedding row order",
    saved_jobs["job_id"].tolist() == jobs["job_id"].tolist()
)

check(
    "Resume embedding row order",
    saved_profiles[["candidate_id", "version"]].reset_index(
        drop=True
    ).equals(
        profiles[["candidate_id", "version"]].reset_index(
            drop=True
        )
    )
)

for label, dimensions in [("sbert", 384), ("e5", 768)]:
    job_emb = np.load(EMB / f"{label}_jobs.npy")
    resume_emb = np.load(EMB / f"{label}_resumes.npy")

    check(
        f"{label.upper()} embedding shapes",
        job_emb.shape == (20, dimensions)
        and resume_emb.shape == (700, dimensions)
        and np.isfinite(job_emb).all()
        and np.isfinite(resume_emb).all()
    )

# 5. Validate experiment results
pairs = pd.read_csv(SIM / "original_augmented_similarity.csv")
scores = pd.read_csv(SIM / "job_resume_similarity.csv")
rankings = pd.read_csv(SIM / "ranking_changes.csv")

check("1200 original-augmented scores", len(pairs) == 1200)
check("28000 job-resume scores", len(scores) == 28000)
check("1200 ranking comparisons", len(rankings) == 1200)

check(
    "Unique similarity records",
    not scores.duplicated(
        ["model", "job_id", "candidate_id", "version"]
    ).any()
)

check(
    "No missing experimental scores",
    scores[["similarity", "score_difference"]]
    .notna().all().all()
)

# 6. Check how much text was retained
merged = original.merge(
    prepared[["candidate_id", "model_text"]],
    on="candidate_id",
    validate="one_to_one"
)

original_lengths = merged["resume_text"].fillna("").str.len()
prepared_lengths = merged["model_text"].fillna("").str.len()

retained = (
    prepared_lengths / original_lengths.replace(0, np.nan)
).fillna(0)

low_retention = int((retained < 0.70).sum())

print("\nTEXT RETENTION")
print("Average character retention:", round(retained.mean(), 3))
print("Resumes below 70% retention:", low_retention)

if low_retention:
    warnings.append(
        f"{low_retention} resumes retain less than "
        "70% of original characters."
    )

warnings.append(
    "Manually inspect retained qualifications, "
    "demographic conflicts, personal identifiers, "
    "and dataset reuse permissions."
)

# 7. Save validation report
report = {
    "checks": checks,
    "text_retention": {
        "average_fraction": float(retained.mean()),
        "resumes_below_70_percent": low_retention
    },
    "warnings": warnings
}

output = OUT / "experiment_validation.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\nSTEP 12 COMPLETED")
print("Passed:", sum(checks.values()), "/", len(checks))
print("Saved:", output)
