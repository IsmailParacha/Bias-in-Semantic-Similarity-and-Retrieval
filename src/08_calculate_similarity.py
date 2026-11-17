
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "results" / "embeddings"
OUT = ROOT / "results" / "similarity"
OUT.mkdir(parents=True, exist_ok=True)

jobs = pd.read_csv(EMB / "job_ids.csv", dtype=str)
resumes = pd.read_csv(EMB / "resume_ids.csv", dtype=str)

pair_results = []
job_results = []

for model in ["sbert", "e5"]:
    print(f"\nCalculating {model} similarity...")

    job_emb = np.load(EMB / f"{model}_jobs.npy")
    resume_emb = np.load(EMB / f"{model}_resumes.npy")

    assert len(job_emb) == len(jobs)
    assert len(resume_emb) == len(resumes)

    # Normalized embeddings: dot product = cosine similarity
    job_scores = job_emb @ resume_emb.T

    # Experiment A: Original vs augmented
    for candidate_id, group in resumes.groupby("candidate_id"):
        original = group[group["version"] == "original"]

        if len(original) != 1:
            raise ValueError(f"Invalid original: {candidate_id}")

        original_idx = original.index[0]

        for idx in group.index:
            if idx == original_idx:
                continue

            score = float(
                resume_emb[original_idx] @ resume_emb[idx]
            )

            pair_results.append({
                "model": model,
                "candidate_id": candidate_id,
                "version": resumes.loc[idx, "version"],
                "original_augmented_similarity": score
            })

    # Experiment B: Job vs every profile
    for j, job_id in enumerate(jobs["job_id"]):
        for r, row in resumes.iterrows():
            job_results.append({
                "model": model,
                "job_id": job_id,
                "candidate_id": row["candidate_id"],
                "version": row["version"],
                "similarity": float(job_scores[j, r])
            })

pairs = pd.DataFrame(pair_results)
scores = pd.DataFrame(job_results)

# Calculate the score difference from each baseline
baseline = scores[scores["version"] == "original"][
    ["model", "job_id", "candidate_id", "similarity"]
].rename(columns={"similarity": "original_similarity"})

scores = scores.merge(
    baseline,
    on=["model", "job_id", "candidate_id"],
    how="left",
    validate="many_to_one"
)

scores["score_difference"] = (
    scores["similarity"] - scores["original_similarity"]
)

pairs.to_csv(
    OUT / "original_augmented_similarity.csv", index=False
)

scores.to_csv(
    OUT / "job_resume_similarity.csv", index=False
)

print("\nSTEP 8 COMPLETED")
print("Original-augmented comparisons:", len(pairs))
print("Job-resume comparisons:", len(scores))
print("Results saved in:", OUT)
