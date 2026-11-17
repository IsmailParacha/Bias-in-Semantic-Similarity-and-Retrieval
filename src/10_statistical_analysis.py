
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
SOURCE = ROOT / "results" / "similarity"
OUT = ROOT / "results" / "statistics"
OUT.mkdir(parents=True, exist_ok=True)

links = pd.read_csv(
    DATA / "selected_job_resume_links.csv",
    dtype={"job_id": str, "candidate_id": str}
)

scores = pd.read_csv(
    SOURCE / "job_resume_similarity.csv",
    dtype={"job_id": str, "candidate_id": str}
)

rankings = pd.read_csv(
    SOURCE / "ranking_changes.csv",
    dtype={"job_id": str, "candidate_id": str}
)

pairs = pd.read_csv(
    SOURCE / "original_augmented_similarity.csv",
    dtype={"candidate_id": str}
)

# Only evaluate linked job-candidate pairs
scores = scores.merge(
    links[["job_id", "candidate_id"]].drop_duplicates(),
    on=["job_id", "candidate_id"],
    how="inner",
    validate="many_to_one"
)

scores = scores[scores["version"] != "original"].copy()

# Combine all three experiment results
df = scores.merge(
    rankings[[
        "model", "job_id", "candidate_id", "version",
        "rank_change", "ranking_changed"
    ]],
    on=["model", "job_id", "candidate_id", "version"],
    validate="one_to_one"
)

df = df.merge(
    pairs,
    on=["model", "candidate_id", "version"],
    validate="many_to_one"
)

assert len(df) == 1200
assert df["score_difference"].notna().all()

# Cluster bootstrap: resample jobs, not individual rows
rng = np.random.default_rng(42)

def confidence_interval(group, column, iterations=3000):
    job_means = group.groupby("job_id")[column].mean()
    values = job_means.to_numpy()

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True
    )

    estimates = samples.mean(axis=1)

    return np.percentile(estimates, [2.5, 97.5])


results = []

for (model, version), group in df.groupby(
    ["model", "version"]
):
    score_ci = confidence_interval(
        group, "score_difference"
    )

    rank_ci = confidence_interval(
        group, "ranking_changed"
    )

    results.append({
        "model": model,
        "version": version,
        "mean_score_difference":
            group["score_difference"].mean(),
        "score_ci_lower": score_ci[0],
        "score_ci_upper": score_ci[1],
        "mean_original_augmented_similarity":
            group["original_augmented_similarity"].mean(),
        "ranking_change_rate":
            group["ranking_changed"].mean(),
        "rank_ci_lower": rank_ci[0],
        "rank_ci_upper": rank_ci[1],
        "mean_rank_change":
            group["rank_change"].mean()
    })

summary = pd.DataFrame(results)

summary.to_csv(
    OUT / "demographic_effects_summary.csv",
    index=False
)

print("\nDEMOGRAPHIC EFFECTS")
print(summary.round(4).to_string(index=False))

# Compare E5 and SBERT for the same experiments
job_rates = df.groupby(
    ["job_id", "model"]
)["ranking_changed"].mean().unstack()

job_rates["e5_minus_sbert"] = (
    job_rates["e5"] - job_rates["sbert"]
)

difference = job_rates["e5_minus_sbert"]

samples = rng.choice(
    difference.to_numpy(),
    size=(3000, len(difference)),
    replace=True
).mean(axis=1)

print("\nMODEL COMPARISON")
print("E5 - SBERT ranking-change rate:",
      round(difference.mean(), 4))
print("95% CI:",
      np.percentile(samples, [2.5, 97.5]).round(4))

print("\nSTEP 10 COMPLETED")
print("Saved:", OUT / "demographic_effects_summary.csv")
