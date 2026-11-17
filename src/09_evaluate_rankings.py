
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "similarity"

# Load existing experiment results
scores = pd.read_csv(
    RESULTS / "job_resume_similarity.csv",
    dtype={"job_id": str, "candidate_id": str}
)

links = pd.read_csv(
    DATA / "selected_job_resume_links.csv",
    dtype={"job_id": str, "candidate_id": str}
)

VERSIONS = [
    "female_name", "male_name",
    "name_james", "name_ahmed",
    "age_25", "age_50"
]

# Use the 5 linked candidates per job
links = links[["job_id", "candidate_id"]].drop_duplicates()

assert len(links) == 100
assert links.groupby("job_id").size().eq(5).all()

# Fast score lookup
lookup = scores.set_index(
    ["model", "job_id", "candidate_id", "version"]
)["similarity"]

results = []

for model in ["sbert", "e5"]:
    print(f"Evaluating {model} rankings...")

    for job_id, group in links.groupby("job_id"):
        candidates = group["candidate_id"].tolist()

        # Ranking with all original profiles
        original_scores = pd.Series({
            cid: lookup.loc[(model, job_id, cid, "original")]
            for cid in candidates
        })

        original_ranks = original_scores.rank(
            ascending=False, method="min"
        )

        for cid in candidates:
            for version in VERSIONS:

                # Change only this candidate's score
                modified_scores = original_scores.copy()
                modified_scores.loc[cid] = lookup.loc[
                    (model, job_id, cid, version)
                ]

                modified_ranks = modified_scores.rank(
                    ascending=False, method="min"
                )

                old_rank = int(original_ranks.loc[cid])
                new_rank = int(modified_ranks.loc[cid])

                results.append({
                    "model": model,
                    "job_id": job_id,
                    "candidate_id": cid,
                    "version": version,
                    "original_rank": old_rank,
                    "augmented_rank": new_rank,
                    "rank_change": new_rank - old_rank,
                    "ranking_changed": old_rank != new_rank
                })

df = pd.DataFrame(results)

output = RESULTS / "ranking_changes.csv"
df.to_csv(output, index=False)

print("\nSTEP 9 COMPLETED")
print("Total comparisons:", len(df))

print("\nRanking changes by model:")
print(df.groupby("model")["ranking_changed"].agg(
    ["sum", "count", "mean"]
))

print("\nSaved:", output)
