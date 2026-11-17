
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "statistics" / "demographic_effects_summary.csv"
OUTPUT = ROOT / "results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

labels = {
    "female_name": "Anna",
    "male_name": "Adam",
    "name_ahmed": "Ahmed",
    "name_james": "James",
    "age_25": "Age 25",
    "age_50": "Age 50",
}

df["variation"] = df["version"].map(labels)
df["model"] = df["model"].str.upper()

sns.set_theme(style="whitegrid")

# Figure 1: Ranking changes
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="variation",
    y="ranking_change_rate",
    hue="model"
)
plt.title("Candidate Ranking Changes")
plt.ylabel("Ranking Change Rate")
plt.xlabel("Demographic Variation")
plt.tight_layout()
plt.savefig(OUTPUT / "ranking_changes.png", dpi=300)
plt.close()

# Figure 2: Similarity score differences with 95% CIs
fig, ax = plt.subplots(figsize=(10, 5))

for i, model in enumerate(["SBERT", "E5"]):
    subset = df[df["model"] == model]

    x = [
        list(labels.values()).index(v) + (i - 0.5) * 0.15
        for v in subset["variation"]
    ]

    y = subset["mean_score_difference"].to_numpy()

    errors = [
        y - subset["score_ci_lower"].to_numpy(),
        subset["score_ci_upper"].to_numpy() - y
    ]

    ax.errorbar(
        x, y, yerr=errors,
        fmt="o", capsize=4, label=model
    )

ax.axhline(0, linestyle="--", color="gray")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels.values())
ax.set_ylabel("Mean Similarity Score Difference")
ax.set_title("Demographic Effects with 95% Confidence Intervals")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT / "similarity_differences.png", dpi=300)
plt.close()

# Figure 3: Original vs augmented similarity
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="variation",
    y="mean_original_augmented_similarity",
    hue="model"
)
plt.title("Original vs Augmented Resume Similarity")
plt.ylabel("Mean Cosine Similarity")
plt.xlabel("Demographic Variation")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUTPUT / "original_augmented_similarity.png", dpi=300)
plt.close()

print("STEP 11 COMPLETED")
print("Saved 3 figures in:", OUTPUT)
