
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "embeddings"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load datasets
jobs = pd.read_csv(DATA / "model_ready_jobs.csv")
resumes = pd.read_csv(DATA / "augmented_resumes.csv")

assert len(jobs) == 20
assert len(resumes) == 700

MODELS = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
}

for label, model_name in MODELS.items():

    print(f"\nLoading {label} model...")

    model = SentenceTransformer(model_name)
    model.eval()

    # Prepare texts
    job_texts = jobs["model_text"].fillna("").tolist()
    resume_texts = resumes["profile_text"].fillna("").tolist()

    # E5 requires query/passage prefixes for retrieval
    if label == "e5":
        job_texts = ["query: " + t for t in job_texts]
        resume_texts = ["passage: " + t for t in resume_texts]

    print("Generating job embeddings...")

    job_embeddings = model.encode(
        job_texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Generating resume embeddings...")

    resume_embeddings = model.encode(
        resume_texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Save embeddings
    np.save(OUTPUT / f"{label}_jobs.npy", job_embeddings)
    np.save(OUTPUT / f"{label}_resumes.npy", resume_embeddings)

    print("Job shape:", job_embeddings.shape)
    print("Resume shape:", resume_embeddings.shape)
    print(f"{label} embeddings saved!")

# Save row order for later comparisons
jobs[["job_id"]].to_csv(
    OUTPUT / "job_ids.csv", index=False
)

resumes[["candidate_id", "version"]].to_csv(
    OUTPUT / "resume_ids.csv", index=False
)

print("\nSTEP 7 COMPLETED")
