
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

INPUT = DATA / "model_ready_resumes.csv"
OUTPUT = DATA / "augmented_resumes.csv"

# Synthetic baseline attributes
BASE_NAME = "Alex"
BASE_AGE = 30

# One attribute changes per variation
VARIATIONS = [
    ("original", BASE_NAME, BASE_AGE, "baseline"),
    ("female_name", "Anna", 30, "name"),
    ("male_name", "Adam", 30, "name"),
    ("name_james", "James", 30, "name"),
    ("name_ahmed", "Ahmed", 30, "name"),
    ("age_25", BASE_NAME, 25, "age"),
    ("age_50", BASE_NAME, 50, "age"),
]


def create_profile(name, age, resume_text):
    """Combine synthetic attributes with unchanged resume text."""
    return (
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Profile:\n{resume_text}"
    )


def generate_variations(df):
    records = []

    for _, row in df.iterrows():
        candidate_id = str(row["candidate_id"])
        resume_text = str(row["model_text"])

        if not resume_text.strip() or resume_text == "nan":
            raise ValueError(
                f"Empty resume for candidate {candidate_id}"
            )

        for version, name, age, attribute in VARIATIONS:
            records.append({
                "candidate_id": candidate_id,
                "version": version,
                "changed_attribute": attribute,
                "name": name,
                "age": age,
                "profile_text": create_profile(
                    name, age, resume_text
                )
            })

    return pd.DataFrame(records)


def main():
    df = pd.read_csv(
        INPUT,
        dtype={"candidate_id": str}
    )

    if df["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs found")

    if df["model_text"].isna().any():
        raise ValueError("Missing model_text found")

    augmented = generate_variations(df)

    # Validate that each candidate has 7 versions
    counts = augmented.groupby("candidate_id").size()

    assert (counts == 7).all(), (
        "Some candidates do not have 7 versions"
    )

    # Save output
    augmented.to_csv(OUTPUT, index=False)

    print("\nSTEP 5 COMPLETED")
    print("Original resumes:", len(df))
    print("Total profiles:", len(augmented))
    print("Augmented profiles:", len(augmented) - len(df))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
    print("Step 5 script started")
