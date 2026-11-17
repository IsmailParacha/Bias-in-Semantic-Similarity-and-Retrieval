
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_TOKENS = 200


def prepare_text(text):
    text = " ".join(str(text).split())

    tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_TOKENS
    )

    return tokenizer.decode(
        tokens["input_ids"],
        skip_special_tokens=True
    )


def process_file(input_file, text_column, output_file):
    df = pd.read_csv(DATA / input_file)

    df["model_text"] = df[text_column].fillna("").apply(
        prepare_text
    )

    df.to_csv(DATA / output_file, index=False)

    empty = df["model_text"].str.strip().eq("").sum()

    print(f"{output_file}: {len(df)} rows, {empty} empty texts")


process_file(
    "selected_jobs.csv",
    "job_description",
    "model_ready_jobs.csv"
)

process_file(
    "selected_resumes.csv",
    "resume_text",
    "model_ready_resumes.csv"
)

print("\nStep 3 completed!")
