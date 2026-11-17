
import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv(
    "data/processed/augmented_resumes.csv"
)

models = {
    "SBERT": ("sentence-transformers/all-MiniLM-L6-v2", 256),
    "E5": ("intfloat/e5-base-v2", 512)
}

for name, (model_id, limit) in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    lengths = [
        len(tokenizer.encode(
            ("passage: " if name == "E5" else "") + str(text),
            add_special_tokens=True,
            truncation=False
        ))
        for text in df["profile_text"]
    ]

    print(f"\n{name}")
    print("Maximum tokens:", max(lengths))
    print("Profiles exceeding limit:", sum(n > limit for n in lengths))
