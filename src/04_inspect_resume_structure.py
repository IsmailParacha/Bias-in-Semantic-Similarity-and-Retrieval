
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

df = pd.read_csv(DATA / "selected_resumes.csv")

print("Total resumes:", len(df))

for index, row in df.head(5).iterrows():
    text = str(row["resume_text"])

    print(f"\n--- Resume {index + 1} ---")
    print("Characters:", len(text))
    print("Words:", len(text.split()))
    print("First 500 characters:")
    print(text[:500])
