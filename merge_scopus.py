"""Merge Scopus CSV exports into a balanced abstracts_dataset.csv.
Put your exports in the same folder, named by class (several files per class are fine):
    scopus_science*.csv   scopus_business*.csv   scopus_social*.csv
Then run:  python merge_scopus.py
"""
import glob
import pandas as pd

PATTERNS = {"Science": "scopus_science*.csv",
            "Business": "scopus_business*.csv",
            "Social Science": "scopus_social*.csv"}
PER_CLASS = 500          # set to None to use the size of the smallest class


def mostly_english(t):
    """Drop abstracts written in Cyrillic/other scripts (some journals add them)."""
    return sum(ord(c) > 127 for c in t) / max(len(t), 1) < 0.02


parts = []
for label, pattern in PATTERNS.items():
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d.columns = [c.strip() for c in d.columns]
    keep = [c for c in ["Title", "Abstract", "Year"] if c in d.columns]
    d = d[keep].copy()
    d["Abstract"] = (d["Abstract"].fillna("").astype(str)
                     .str.replace(r"\[No abstract available\]", "", regex=True)
                     .str.replace(r"(©|Copyright|\(c\)).{0,150}$", "", regex=True)
                     .str.strip())
    d = d[d["Abstract"].str.split().str.len() >= 50]
    d = d[d["Abstract"].apply(mostly_english)]
    d = d.drop_duplicates(subset="Abstract")
    d["label"] = label
    print(f"{label}: {len(files)} file(s), {len(d)} usable abstracts")
    parts.append(d)

n = PER_CLASS or min(len(p) for p in parts)
parts = [p.sample(n=min(n, len(p)), random_state=42) for p in parts]
df = pd.concat(parts).rename(columns={"Title": "title", "Abstract": "abstract", "Year": "year"})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("abstracts_dataset.csv", index=False)
print(df["label"].value_counts())
