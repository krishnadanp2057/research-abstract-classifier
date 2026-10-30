"""
Collect a balanced, labeled abstract dataset from OpenAlex (free, no API key).
Run in Colab / locally:   pip install requests pandas   &&   python collect_data.py

Output: abstracts_dataset.csv with columns: id, title, abstract, label, year, field
OpenAlex indexes the same peer-reviewed literature as Scopus / Web of Science.
"""
import time
import requests
import pandas as pd

EMAIL = "your_email@example.com"   # <-- put your email (OpenAlex "polite pool", faster)
PER_CLASS = 500                    # 3 x 500 = 1500 abstracts (assignment needs >= 1000)
YEARS = "2018-2025"

# OpenAlex top-level "field" IDs
CLASSES = {
    "Science":        "31|16|13|19|11|26|12",   # Physics, Chemistry, Biochem, Earth, Agri/Bio, Maths, Materials etc.
    "Business":       "14",                     # Business, Management and Accounting
    "Social Science": "33",                     # Social Sciences (sociology, education, politics, ...)
}
BASE = "https://api.openalex.org/works"


def rebuild_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch_class(label, field_ids, n):
    rows, page = [], 1
    while len(rows) < n and page <= 15:
        params = {
            "filter": f"primary_topic.field.id:{field_ids},has_abstract:true,type:article,"
                      f"language:en,publication_year:{YEARS}",
            "sample": n * 2, "seed": 42,       # random sample -> avoids bias to newest/most-cited
            "per-page": 100, "page": page, "mailto": EMAIL,
        }
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        for w in results:
            abs_text = rebuild_abstract(w.get("abstract_inverted_index"))
            if len(abs_text.split()) < 50:      # drop very short abstracts
                continue
            rows.append({
                "id": w["id"], "title": w.get("title"), "abstract": abs_text,
                "label": label, "year": w.get("publication_year"),
                "field": (w.get("primary_topic") or {}).get("field", {}).get("display_name"),
            })
        page += 1
        time.sleep(0.2)
    return rows[:n]


if __name__ == "__main__":
    all_rows = []
    for label, ids in CLASSES.items():
        got = fetch_class(label, ids, PER_CLASS)
        print(label, len(got))
        all_rows += got
    df = pd.DataFrame(all_rows).drop_duplicates(subset="id")
    df.to_csv("abstracts_dataset.csv", index=False)
    print(df.label.value_counts())
