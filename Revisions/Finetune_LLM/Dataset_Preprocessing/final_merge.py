# Merge two JSONL datasets (PlantVillageVQA + YuvrajSingh)
# Output: combined_train.jsonl with all unique {question, answer, split}

import json
from pathlib import Path
import re

# ==== CHANGE THESE ====
PV_TRAIN = "C:/Users/chama/Downloads/Telegram Desktop/PlantVillageVQA/train.jsonl"   
YS_TRAIN = "C:/Users/chama/Downloads/Telegram Desktop/train.jsonl"  
OUT_FILE = "C:/Users/chama/Downloads/Telegram Desktop/PlantVillageVQA/final dataset/final_dataset.jsonl"          
# ======================

def norm_key(q, a):
    """Normalize question+answer to detect duplicates."""
    def n(s):
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s
    return n(q) + "||" + n(a)

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

# Load both datasets
pv_rows = read_jsonl(PV_TRAIN)
ys_rows = read_jsonl(YS_TRAIN)

print(f"Loaded: PlantVillageVQA = {len(pv_rows)}, YuvrajSingh = {len(ys_rows)}")

# Merge and deduplicate
seen = set()
merged = []

for r in pv_rows + ys_rows:
    q = r.get("question", "").strip()
    a = r.get("answer", "").strip()
    k = norm_key(q, a)
    if not q or not a or k in seen:
        continue
    seen.add(k)
    merged.append({"question": q, "answer": a, "split": "train"})

# Write combined file
out_path = Path(OUT_FILE)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Done. Combined samples: {len(merged)} | Saved to: {out_path}")
