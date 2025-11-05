# Robust CSV → train.jsonl (built-ins only)
# - Puts ALL rows into train split (no test)
# - Supports weird headers like: ['', 'answer', 'question.question', 'question.paragraph']
# - Merges paragraph/context into the question if present

import csv
import json
from pathlib import Path

# ==== CHANGE THESE ====
IN_FILE = "C:/Users/chama/Downloads/Telegram Desktop/qna-dataset-farmgenie-plant-diseases_v2.csv"
OUT_DIR = "C:/Users/chama/Downloads/Telegram Desktop"
MERGE_PARAGRAPH = True
MIN_Q_LEN = 1
MIN_A_LEN = 1
# ======================

def norm(s): return (s or "").strip()
def norm_key(s): return (s or "").strip().lower()

# Heuristic: pick a question column.
# Prefer exact names, else pick the first header that contains "question"
# but is NOT the paragraph.
def pick_question_col(fieldnames):
    exacts = {"question", "questions", "q", "prompt", "question_text"}
    for fn in fieldnames:
        if fn in exacts:
            return fn
    # common nested keys like "question.question"
    for fn in fieldnames:
        if "question" in fn and "paragraph" not in fn:
            return fn
    return None

# Heuristic: pick paragraph/context column if present
def pick_paragraph_col(fieldnames):
    candidates = [
        "question.paragraph", "question_paragraph", "paragraph", "context", "passage"
    ]
    for fn in fieldnames:
        if fn in candidates:
            return fn
    # fallback: any header containing both "question" and "paragraph"
    for fn in fieldnames:
        if "question" in fn and "paragraph" in fn:
            return fn
    return None

in_path = Path(IN_FILE)
out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "train.jsonl"

# Open CSV and normalize headers
with open(in_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    try:
        raw_headers = next(reader)
    except StopIteration:
        raise SystemExit("ERROR: Empty CSV (no header row).")

# Normalize header names (lowercase, stripped); keep mapping to original positions
norm_headers = [norm_key(h) for h in raw_headers]

# Reopen with DictReader using normalized headers
with open(in_path, "r", encoding="utf-8", newline="") as f:
    dict_reader = csv.DictReader(f)
    dict_reader.fieldnames = norm_headers  # force normalized names

    fieldnames = dict_reader.fieldnames or []
    # Remove empty header names like "" that sometimes appear
    fieldnames = [h for h in fieldnames if h != ""]

    q_col = pick_question_col(fieldnames)
    a_col = "answer" if "answer" in fieldnames else None
    p_col = pick_paragraph_col(fieldnames) if MERGE_PARAGRAPH else None

    print("Detected columns:", fieldnames)
    print(f"Using: question='{q_col}', answer='{a_col}', paragraph='{p_col}'")

    if q_col is None or a_col is None:
        raise SystemExit("ERROR: Could not find required 'question' and/or 'answer' columns.")

    kept = 0
    dropped_empty = 0
    dropped_short = 0
    out_rows = []

    for raw in dict_reader:
        # keys already normalized; values may be None
        q = norm(raw.get(q_col, ""))
        a = norm(raw.get(a_col, ""))
        para = norm(raw.get(p_col, "")) if p_col else ""

        if MERGE_PARAGRAPH and para:
            q = f"Context: {para} Question: {q}"

        if not q and not a:
            dropped_empty += 1
            continue
        if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
            dropped_short += 1
            continue

        out_rows.append({"question": q, "answer": a, "split": "train"})
        kept += 1

# Write JSONL
with open(out_path, "w", encoding="utf-8") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Done. Total samples kept: {kept} | Saved to: {out_path}")
if kept == 0:
    print("DEBUG: No rows kept.")
    print(f" - dropped_empty: {dropped_empty}")
    print(f" - dropped_short (q<{MIN_Q_LEN} or a<{MIN_A_LEN}): {dropped_short}")
    print("HINT: Check the printed headers above. If your question column has a different name, extend pick_question_col().")

