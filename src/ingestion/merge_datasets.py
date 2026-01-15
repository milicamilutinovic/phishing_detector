# src/ingestion/merge_datasets.py
import pandas as pd
import numpy as np
import glob, os, re, hashlib
from pathlib import Path
import unicodedata

RAW_DIR = Path("data/raw")
OUT_CSV = Path("data/processed/all_emails.csv")

TEXT_CANDIDATES = [
    "body", "text", "Email Text", "EmailText", "Message", "message", "content",
    "Email_body", "Email", "email_text"
]
SUBJECT_CANDIDATES = [
    "subject", "Subject", "SUBJECT", "Email Subject", "Title", "topic"
]
LABEL_CANDIDATES = [
    "label", "Label", "is_phishing", "target", "class", "Class", "Category", "spam",
    "labels", "phishing", "Phishing"
]

PHISH_VALUES = {
    "phishing", "phish", "1", 1, "spam", "bad", "fraud", "fraudulent", "malicious",
    "phishing email", "yes", "true", True
}
HAM_VALUES = {
    "legit", "ham", "0", 0, "good", "benign", "legitimate", "not phishing",
    "non-phishing", "non phishing", "no", "false", False
}

def pick_col(cols, candidates):
    # tacno ime
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive
    cl = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in cl:
            return cl[c.lower()]
    return None

def normalize_label(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in PHISH_VALUES:
        return 1
    if s in HAM_VALUES:
        return 0
    if s.isdigit():
        return int(s)
    return np.nan

def normalize_text(x):
    if pd.isna(x): 
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def to_series(x, n, fill=""):
    if isinstance(x, pd.Series):
        return x
    return pd.Series([x if x is not None else fill] * n)

def text_hash(subject, body):
    base = (subject or "") + "||" + (body or "")
    return hashlib.sha256(base.lower().encode("utf-8", errors="ignore")).hexdigest()

def load_one(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception as e:
        raise RuntimeError(f"CSV parse failed: {e}")

    orig_cols = list(df.columns)

    subj_col = pick_col(orig_cols, SUBJECT_CANDIDATES)
    text_col = pick_col(orig_cols, TEXT_CANDIDATES)
    lab_col  = pick_col(orig_cols, LABEL_CANDIDATES)

    if subj_col is None and text_col is None:
        textish = max(orig_cols, key=lambda c: df[c].notna().sum())
        subject = pd.Series([""] * len(df))
        body = df[textish]
    else:
        subject = df[subj_col] if subj_col in df else pd.Series([""] * len(df))
        body = df[text_col] if text_col in df else pd.Series([""] * len(df))

    subject = to_series(subject, len(df), fill="")
    body = to_series(body, len(df), fill="")

    label = df[lab_col] if lab_col in df else pd.Series([np.nan] * len(df))
    label = to_series(label, len(df), fill=np.nan)

    out = pd.DataFrame({
        "subject": subject.map(normalize_text),
        "body":    body.map(normalize_text),
        "label":   label.map(normalize_label)
    })
    out["source_file"] = os.path.basename(path)
    return out

def main():
    paths = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files found in {RAW_DIR}. Put your datasets there.")

    parts = []
    for p in paths:
        try:
            df = load_one(p)
            parts.append(df)
            print(f"Loaded {p}: {len(df)} rows")
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}")

    if not parts:
        raise SystemExit("No datasets could be loaded.")

    all_df = pd.concat(parts, ignore_index=True)

    # odbaci prazne tekstove
    all_df = all_df[(all_df["subject"].fillna("") != "") | (all_df["body"].fillna("") != "")]
    # zadrzi samo 0/1 labelu
    all_df = all_df[all_df["label"].isin([0, 1])]

    # deduplikacija
    all_df["hash"] = [
        text_hash(s, b) 
        for s, b in zip(all_df["subject"], all_df["body"])
    ]
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["hash"]).drop(columns=["hash"])
    after = len(all_df)
    print(f"Deduplicated: {before} -> {after}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {OUT_CSV} with {len(all_df)} rows")

if __name__ == "__main__":
    main()
