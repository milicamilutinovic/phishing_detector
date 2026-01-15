# src/serve/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from fastapi import UploadFile, File
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import re
import os
import time
import csv
import math


ART = Path("models/artifacts") 
QUAR_DIR = Path("quarantine")
DEBUG_DIR = Path("debug_emls")
LOG_CSV = Path("predictions_log.csv")
QUAR_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

import re
import unicodedata

def clean_text(x: str):
    if not x:
        return ""
    # normalize unicode 
    x = unicodedata.normalize("NFKD", x)
    # remove dangerous characters
    x = x.replace('"', '').replace('„', '').replace('“', '')
    # replace commas so CSV doesn't break
    x = x.replace(',', ';')
    # remove newlines
    x = x.replace('\n', ' ').replace('\r', ' ')
    # collapse whitespace
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

try:
    clf = joblib.load(ART / "sgd_lr.joblib")
    ftr = joblib.load(ART / "text_url_featurizer_sgd.joblib")
except Exception as e:
    # try alternative common names (in case you used other names when saving)
    try:
        clf = joblib.load(ART / "baseline_model.joblib")
        ftr = joblib.load(ART / "text_url_featurizer.joblib")
    except Exception as e2:
        raise RuntimeError(f"Could not load model/featurizer from {ART}: {e}; {e2}")

def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# regex to find http(s) URLs as a fallback
URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)


class MailIn(BaseModel):
    subject: str = ""
    body: str

app = FastAPI(title="PhishDetector")

@app.get("/")
def root():
    return {"status": "ok", "service": "PhishDetector"}

@app.post("/score")
def score(m: MailIn):
    try:
        df = pd.DataFrame([{"subject": m.subject or "", "body": m.body or ""}])
        X = ftr.transform(df)
        # predict probability 
        if hasattr(clf, "predict_proba"):
            score = float(clf.predict_proba(X)[0, 1])
        else:
            df_val = clf.decision_function(X)
            score = float(_sigmoid(float(df_val[0])))
        label = int(score >= 0.5)
        
        # log
        _log_prediction(subject=m.subject, from_hdr="", score=score, label=label, raw_bytes=None)
        return {"score": score, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring input: {e}")



def _log_prediction(subject: str, from_hdr: str, score: float, label: int, raw_bytes: bytes = None):
    write_header = not LOG_CSV.exists()

    quarant_path = ""
    if raw_bytes is not None and label == 1:
        fn = QUAR_DIR / f"{int(time.time())}_{abs(hash(subject))}.eml"
        fn.write_bytes(raw_bytes)
        quarant_path = str(fn)

    ts = int(time.time())

    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["ts", "subject", "from", "score", "label", "quarantine_path"])

        # samo jednom upisujemo red
        writer.writerow([
            ts,
            clean_text(subject)[:200],
            clean_text(from_hdr)[:200],
            f"{score:.6f}",
            label,
            clean_text(quarant_path)
        ])


def _eml_to_subject_body(raw_bytes: bytes):
    """
    Robust extraction:
    - collects text/plain parts
    - collects text/html parts, extracts visible text AND hrefs
    - appends hrefs (and regex-found URLs) to final body so featurizer sees URLs
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = msg.get('subject', '') or ''
    body_text = []
    body_html = []
    for part in msg.walk():
        ctype = part.get_content_type()
        try:
            content = part.get_content()
        except Exception:
            content = ""
        if ctype == 'text/plain':
            if content:
                body_text.append(content)
        elif ctype == 'text/html':
            if content:
                body_html.append(content)

    plain = "\n".join(body_text).strip()
    html = "\n".join(body_html).strip()

    urls_from_html = []
    visible_text = ""
    if html:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href:
                urls_from_html.append(href)
        visible_text = soup.get_text(" ", strip=True)

    if plain:
        body = plain
    elif visible_text:
        body = visible_text
    else:
        body = ""

    if urls_from_html:
        body = body + "\n\n" + " ".join(urls_from_html)

    if not URL_RE.search(body):
        urls_via_regex = URL_RE.findall(html) if html else []
        if not urls_via_regex:
            try:
                raw_text = raw_bytes.decode("utf-8", errors="ignore")
                urls_via_regex = URL_RE.findall(raw_text)
            except Exception:
                urls_via_regex = []
        if urls_via_regex:
            body = body + "\n\n" + " ".join(urls_via_regex)

    return subject, body

@app.post("/score_eml")
async def score_eml(file: UploadFile = File(...), threshold: float = 0.5):
    raw = await file.read()
    subject, body = _eml_to_subject_body(raw)

   # ml prediction
    try:
        df = pd.DataFrame([{"subject": subject or "", "body": body or ""}])
        X = ftr.transform(df)

        if hasattr(clf, "predict_proba"):
            score = float(clf.predict_proba(X)[0, 1])
        else:
            val = clf.decision_function(X)
            score = float(_sigmoid(float(val[0])))

        label = int(score >= threshold)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ML prediction: {e}")

   # extract urls
    urls = URL_RE.findall(body)
    urls = list(set(urls))  # remove duplicates

    # url sandbox
    from src.sandbox.url_sandbox import scan_url
    url_sandbox_results = []

    for u in urls:
        url_sandbox_results.append({
            "url": u,
            "urlscan": scan_url(u)
        })

    # file sandbox
    from src.sandbox.file_sandbox import scan_attachment

    attachment_results = []
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    path = f"temp_{filename}"
                    with open(path, "wb") as f:
                        f.write(part.get_payload(decode=True))

                    vt_result = scan_attachment(path)
                    attachment_results.append({
                        "filename": filename,
                        "virustotal": vt_result
                    })

                    os.remove(path)
    except Exception as e:
        print(f"[WARN] Attachment scan failed: {e}")

    # log
    _log_prediction(
        subject=subject,
        from_hdr=(msg_get_from(raw) or ""),
        score=score,
        label=label,
        raw_bytes=raw if label == 1 else None
    )

   # final output
    return {
        "score": score,
        "label": label,
        "subject": subject[:150],
        "urls_found": urls,
        "url_sandbox": url_sandbox_results,
        "attachments": attachment_results
    }

def msg_get_from(raw_bytes: bytes):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        return msg.get("from", "") or ""
    except Exception:
        return ""

