# src/integration/imap_poller.py

import imaplib
import email
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASS = os.getenv("IMAP_PASS")
API_URL   = os.getenv("API_URL", "http://127.0.0.1:8000/score_eml")



POLL_INTERVAL = 60

def classify_message(raw_bytes):
    try:
        r = requests.post(API_URL, files={"file": ("msg.eml", raw_bytes)})
        if r.status_code == 200:
            result = r.json()
            label = result.get("label")
            score = result.get("score")
            subj  = result.get("subject", "")[:80]
            print(f"[RESULT] {'PHISH' if label == 1 else 'LEGIT'} | Score={score:.3f} | Subject={subj}")
        else:
            print(f"[ERROR] API returned {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[ERROR] Could not reach API: {e}")

def main():
    print(f"[INFO] Connecting to IMAP server {IMAP_HOST}...")
    M = imaplib.IMAP4_SSL(IMAP_HOST)

    # login with Gmail IMAP credentialsa
    M.login(IMAP_USER, IMAP_PASS)
    print("[INFO] Login successful.")

    while True:
        try:
            M.select("INBOX")
            typ, data = M.search(None, 'UNSEEN')

            if data and data[0]:
                for num in data[0].split():
                    typ, msg_data = M.fetch(num, '(RFC822)')
                    raw = msg_data[0][1]
                    classify_message(raw)
                    M.store(num, '+FLAGS', '\\Seen')
            else:
                print("[INFO] No new messages.")

        except Exception as e:
            print(f"[WARN] Loop error: {e}")

            # reconnect
            try:
                time.sleep(5)
                M = imaplib.IMAP4_SSL(IMAP_HOST)
                M.login(IMAP_USER, IMAP_PASS)
            except Exception as e2:
                print(f"[FATAL] Cannot reconnect: {e2}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
