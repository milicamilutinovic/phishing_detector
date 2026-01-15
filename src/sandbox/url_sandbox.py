import requests
import os

from dotenv import load_dotenv
load_dotenv() # for loading .env
API_KEY = os.getenv("URLSCAN_API_KEY")
URLSCAN_SUBMIT = "https://urlscan.io/api/v1/scan/"
URLSCAN_RESULT = "https://urlscan.io/api/v1/result/"


def scan_url(url: str):
    """
    Submit URL to Urlscan.io and return sandbox summary.
    Returns:
        {
            "status": "ok" | "error",
            "uuid": str,
            "malicious": bool,
            "tags": list,
            "domain": str,
            "ip": str,
            "country": str
        }
    """

    if API_KEY is None or API_KEY.strip() == "":
        return {"status": "error", "detail": "Urlscan API key not set"}

    headers = {"API-Key": API_KEY, "Content-Type": "application/json"}

    data = {
        "url": url,
        "visibility": "public"
    }

    try:
        submit = requests.post(URLSCAN_SUBMIT, headers=headers,
                               json=data, timeout=20)
        submit_json = submit.json()

        if "uuid" not in submit_json:
            return {"status": "error", "detail": submit_json}

        uuid = submit_json["uuid"]

        # fetch results
        result = requests.get(URLSCAN_RESULT + uuid, timeout=20).json()

        verdict = result.get("verdicts", {}).get("overall", {})
        page = result.get("page", {})

        return {
            "status": "ok",
            "uuid": uuid,
            "malicious": verdict.get("malicious"),
            "tags": verdict.get("tags", []),
            "domain": page.get("domain"),
            "ip": page.get("ip"),
            "country": page.get("country")
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}
