import requests
import os
from dotenv import load_dotenv
load_dotenv() #for loading .env

VT_API = os.getenv("VT_API_KEY")  # must be set in .env
VT_FILE_SCAN = "https://www.virustotal.com/api/v3/files"
VT_FILE_REPORT = "https://www.virustotal.com/api/v3/files/"


def scan_attachment(filepath: str):
    """
    Upload attachment to VirusTotal and return analysis summary.
    Returns:
        {
            "status": "ok" | "error",
            "malicious": int,
            "suspicious": int,
            "harmless": int,
            "undetected": int,
            "sha256": str,
            "type": str
        }
    """

    if VT_API is None or VT_API.strip() == "":
        return {"status": "error", "detail": "VirusTotal API key not set"}

    headers = {"x-apikey": VT_API}

    try:
        # upload file
        with open(filepath, "rb") as f:
            upload = requests.post(VT_FILE_SCAN, headers=headers,
                                   files={"file": f}, timeout=20)
        upload_json = upload.json()

        if "data" not in upload_json:
            return {"status": "error", "detail": upload_json}

        analysis_id = upload_json["data"]["id"]

        # fetch analysis result
        report = requests.get(
            VT_FILE_REPORT + analysis_id,
            headers=headers,
            timeout=20
        ).json()

        attr = report.get("data", {}).get("attributes", {})
        stats = attr.get("stats", {})

        return {
            "status": "ok",
            "sha256": attr.get("sha256"),
            "type": attr.get("type_description"),
            "malicious": stats.get("malicious", 0),
            "suspicious": stats.get("suspicious", 0),
            "harmless": stats.get("harmless", 0),
            "undetected": stats.get("undetected", 0)
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}
