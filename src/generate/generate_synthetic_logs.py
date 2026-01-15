import csv
import random
import time
from pathlib import Path

OUTPUT = Path("predictions_log.csv")

PHISH_SUBJECTS = [
    "Your Apple ID Has Been Locked1",
    "Payment Failed: Update Billing Information1",
    "Final Notice: Confirm Your Mailbox1",
    "URGENT: Your Account Will Be Suspended1",
    "Security Alert: Unusual Login Attempt1",
    "Delivery Failure Notification1",
    "Password Expiring Soon1",
    "PayPal: Confirm Transaction1",
]

LEGIT_SUBJECTS = [
    "Meeting scheduled for tomorrow",
    "Project update attached",
    "Thank you for your submission",
    "Invoice successfully paid",
    "University announcement",
    "New study material available",
    "Order confirmation",
    "Newsletter: Weekly Highlights",
]

FROM_SENDERS = [
    "support@apple.com",
    "billing@paypal.com",
    "admin@university.edu",
    "security@google.com",
    "hr@company.com",
    "newsletter@service.com",
]

def generate_log_entry():
    is_phish = random.random() < 0.5

    ts = int(time.time())               # real ts
    subject = random.choice(PHISH_SUBJECTS if is_phish else LEGIT_SUBJECTS)
    sender = random.choice(FROM_SENDERS)

    # scores: phishing-high, legit-low
    score = random.uniform(0.65, 0.99) if is_phish else random.uniform(0.0, 0.35)
    label = 1 if score >= 0.5 else 0
    quarantine_path = "" if label == 0 else f"quarantine/{ts}_{abs(hash(subject))}.eml"

    return [ts, subject, sender, round(score, 6), label, quarantine_path]


def main(n=1000):
    write_header = not OUTPUT.exists()

    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["ts", "subject", "from", "score", "label", "quarantine_path"])

        for _ in range(n):
            row = generate_log_entry()
            writer.writerow(row)

    print(f"[OK] Generated {n} synthetic log entries → {OUTPUT}")


if __name__ == "__main__":
    main(1200)
