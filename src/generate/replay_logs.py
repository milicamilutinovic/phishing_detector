import csv
import time

SOURCE = "predictions_log.csv"
OUTPUT = "predictions_log.csv"

def replay(delay=2):
    with open(SOURCE, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        rows = reader[1:]

    print("[INFO] Starting replay...")
    for r in rows:
        r[0] = int(time.time())  # overwrite ts
        with open(OUTPUT, "a", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(r)
        print(f"[REPLAY] {r}")
        time.sleep(delay)

if __name__ == "__main__":
    replay(1)
