import csv
import io
import json
import re
from pathlib import Path

import requests


def to_snake_case(name):
    name = name.strip().lower()
    name = re.sub(r"[^\w\s]", "", name)  # remove punctuation
    name = re.sub(r"\s+", "_", name)  # spaces to underscores
    return name


def csv_to_json(runlog_url, json_path):
    payload = requests.get(runlog_url)
    payload.raise_for_status()
    reader = csv.DictReader(io.StringIO(payload.text))
    result = {}
    for row in reader:
        run_num = row.get("Run #", "").strip()
        try:
            key = int(run_num)
        except (ValueError, TypeError):
            continue
        result[key] = {to_snake_case(col): val for col, val in row.items()}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {len(result)} runs to {json_path}")
    return


def main():
    runlog_url = "https://docs.google.com/spreadsheets/d/155-Pae4rAD17RCBqGwFKtvVysDgTgsfsxgUWiHb7vSM/export?format=csv&gid=0"  # noqa E501
    json_path = Path().cwd() / "test.json"
    csv_to_json(runlog_url, json_path)


if __name__ == "__main__":
    main()
