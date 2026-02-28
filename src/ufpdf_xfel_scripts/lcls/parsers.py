import csv
import json
import re
import sys


def to_snake_case(name):
    name = name.strip().lower()
    name = re.sub(r"[^\w\s]", "", name)  # remove punctuation
    name = re.sub(r"\s+", "_", name)  # spaces to underscores
    return name


def csv_to_json(csv_path, json_path):
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_num = row.get("Run #", "").strip()
            try:
                key = int(run_num)
            except (ValueError, TypeError):
                continue  # Skip rows without a valid integer Run #
            result[key] = {to_snake_case(col): val for col, val in row.items()}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {len(result)} runs to {json_path}")
    return


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json.py <input.csv> <output.json>")
        sys.exit(1)
    csv_to_json(sys.argv[1], sys.argv[2])
