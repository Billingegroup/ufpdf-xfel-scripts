import json
from pathlib import Path

from ufpdf_xfel_scripts.lcls.parsers import csv_to_json


def test_csv_to_json(tmp_path):
    here = Path(__file__).parent
    csv_path = here / "data" / "example_runlog.csv"
    json_path = tmp_path / "example_runlog.json"
    csv_to_json(csv_path, json_path)
    with open(json_path) as json_file:
        actual = json.load(json_file)
    expected = {}
    assert actual == expected
