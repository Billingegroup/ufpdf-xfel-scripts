import json

from ufpdf_xfel_scripts.lcls.parsers import csv_to_json

MOCK_CSV = """Cell,Well,Sample ,Scan Type,Run #,THz ,Transmission,Number of Frames per step,"Delay : Start, Stop (ps), npts",Notes ,Detector Position (cm),"x,y position of beam on sample",Run quality check
N/A,N/A,N/A,Beamline Alignment ,1,,,,,,,,
1,4,LaB6,,2,,,,,"2 Mins of total Data, but no images were saved.",10,,
1,8,Bi,Delay,3,,,,,First Attempt at Delay scan with Bi,10,,
1,4,LaB6,Static ,4,,,,,"Motor Position 340mm, Detector distance 100mm",10,,
1,5,HVG + Kapton,Attenuation Series,5,,,,,"Motor position 279 mm, detector distance43mm",4,,
2,7,Empty,Static ,6,,,,,"Background from Quartz, just looking at window. ",,,
"""  # noqa E501


def test_csv_to_json(mocker, tmp_path):
    mock_response = mocker.MagicMock()
    mock_response.text = MOCK_CSV
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)
    json_path = tmp_path / "example_runlog.json"
    csv_to_json("https://fake-url.com", json_path)
    with open(json_path) as json_file:
        actual = json.load(json_file)
    expected = expected_csv_to_json
    assert actual == expected


expected_csv_to_json = {
    "1": {
        "cell": "N/A",
        "delay_start_stop_ps_npts": "",
        "detector_position_cm": "",
        "notes": "",
        "number_of_frames_per_step": "",
        "run_": "1",
        "run_quality_check": "",
        "sample": "N/A",
        "scan_type": "Beamline Alignment ",
        "thz": "",
        "transmission": "",
        "well": "N/A",
        "xy_position_of_beam_on_sample": "",
    },
    "2": {
        "cell": "1",
        "delay_start_stop_ps_npts": "",
        "detector_position_cm": "10",
        "notes": "2 Mins of total Data, but no images were saved.",
        "number_of_frames_per_step": "",
        "run_": "2",
        "run_quality_check": "",
        "sample": "LaB6",
        "scan_type": "",
        "thz": "",
        "transmission": "",
        "well": "4",
        "xy_position_of_beam_on_sample": "",
    },
    "3": {
        "cell": "1",
        "delay_start_stop_ps_npts": "",
        "detector_position_cm": "10",
        "notes": "First Attempt at Delay scan with Bi",
        "number_of_frames_per_step": "",
        "run_": "3",
        "run_quality_check": "",
        "sample": "Bi",
        "scan_type": "Delay",
        "thz": "",
        "transmission": "",
        "well": "8",
        "xy_position_of_beam_on_sample": "",
    },
    "4": {
        "cell": "1",
        "delay_start_stop_ps_npts": "",
        "detector_position_cm": "10",
        "notes": "Motor Position 340mm, Detector distance 100mm",
        "number_of_frames_per_step": "",
        "run_": "4",
        "run_quality_check": "",
        "sample": "LaB6",
        "scan_type": "Static ",
        "thz": "",
        "transmission": "",
        "well": "4",
        "xy_position_of_beam_on_sample": "",
    },
    "5": {
        "cell": "1",
        "delay_start_stop_ps_npts": "",
        "detector_position_cm": "4",
        "notes": "Motor position 279 mm, detector distance43mm",
        "number_of_frames_per_step": "",
        "run_": "5",
        "run_quality_check": "",
        "sample": "HVG + Kapton",
        "scan_type": "Attenuation Series",
        "thz": "",
        "transmission": "",
        "well": "5",
        "xy_position_of_beam_on_sample": "",
    },
    "6": {
        "cell": "2",
        "delay_start_stop_ps_npts": "",
        "detector_position_cm": "",
        "notes": "Background from Quartz, just looking at window. ",
        "number_of_frames_per_step": "",
        "run_": "6",
        "run_quality_check": "",
        "sample": "Empty",
        "scan_type": "Static ",
        "thz": "",
        "transmission": "",
        "well": "7",
        "xy_position_of_beam_on_sample": "",
    },
}
