from ufpdf_xfel_scripts.lcls.run import Run

run_kwargs = {"10": {"run_number": "10"}}


def test_run_constructor():
    actual = Run(10, run_kwargs)
    expected_run_number = 10
    assert actual.run_number == expected_run_number
