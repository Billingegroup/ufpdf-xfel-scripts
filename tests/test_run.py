from ufpdf_xfel_scripts.lcls.run import Run

run_kwargs = {"10": {"run_number": "10"}}


def test_run_constructor(user_filesystem, mocker):
    mocker.patch(
        "ufpdf_xfel_scripts.lcls.run.experiment_data_dir", user_filesystem
    )
    actual = Run(10, run_kwargs)
    expected_run_number = 10
    assert actual.run_number == expected_run_number
