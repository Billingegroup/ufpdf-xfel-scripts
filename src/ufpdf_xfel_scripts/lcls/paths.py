from pathlib import Path

experiment_symlink = "data-202601"
experiment_number = "mfxl1044925"

experiment_dir = (
    Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / ".."
    / ".."
    / experiment_symlink
    / experiment_number
)
synchrotron_data_dir = (
    Path(__file__).parent / ".." / ".." / "data" / "synchrotron-data"
)
synchrotron_data_dir = synchrotron_data_dir.resolve()
experiment_dir = experiment_dir.resolve()
setup_data_dir = experiment_dir / "scratch" / "setup-data"
experiment_data_dir = experiment_dir / "hdf5" / "smalldata"

global_output_dir = experiment_dir / "scratch" / "users"


def main():
    print("experiment_dir=", experiment_dir)
    print("synchrotron_data_dir=", synchrotron_data_dir)
    print("experiment_data_dir=", experiment_data_dir)
    print("global_output_dir=", global_output_dir)


if __name__ == "__main__":
    main()
