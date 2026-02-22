import argparse

from bglk_euxfel.version import __version__  # noqa


def main():
    parser = argparse.ArgumentParser(
        prog="bglk-euxfel",
        description=(
            "Scripts and code for XFEL experiments to do ultrafast "
            "PDF measurements\n\n"
            "For more information, visit: "
            "https://github.com/billingegroup/bglk-euxfel/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the program's version number and exit",
    )

    args = parser.parse_args()

    if args.version:
        print(f"bglk-euxfel {__version__}")
    else:
        # Default behavior when no arguments are given
        parser.print_help()


if __name__ == "__main__":
    main()
