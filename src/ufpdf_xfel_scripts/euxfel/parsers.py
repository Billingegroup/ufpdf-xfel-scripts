from argparse import ArgumentParser

# from gooey import Gooey, GooeyParser
# from diffpy.utils.scattering_objects.diffraction_objects
# import QQUANTITIES, XQUANTITIES
# from diffpy.utils.tools import get_package_info, get_user_info


def define_arguments():
    args = [
        {
            "name": ["run_number"],
            "help": "The run number to process",
            "type": int,
        },
        {
            "name": ["-n", "--sample-name"],
            "help": "The name of the sample for annotating figures."
            "The default is an"
            "empty string.",
            "type": str,
            "default": "",
        },
        {
            "name": ["-m", "--q-min-assess"],
            "help": "The lower Q-bound of the range over which to do "
            "an assessment. The default is the beginning of "
            "the measured Q-range. The assessment is carried "
            "out over a Q-range between q-min-assess and "
            "q-max-assess.",
            "type": float,
            "default": None,
        },
        {
            "name": ["-x", "--q-max-assess"],
            "help": "The upper Q-bound of the range over which to do "
            "an assessment. The default is the end of the measured "
            "Q-range. The assessment is carried out over a Q-range "
            "between q-min-assess and q-max-assess.",
            "type": float,
            "default": None,
        },
        {
            "name": ["--normalize-to-target-id"],
            "help": "The id of the scan to use as a target for the "
            "normalization expressed as an integer.  All other "
            "diffraction patterns will be normalized to this "
            "pattern. Any pattern can be used as long as it is "
            "population.  The Default is the pattern in the middle "
            "of the series. E.g., for a series of 11 patterns it "
            "would be 5 (patterns are zero indexed).",
            "type": int,
            "default": None,
        },
        {
            "name": ["--q-min-normalize"],
            "help": "The minimum Q-value to for the normalization.  The "
            "Default is the beginning of the measured Q-range.  "
            "The data will be normalized by comparing over a range "
            "from q-min-normalize to q-max-normalize.",
            "type": float,
            "default": None,
        },
        {
            "name": ["--q-max-normalize"],
            "help": "The maximum Q-value to for the normalization. "
            "The default is the end of the measured Q-range.  The "
            "data will be normalized by comparing over a range "
            "from q-min-normalize to q-max-normalize.",
            "type": float,
            "default": None,
        },
        {
            "name": ["--t0"],
            "help": "The value of the time delay in the data that "
            "corresponds to t_0 where the pump and the probe "
            "arrive coincidentally.  The default value is -751.0 ps",
            "type": float,
            "default": -751.0,
        },
        {
            "name": ["--initial-scale"],
            "help": "The initial value for the scale factor in the adaptive "
            "normalization scheme."
            "The default is 1.01.",
            "type": float,
            "default": 1.01,
        },
        {
            "name": ["--initial-stretch"],
            "help": "The initial value for the stretch factor in the "
            "adaptive normalization scheme. The default is to "
            "suppress this transformation.  Specify "
            "'--initial-stretch 0.01', or some similar number, "
            "to turn on this transformation in the normalization "
            "step.",
            "type": float,
            "default": None,
        },
        {
            "name": ["--initial-smear"],
            "help": "The initial value for the smear factor in the adaptive "
            "normalization scheme. The default is to suppress this "
            "transformation.  Specify '--initial-smear 0.005', or  "
            "some similar number, to turn on this transformation in "
            "the normalization step.",
            "type": float,
            "default": None,
        },
        {
            "name": ["--path-to-data"],
            "help": "A string representation of the relative path from your "
            "current directory to the directory containing the data "
            "to assess. E.g., '../my/data'. The default is '.' which "
            "would imply that the data are all in the current "
            "directory.",
            "type": str,
            "default": ".",
        },
    ]
    # # {
    # #     "name": ["input"],
    # #     "help": (
    # #     ),
    #     "nargs": "+",
    #     "widget": "MultiFileChooser",
    # },
    # {
    #     "name": ["-u", "--user-metadata"],
    #     "help": (
    #         "Specify key-value pairs to be loaded into metadata "
    #         "using the format key=value. "
    #         "Separate pairs with whitespace, "
    #         "and ensure no whitespaces before or after the = sign. "
    #         "Avoid using = in keys. If multiple = signs are present, "
    #         "only the first separates the key and value. "
    #         "If a key or value contains whitespace, enclose it in quotes. "
    #         "For example, facility='NSLS II', "
    #         "'facility=NSLS II', beamline=28ID-2, "
    #         "'beamline'='28ID-2', 'favorite color'=blue, "
    #         "are all valid key=value items."
    #     ),
    #     "nargs": "+",
    #     "metavar": "KEY=VALUE",
    # },
    return args


def get_args(override_cli_inputs=None):
    p = ArgumentParser("")
    for arg in define_arguments():
        kwargs = {
            key: value
            for key, value in arg.items()
            if key != "name" and key != "widget"
        }
        p.add_argument(*arg["name"], **kwargs)
    args = p.parse_args(override_cli_inputs)
    return args


def _load_key_value_pair(s):
    items = s.split("=")
    key = items[0].strip()
    if len(items) > 1:
        value = "=".join(items[1:])
    return (key, value)


# @Gooey(required_cols=1, optional_cols=1, program_name="Labpdfproc GUI")
# def gooey_parser():
#     p = GooeyParser()
#     for arg in define_arguments():
#         kwargs = {key: value for key, value in arg.items() if key != "name"}
#         p.add_argument(*arg["name"], **kwargs)
#     args = p.parse_args()
#     return args


def preprocessing_args(args):
    """Perform preprocessing on the provided argparse Namespace.

    Parameters
    ----------
    args argparse.Namespace
        the arguments from the parser, default is None

    Returns
    -------
    the updated argparse Namespace with arguments preprocessed
    """
    metadata = {"args": args.__dict__}
    # metadata.update(get_package_info("euxfel"))
    # metadata.update(get_user_info())
    # load_package_info(args))
    # args = load_user_info(args)
    # args = set_input_lists(args)
    # args.output_directory = set_output_directory(args)
    # args = set_wavelength(args)
    # args = set_xtype(args)
    # args = set_mud(args)
    # args = load_user_metadata(args)
    return metadata
