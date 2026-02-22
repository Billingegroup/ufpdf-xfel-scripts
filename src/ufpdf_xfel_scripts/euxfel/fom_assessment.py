import diffpy.morph.morph_api as morph
import matplotlib
import numpy as np
from bglk_euxfel.functions import (
    build_delay_dict,
    build_paths,
    find_nearest,
    set_limits,
)
from bglk_euxfel.parsers import get_args, preprocessing_args
from bglk_euxfel.plotters import assessment_plotter

matplotlib.use("TkAgg")

# run_number = 204
# sample_name = 'Bismuth'
# target_id = 5
# q_min = 4.2
# q_max = 4.4
# q_min_morph = None
# q_max_morph = None
# scale = 1.01
# stretch = 0.01
# smear = 0.005
# baselineslope = None
# qdamp = None
# t0 = -751 # -749.8 for earlier days
# points_away_t0_plot_on_off = -6
# rel_path_from_here_to_data = '.'
# rel_path_from_here_to_data = '../../doc/example/input_data'


def main():
    # args = (
    #     gooey_parser()
    #     if len(sys.argv) == 1 or "--gui" in sys.argv
    #     else get_args()
    # )
    args = get_args()
    metadata = preprocessing_args(args)

    paths, metadata = build_paths(args, metadata)
    on = np.load(paths.get("on_data_path"))
    off = np.load(paths.get("off_data_path"))
    q = np.load(paths.get("q_path"))
    args = set_limits(args, q)
    delay = np.load(paths.get("delay_positions_path"))
    delay = args.t0 - delay  # remove the t0 offset
    if args.normalize_to_target_id is None:
        midpoint = (max(delay) + min(delay)) / 2.0
        args.normalize_to_target_id = find_nearest(delay, midpoint)
    target_key = delay[args.normalize_to_target_id]
    morph_cfg = morph.morph_default_config(
        scale=args.initial_scale,
        stretch=args.initial_stretch,
        smear=args.initial_smear,
    )

    raw_delays = {}
    for i, step in enumerate(delay):
        raw_delays = build_delay_dict(
            raw_delays,
            step,
            q,
            on[i],
            off[i],
            q_min=args.q_min_assess,
            q_max=args.q_max_assess,
        )

    morph_delays = {}
    target = raw_delays[target_key]
    for delay_t, data in raw_delays.items():
        morphed = morph.morph(
            data[0],
            data[1],
            data[0],
            target[1],
            rmin=args.q_min_normalize,
            rmax=args.q_max_normalize,
            **morph_cfg,
        )  # on
        _, on_morph, _, _ = morphed["morph_chain"].xyallout
        morphed = morph.morph(
            data[0],
            data[2],
            data[0],
            target[1],
            rmin=args.q_min_normalize,
            rmax=args.q_max_normalize,
            **morph_cfg,
        )  # off
        _, off_morph, _, _ = morphed["morph_chain"].xyallout
        morph_delays = build_delay_dict(
            morph_delays,
            delay_t,
            data[0],
            on_morph,
            off_morph,
            morphed=morphed,
            q_min=args.q_min_assess,
            q_max=args.q_max_assess,
        )
    # if points_away_t0_plot_on_off is not None:
    #     t0_index = find_nearest(delay,0)
    #     time_away_t0_index_plot = t0_index + points_away_t0_plot_on_off
    #     time_away_t0 = delay[time_away_t0_index_plot]

    # plot_function(raw_delays)
    assessment_plotter(morph_delays, args)
    print(metadata)


if __name__ == "__main__":
    main()
