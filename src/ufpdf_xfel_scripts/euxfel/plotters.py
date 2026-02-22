import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.widgets import Button, RadioButtons, Slider


def assessment_plotter(delays, args):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(8, 14))
    keys = [key for key in delays.keys()]
    # delay_times_l1 = [delay[4] for delay in delays.values()]
    delay_times_off = [delay[5] for delay in delays.values()]
    delay_times_on = [delay[6] for delay in delays.values()]
    delay_times_l2 = [delay[7] for delay in delays.values()]
    # delay_times_rw = [delay[9] for delay in delays.values()]
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(keys))]
    key_to_color_idx = {key: i for i, key in enumerate(keys)}
    # cmap = matplotlib.colormaps.get_cmap('viridis', len(keys))
    # key_to_color_idx = {key: i for i, key in enumerate(keys)}
    max_diff, min_diff = -np.inf, np.inf
    for key, delay in delays.items():
        # if key == time_away_t0:
        #     on_plot = delay[1]
        #     off_plot = delay[2]
        color = colors[key_to_color_idx[key]]
        max_diff = max([max_diff, max(delay[3][:-20])])
        min_diff = min([max_diff, min(delay[3][:-20])])
        ax0.plot(delay[0], delay[1], label=key, color=color)
        ax1.plot(delay[0], delay[3], label=key, color=color)
        ax1.axvline(x=args.q_min_assess, color="red")
        ax1.axvline(x=args.q_max_assess, color="red")
    # ax1.plot(delay[0],args.on_plot,label='on',color='black')
    # ax1.plot(delay[0],args.off_plot,label='off',color='orange')
    ax2.plot(keys, delay_times_off, marker="o", linestyle="-", label="off")
    ax2.plot(keys, delay_times_on, marker="o", linestyle="-", label="on")
    ax3.plot(
        keys, np.sqrt(delay_times_l2), marker="o", linestyle="-", label="diff"
    )
    # ax3.plot(keys,np.sqrt(delay_times_rw),marker='o',
    #    linestyle='-',label='diff')
    ax0.set_xlabel("Q [1/A]")
    ax0.set_ylabel("Pump On Intensity [a.u.]")
    ax1.set_xlabel("Q [1/A]")
    ax1.set_ylim([min_diff, max_diff])
    ax1.set_ylabel("Pump Off Intensity [a.u.]")
    ax2.set_xlabel("Q [1/A]")
    ax2.set_ylabel("On-Off Intensity [a.u.]")
    ax3.set_xlabel("Time delay (ps)")
    ax3.set_ylabel("Sum intensities")
    # ax4.set_xlabel('Time delay (ps)')
    # ax4.set_ylabel('RMS')
    ax3.legend()
    ax0.set_title(
        f"sample = {args.sample_name}, run = {args.run_number}, "
        f"qmin = {args.q_min_assess:.2f}, qmax = {args.q_max_assess:.2f}"
    )
    # ax1.set_title(f'I(q) On vs Off, time_delay ={args.time_away_t0},
    # run = {args.run_number}')
    ax2.set_title(f"I(q) On - I(q) Off run = {args.run_number}")
    # ax4.set_title(f'Figure of Merit run = {args.run_number},
    # q_min = {args.q_min_assess:.2f}, q_max = {args.q_max_assess:.2f}')
    plt.tight_layout()
    plt.show()
