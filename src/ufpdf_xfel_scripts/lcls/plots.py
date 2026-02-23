import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ufpdf_xfel_scripts.lcls.run import find_nearest


def plot_delay_scans(scan_dict, run):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, figsize=(8, 16))
    keys = [key for key in scan_dict.keys()]
    # delay_times_l1 = [delay[4] for delay in scan_dict.values()]
    delay_times_off = [delay[5] for delay in scan_dict.values()]
    delay_times_on = [delay[6] for delay in scan_dict.values()]
    delay_times_l2 = [delay[7] for delay in scan_dict.values()]
    delay_times_int = [delay[8] for delay in scan_dict.values()]
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(keys))]
    key_to_color_idx = {key: i for i, key in enumerate(keys)}
    for key, delay in scan_dict.items():
        # if key == time_away_t0:
        # on_plot = scan[1]
        # off_plot = scan[2]
        color = colors[key_to_color_idx[key]]
        ax0.plot(delay[0], delay[1], label=key, color=color)
        ax1.plot(delay[0], delay[2], label=key, color=color)
        ax2.plot(delay[0], delay[3], label=key, color=color)
        if run.q_min is not None:
            ax2.axvline(x=run.q_min, color="red")
        if run.q_max is not None:
            ax2.axvline(x=run.q_max, color="red")

    q_vals = list(scan_dict.values())[0][0]
    on_minus_off_matrix = np.array([delay[3] for delay in scan_dict.values()])
    delay_times = np.array([key for key in scan_dict.keys()])
    sort_idx = np.argsort(delay_times)
    on_minus_off_matrix = on_minus_off_matrix[sort_idx]
    delay_times = delay_times[sort_idx]
    extent = [q_vals[0], q_vals[-1], delay_times[0], delay_times[-1]]
    ax3.imshow(
        on_minus_off_matrix,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
    )
    ax3.invert_yaxis()
    if run.q_min is not None:
        ax3.axvline(x=run.q_min, color="red")
    if run.q_max is not None:
        ax3.axvline(x=run.q_max, color="red")
    ax3.set_xlabel("Q [1/A]")
    ax3.set_ylabel("Time scan (ps)")
    ax3.set_title("On - Off")
    ax4.plot(keys, delay_times_off, marker="o", linestyle="-", label="off")
    ax4.plot(keys, delay_times_on, marker="o", linestyle="-", label="on")
    ax5.plot(
        keys, np.sqrt(delay_times_l2), marker="o", linestyle="-", label="diff"
    )
    ax5.plot(keys, delay_times_int, marker="o", linestyle="-", label="diff")
    ax0.set_xlabel("Q [1/A]")
    ax0.set_ylabel("Pump On Intensity [a.u.]")
    ax1.set_xlabel("Q [1/A]")
    ax1.set_ylabel("Pump Off Intensity [a.u.]")
    ax2.set_xlabel("Q [1/A]")
    ax2.set_ylabel("On-Off Intensity [a.u.]")
    ax4.set_xlabel("Time scan (ps)")
    ax4.set_ylabel("Sum intensities")
    ax5.set_xlabel("Time scan (ps)")
    ax5.set_ylabel("RMS")
    ax4.legend()
    ax0.set_title(
        f"sample = {run.sample_name}, run = {run.run_number}, "
        f"qmin = {run.q_min}, qmax = {run.q_max}"
    )
    ax1.set_title(f"run = {run.run_number}")
    ax2.set_title(f"I(q) On - I(q) Off run = {run.run_number}")
    ax5.set_title(
        f"Figure of Merit run = {run.run_number}, run.q_min = {run.q_min}, "
        f"run.q_max = {run.q_max}"
    )
    plt.tight_layout()
    plt.show()


def plot_static_scans(scan_dict, run):
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    keys = sorted(scan_dict.keys())
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(keys))]
    key_to_color_idx = {key: i for i, key in enumerate(keys)}

    roi_sum_I = []
    roi_l1 = []
    roi_rms = []
    roi_int = []

    for key in keys:
        color = colors[key_to_color_idx[key]]

        q = scan_dict[key][0]
        Iq = scan_dict[key][1]
        diff = scan_dict[key][2]

        ax0.plot(q, Iq, label=str(key), color=color)
        ax1.plot(q, diff, label=str(key), color=color)

        # ROI indices
        if run.q_min is not None:
            i0 = find_nearest(q, run.q_min)
        else:
            i0 = 0
        if run.q_max is not None:
            i1 = find_nearest(q, run.q_max)
        else:
            i1 = len(q) - 1
        if i1 < i0:
            i0, i1 = i1, i0

        sl = slice(i0, i1 + 1)

        roi_sum_I.append(np.nansum(Iq[sl]))
        roi_l1.append(np.nansum(np.abs(diff[sl])))
        roi_rms.append(np.sqrt(np.nanmean(diff[sl] ** 2)))
        roi_int.append(np.nansum(diff[sl]))

    if run.q_min is not None:
        ax0.axvline(x=run.q_min, color="red")
        ax1.axvline(x=run.q_min, color="red")
    if run.q_max is not None:
        ax0.axvline(x=run.q_max, color="red")
        ax1.axvline(x=run.q_max, color="red")

    ax2.plot(keys, roi_sum_I, marker="o", linestyle="-", label="ROI sum I(q)")
    # ax2.plot(keys, roi_rms,   marker='o', linestyle='-', label='ROI RMS(ΔI)')
    ax2.plot(keys, roi_int, marker="o", linestyle="-", label="ROI ∑ΔI")
    # ax2.plot(keys, roi_l1,    marker='o', linestyle='-', label='ROI ∑|ΔI|')

    ax0.set_xlabel("Q [1/A]")
    ax0.set_ylabel("Intensity [a.u.]")
    ax1.set_xlabel("Q [1/A]")
    ax1.set_ylabel("I(q) - Iref(q) [a.u.]")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("ROI metric [a.u.]")

    ax0.set_title(
        f"sample = {run.sample_name}, run = {run.run_number}, "
        f"qmin = {run.q_min}, qmax = {run.q_max}"
    )
    ax1.set_title(f"I(q) - I(q)_ref run = {run.run_number}")
    ax2.set_title(
        f"ROI metrics run = {run.run_number}, run.q_min = {run.q_min}, "
        f"run.q_max = {run.q_max}"
    )

    plt.tight_layout()
    plt.show()
