"""
lcls_run.py
-----------
A module providing the `Run` class, which loads and reduces LCLS pump-probe
diffraction data from an HDF5 file, storing only the reduced arrays as
attributes (the large raw arrays are discarded after reduction).
"""

import logging
import warnings

import h5py
import numpy as np
from diffpy.morph.morphpy import morph_arrays
from diffpy.utils.parsers import load_data

from ufpdf_xfel_scripts.lcls.paths import (
    experiment_data_dir,
    synchrotron_data_dir,
)

warnings.filterwarnings("ignore")
logging.getLogger("diffpy.pdfgetx").setLevel(logging.ERROR)
logging.getLogger("diffpy.pdfgetx.user").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


class Run:
    """Loads and reduces a single LCLS pump-probe run.

    Parameters
    ----------
    run_number : int
        The run number to load (e.g. 22).
    background_number : int
        The background run number (e.g. 1).
    sample_name : str
        The hort sample label used in file names (e.g. 'NSPSe').
    sample_composition : str
        The chemical composition string for PDFGetter (e.g. 'Na11SnPSe12').
    instrument : str
        The instrument prefix (e.g. 'mfx').
    experiment_number : str
        The experiment identifier (e.g. 'l1044925').
    target_id : int
        The index of the delay to use as the morph target (default 0).
    q_min : float
        The lower Q bound for the I(Q) figure of merit (default 9).
    q_max : float
        the upper Q bound for the I(Q) figure of merit (default 9.5).
    r_min_fom : float
        The lower r bound for the G(r) figure of merit (default 2).
    r_max_fom : float
        The upper r bound for the G(r) figure of merit (default 5).
    q_min_morph : float
        the lower Q bound for morph normalisation (default 0).
    q_max_morph : float
        The upper Q bound for morph normalisation (default 12).
    scale : float
        The initial scale parameter for morphing (default 1.01).
    stretch : float or None
        The initial stretch parameter for morphing (default None).
    smear : float or None
        The initial smear parameter for morphing (default None).
    points_away_t0_plot_on_off : int
        The number of delay points away from t0 to select for on/off
        plots (default 0).
    verbose : bool
        The verbosity for debugging and assessing (default, False, is
        low verbosity).

    Attributes
    ----------
    q : np.ndarray
        Q-grid (1-D, shape (n_q,)).
    delays : np.ndarray
        Sorted unique delay times in ps (1-D, shape (n_delays,)).
    Is_on : np.ndarray
        Delay-averaged, sorted pump-ON I(Q) (shape (n_delays, n_q)).
    Is_off : np.ndarray
        Delay-averaged, sorted pump-OFF I(Q) (shape (n_delays, n_q)).
    target_delay : float
        The delay value used as the morph target.
    raw_delays : dict
        Dict keyed by delay time containing raw [q, on, off, diff, ...] lists.
    morph_delays : dict
        Dict keyed by delay time containing morphed [q, on, off,
        diff, ...] lists.
    delay_scan : bool
        True if the run contains a delay scan, False otherwise.
    q_synchrotron : np.ndarray
        Q-grid from the synchrotron reference file.
    fq_synchrotron : np.ndarray
        F(Q) from the synchrotron reference file.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        run_number,
        background_number,
        sample_name,
        sample_composition,
        instrument,
        experiment_number,
        number_of_static_samples=11,
        target_id=0,
        q_min=9,
        q_max=9.5,
        r_min_fom=2,
        r_max_fom=5,
        q_min_morph=0,
        q_max_morph=12,
        scale=1.01,
        stretch=None,
        smear=None,
        points_away_t0_plot_on_off=0,
        verbose=False,
    ):
        # --- store run-level metadata ---
        self.run_number = run_number
        self.background_number = background_number
        self.sample_name = sample_name
        self.sample_composition = sample_composition
        self.instrument = instrument
        self.experiment_number = experiment_number
        self.number_of_static_samples = number_of_static_samples
        self.verbose = verbose

        # --- store setup parameters ---
        self.target_id = target_id
        self.q_min = q_min
        self.q_max = q_max
        self.r_min_fom = r_min_fom
        self.r_max_fom = r_max_fom
        self.morph_params = {
            "xmin": q_min_morph,
            "xmax": q_max_morph,
            "scale": scale,
            "stretch": stretch,
            "smear": smear,
        }
        self.points_away_t0_plot_on_off = points_away_t0_plot_on_off

        # --- run the reduction pipeline ---
        self._load()
        self._reduce()
        self._morph()
        self._cleanup()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_evenly(arr, n_points):
        N = len(arr)
        indices = np.linspace(0, N - 1, n_points, dtype=int)
        return arr[indices]

    def _average_equal_times(self):
        # average repeated delays
        self.unique_delays = np.unique(self.delays)

        Is_avg_on = []
        Is_avg_off = []
        for ud in self.unique_delays:
            mask_on = (self.delays == ud) & self.laser_mask
            mask_off = (self.delays == ud) & ~self.laser_mask
            Is_avg_on.append(np.nanmean(self._Is_raw[mask_on], axis=0))
            Is_avg_off.append(np.nanmean(self._Is_raw[mask_off], axis=0))

        self.Is_avg_on = np.array(Is_avg_on)
        self.Is_avg_off = np.array(Is_avg_off)
        self.raw_delays = {}
        for i, step in enumerate(self.unique_delays):
            self.delay_dict = self._build_delay_dict(
                self.raw_delays,
                step,
                self.q,
                self.Is_avg_on[i],
                self.Is_avg_off[i],
            )

    def _build_delay_dict(self, delay_dict, delay_time, q, on, off):
        """Append one delay entry (mirrors the notebook's
        build_delay_dict)."""
        diff = on - off
        q_min, q_max = self.q_min, self.q_max

        qmin_idx = find_nearest(q, q_min) if q_min is not None else 0
        qmax_idx = find_nearest(q, q_max) if q_max is not None else -1

        l1_diff = np.sum(np.abs(diff[qmin_idx:qmax_idx]))
        l2_diff = np.sum(diff[qmin_idx:qmax_idx] ** 2)
        diff_int = np.sum(diff[qmin_idx:qmax_idx])
        i_sum_off = np.sum(off[qmin_idx:qmax_idx])
        i_sum_on = np.sum(on[qmin_idx:qmax_idx])

        delay_dict[delay_time] = [
            q,
            on,
            off,
            diff,
            l1_diff,
            i_sum_off,
            i_sum_on,
            l2_diff,
            diff_int,
        ]
        return delay_dict

    def _build_parameters_dict(
        self,
        parameter_dict,
        delay_time,
        morph_parameters_on,
        morph_parameters_off,
    ):
        """Append one delay entry (mirrors the notebook's
        build_delay_dict)."""
        parameter_dict[delay_time] = [
            morph_parameters_on,
            morph_parameters_off,
        ]
        return parameter_dict

    def _cleanup(self):
        del self._Is_raw

    # ------------------------------------------------------------------
    # pipeline steps
    # ------------------------------------------------------------------

    def _load(self):
        """Load the HDF5 file and synchrotron reference; store only
        reduced arrays."""
        str_run_number = str(self.run_number).zfill(4)
        h5_filename = (
            f"{self.instrument}{self.experiment_number}_Run{str_run_number}.h5"
        )
        input_path = experiment_data_dir / h5_filename
        synchrotron_path = synchrotron_data_dir / f"{self.sample_name}_room.fq"

        # synchrotron reference
        self.q_synchrotron, self.fq_synchrotron = load_data(
            synchrotron_path, unpack=True
        )

        if not input_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {input_path}")

        with h5py.File(input_path, "r") as f:
            qs = f["jungfrau"]["pyfai_q"][:]
            Is_raw = f["jungfrau"]["pyfai_azav"][:]
            Is_raw = np.nanmean(Is_raw, axis=1)

            self.delay_scan = (
                "scan" in f
            )  # true if scan exists in the dataset, false otherwise
            if self.delay_scan:
                delays = f["scan"]["lxt_ttc"][:].squeeze() * 1e12
                self.target_delay = delays[self.target_id]
            else:
                delays = None  # filled below
                self.target_delay = None

            laser_mask = f["lightStatus"]["laser"][:].astype(bool)
            xray_mask = f["lightStatus"]["xray"][:].astype(bool)

        if self.verbose:
            print("shape of qs:", qs.shape)
            print("shape of Is_raw:", Is_raw.shape)
            print("delay_scan:", self.delay_scan)

        # separate x-ray darks and lights
        self.q = qs[0]
        self._Is_raw = Is_raw[xray_mask].copy()
        self.darks = Is_raw[~xray_mask].copy()
        self.delays = delays[xray_mask].copy()
        self.laser_mask = laser_mask[xray_mask].copy()
        return

    def _reduce(self):
        """Build raw_delays dict (unorphed) from the reduced arrays."""

        if not self.delay_scan:
            self.subsample = self._sample_evenly(
                self.delays, self.number_of_static_samples
            )
        else:
            self.subsample = self._average_equal_times()

    def _morph(self):
        """Apply diffpy.morph to each delay and store results in
        morph_delays."""
        params = self.morph_params
        target = self.raw_delays[self.target_delay]
        target_table = np.column_stack([target[0], target[1]])

        self.morph_delays = {}
        self.morph_parameters = {}
        for delay_t, data in self.raw_delays.items():
            x = data[0]
            y_on = data[1]
            y_off = data[2]

            morph_on_table = np.column_stack([x, y_on])
            morph_off_table = np.column_stack([x, y_off])

            # fit morph parameters
            self.morph_parameters_on, _ = morph_arrays(
                morph_on_table, target_table, **params
            )
            self.morph_parameters_off, _ = morph_arrays(
                morph_off_table, target_table, **params
            )

            # apply parameters without refining
            _, table_on_full = morph_arrays(
                morph_on_table,
                target_table,
                scale=self.morph_parameters_on.get("scale"),
                stretch=self.morph_parameters_on.get("stretch"),
                smear=self.morph_parameters_on.get("smear"),
                apply=True,
            )
            _, table_off_full = morph_arrays(
                morph_off_table,
                target_table,
                scale=self.morph_parameters_off.get("scale"),
                stretch=self.morph_parameters_off.get("stretch"),
                smear=self.morph_parameters_off.get("smear"),
                apply=True,
            )

            on_morph = table_on_full[:, 1]
            off_morph = table_off_full[:, 1]

            self.morph_delays = self._build_delay_dict(
                self.morph_delays, delay_t, x, on_morph, off_morph
            )
            self.morph_parameters = self._build_parameters_dict(
                self.morph_parameters,
                delay_t,
                self.morph_parameters_on,
                self.morph_parameters_off,
            )
