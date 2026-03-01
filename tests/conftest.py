import json
from pathlib import Path

import h5py
import numpy as np
import pytest

N_EVENTS = 10  # number of shots
N_Q = 500  # q-points
N_AZ = 13  # azimuthal bins (must cover your azimuthal_selector range)


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    h5_file_dir = base_dir
    h5_file_dir.mkdir(parents=True, exist_ok=True)
    h5_file_path = h5_file_dir / "MFX10449_Run0010.h5"

    with h5py.File(h5_file_path, "w") as f:
        # jungfrau group
        jungfrau = f.create_group("jungfrau")
        jungfrau.create_dataset("pyfai_q", data=np.linspace(0.1, 10.0, N_Q))
        # pyfai_azav shape: (N_EVENTS, N_AZ, N_Q)
        jungfrau.create_dataset(
            "pyfai_azav",
            data=np.random.rand(N_EVENTS, N_AZ, N_Q).astype(np.float32),
        )

        # beam monitors
        f.create_dataset(
            "MfxDg1BmMon/totalIntensityJoules", data=np.random.rand(N_EVENTS)
        )
        f.create_dataset(
            "MfxDg2BmMon/totalIntensityJoules", data=np.random.rand(N_EVENTS)
        )

        # timing
        f.create_dataset("/tt/fltpos_ps", data=np.random.rand(N_EVENTS))
        f.create_dataset(
            "/timestamp", data=np.arange(N_EVENTS, dtype=np.float64)
        )

        # light status
        light = f.create_group("lightStatus")
        light.create_dataset(
            "laser", data=np.tile([1, 0], N_EVENTS // 2).astype(np.uint8)
        )
        light.create_dataset("xray", data=np.ones(N_EVENTS, dtype=np.uint8))

        # optional: scan group (set delay_scan=True)
        # comment this block out to test the non-scan case
        scan = f.create_group("scan")
        scan.create_dataset(
            "mfx_lxt_fast2",
            data=np.linspace(-1e-12, 1e-12, N_EVENTS).reshape(-1, 1),
        )

    yield tmp_path
