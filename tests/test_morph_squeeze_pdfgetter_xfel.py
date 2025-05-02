from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from diffpy.morph.morph_api import morph, morph_default_config
from diffpy.pdfgetx.pdfconfig import PDFConfig
from diffpy.pdfgetx.pdfgetter import PDFGetter


def pdfgetter_function(x, y, rpoly, bgscale):
    cfg = PDFConfig()
    cfg.composition = "Na3Sb0.09W0.1S4"
    background = np.loadtxt(
        Path().cwd() / "test_data" / "kapton_background.txt"
    )
    cfg.background = background
    cfg.mode = "xray"
    cfg.bgscale = bgscale
    cfg.dataformat = "QA"
    cfg.qmin = 0.1
    cfg.qmax = 23.5
    cfg.qmaxinst = 23.5
    cfg.rpoly = rpoly
    pg1 = PDFGetter(cfg)
    _, _ = pg1(x=x, y=y)
    q, fq = pg1.fq
    return np.interp(x, q, fq)


@pytest.mark.parametrize("rpoly", [1, 1.25, 1.5])
def test_morph_pdfgetter(rpoly):
    cwd = Path().cwd()
    input_path = cwd / "test_data"
    iq_path = input_path / "iq_xfel.txt"
    fq_path = input_path / "NSWS.fq"
    iq_load = np.loadtxt(iq_path)
    fq_load = pd.read_csv(fq_path, delimiter=" ", skiprows=26, index_col=0)
    x_morph = iq_load[:, 0]
    y_morph = iq_load[:, 1]
    x_target = np.array(fq_load.index.tolist(), dtype=float)
    y_target = fq_load.iloc[:, 0].values
    parameters = morph_default_config(
        scale=0.01,
        squeeze={"a0": 0.01, "a1": 1e-5, "a2": 0, "a3": 1e-4, "a4": 0},
        funcy={"rpoly": rpoly, "bgscale": 1},
    )
    parameters["function"] = pdfgetter_function
    morph_rv = morph(
        x_morph,
        y_morph,
        x_target,
        y_target,
        rmin=None,
        rmax=None,
        **parameters,
    )
    fitted = morph_rv["morphed_config"]["funcy"]
    x_morph_out, y_morph_out, x_target_out, y_target_out = morph_rv[
        "morph_chain"
    ].xyallout
    # Plotting code for PR 31
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x_morph, y_morph, color="gray", label="Morph")
    axes[0].set_xlabel("Q (1/Å)")
    axes[0].set_ylabel("I(Q)")
    axes[0].set_xlabel("Q (1/Å)")
    axes[0].legend()
    axes[1].plot(x_target, y_target, color="purple", label="Target")
    axes[1].plot(x_morph_out, y_morph_out, "--", color="gold", label="Morphed")
    axes[1].set_title(f'rpoly_init={rpoly}, rpoly_refined={fitted["rpoly"]}')
    axes[1].set_ylabel("F(Q)")
    axes[1].set_xlabel("Q (1/Å)")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(x_target, y_target, color="purple", label="Target")
    plt.plot(x_morph_out, y_morph_out, "--", color="gold", label="Morphed")
    plt.xlim([0, 11])
    plt.ylim([-0.8, 2.5])
    plt.ylabel("F(Q)")
    plt.xlabel("Q (1/Å)")
    plt.legend()
    plt.title(f'rpoly_init={rpoly}, rpoly_refined={fitted["rpoly"]}')
    plt.show()
