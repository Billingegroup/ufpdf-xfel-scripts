from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from diffpy.morph.morph_api import morph, morph_default_config
from diffpy.pdfgetx.pdfconfig import PDFConfig
from diffpy.pdfgetx.pdfgetter import PDFGetter


def pdfgetter_function(x, y, rpoly):
    cfg = PDFConfig()
    cfg.composition = "C"
    cfg.dataformat = "QA"
    cfg.qmin = 0
    cfg.qmax = 27.9863
    cfg.qmaxinst = 27.9863
    cfg.rmax = 30
    cfg.rmin = 0.0
    cfg.rstep = 0.01
    cfg.rpoly = rpoly
    pg1 = PDFGetter(cfg)
    _, _ = pg1(x=x, y=y)
    q, fq = pg1.fq
    return np.interp(x, q, fq)


@pytest.mark.parametrize("rpoly", [0.9, 1.05, 1.2, 1.3, 1.4])
def test_morph_pdfgetter(rpoly):
    cwd = Path().cwd()
    input_path = cwd / "test_data"
    iq_path = input_path / "hard_carbon.iq"
    fq_path = input_path / "hard_carbon.fq"
    iq_load = pd.read_csv(iq_path, delimiter=" ", skiprows=30, index_col=0)
    fq_load = pd.read_csv(fq_path, delimiter=" ", skiprows=30, index_col=0)
    x_morph = np.array(iq_load.index.tolist(), dtype=float)
    y_morph = iq_load.iloc[:, 0].values
    x_target = np.array(fq_load.index.tolist(), dtype=float)
    y_target = fq_load.iloc[:, 0].values
    parameters = morph_default_config(funcy={"rpoly": rpoly})
    parameters["function"] = pdfgetter_function
    morph_rv = morph(x_morph, y_morph, x_target, y_target, **parameters)
    fitted = morph_rv["morphed_config"]["funcy"]
    x_morph_out, y_morph_out, x_target_out, y_target_out = morph_rv[
        "morph_chain"
    ].xyallout
    assert np.allclose(fitted["rpoly"], 1.1, atol=1e-6)
    assert np.allclose(y_morph_out, y_target_out, atol=2e-4)
    assert np.allclose(x_morph_out, x_target_out)
    assert np.allclose(x_target, x_target_out)
    assert np.allclose(y_target, y_target_out)
    # Plotting code for PR 31
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(x_morph, y_morph, color="gray", label="Morph")
    axs[0].set_xlabel("Q (1/Å)")
    axs[0].set_ylabel("I(Q)")
    axs[0].set_xlabel("Q (1/Å)")
    axs[0].legend()
    axs[1].plot(x_target, y_target, color="purple", label="Target")
    axs[1].plot(x_morph_out, y_morph_out, "--", color="gold", label="Morphed")
    axs[1].set_title(f'rpoly_init={rpoly}, rpoly_refined={fitted["rpoly"]}')
    axs[1].set_ylabel("F(Q)")
    axs[1].set_xlabel("Q (1/Å)")
    axs[1].legend()
    plt.tight_layout()
    plt.show()
