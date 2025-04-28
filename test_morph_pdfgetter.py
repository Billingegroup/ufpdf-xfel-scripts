import pytest
import numpy as np
import matplotlib.pyplot as plt
from diffpy.morph.morphs.morphfuncy import MorphFuncy
from diffpy.pdfgetx.pdfconfig import PDFConfig
from diffpy.pdfgetx.pdfgetter import PDFGetter

def pdfgetter_function(x, y, cfg):
    pg1 = PDFGetter(cfg)
    _,_ = pg1(x=x, y=y)
    q, fq = pg1.fq
    return fq

@pytest.mark.parametrize("rpoly",[0.7, 0.9, 1.2])
def test_morph_pdfgetter(rpoly):
    x_morph = np.linspace(0, 22, 200)
    y_morph = (1+np.sin(2*x_morph))*np.exp(-x_morph/2)
    x_target = x_morph.copy()
    y_target = y_morph.copy()
    cfg = PDFConfig()
    cfg.composition = 'C'
    cfg.rpoly = rpoly
    cfg.dataformat = 'QA'
    cfg.qmin = 0
    cfg.qmax = 22
    cfg.qmaxinst = 22
    parameters = {'cfg': cfg}
    x_morph_expected = x_morph
    y_morph_expected = pdfgetter_function(x_morph, y_morph, cfg)
    morph = MorphFuncy()
    morph.function = pdfgetter_function
    morph.parameters = parameters
    x_morph_out, y_morph_out, x_target_out, y_target_out = morph.morph(x_morph,y_morph,x_target,y_target)
    assert np.allclose(y_morph_out, y_morph_expected)
    assert np.allclose(x_morph_out, x_morph_expected)
    assert np.allclose(y_target, y_target_out)
    assert np.allclose(x_target, x_target_out)
    #Plotting code for PR
    plt.figure()
    plt.plot(x_morph,y_morph,color='purple', label='morph')
    plt.plot(x_morph_out,y_morph_out, color = 'gold', label='morphed')
    plt.xlabel('Q (1/A)')
    plt.ylabel('F(Q)')
    plt.legend()
    plt.title(f'rpoly={rpoly}')
    plt.show()