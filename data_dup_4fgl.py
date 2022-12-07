from astropy.io import fits
import sys
import numpy as np
import pandas as pd
import random

def data_dup2(hdul_fgl, outfile):
    global agn_mask, pulsar_mask, classes, data
    classes = hdul_fgl[1].data['CLASS1']
    agn_classes = ['psr  ', 'agn  ', 'FSRQ ', 'fsrq ', 'BLL  ', 'bll  ', 'BCU  ', 'bcu  ', 'RDG  ', 'rdg  ', 'NLSY1', 'nlsy1', 'ssrq ', 'sey  ']
    pulsar_classes = ['PSR  ', 'psr  ']
    no_class = '     '
    agn_mask = np.isin(classes, agn_classes)
    pulsar_mask = np.isin(classes, pulsar_classes)
    noclass_mask = classes == no_class

    oversample = sum(agn_mask) - sum(pulsar_mask)

    nrows1 = hdul_fgl[1].data.shape[0]
    nrows = nrows1 + oversample
    hdu = fits.BinTableHDU.from_columns(hdul_fgl[1].columns, nrows=nrows)

    pulsars = hdul_fgl[1].data[pulsar_mask]

    for colname in hdul_fgl[1].columns.names:
        hdu.data[colname][nrows1:] = random.choice(pulsars[colname])

    hdu.writeto('temptable.fits', overwrite=True)
    hdul_v2 = fits.open('temptable.fits')

    classes = hdul_v2[1].data['CLASS1']
    agn_mask = np.isin(classes, agn_classes)
    pulsar_mask = np.isin(classes, pulsar_classes)
    noclass_mask = classes == no_class

    data = hdul_v2[1].data[(agn_mask | pulsar_mask)] # data is both pulsars and AGNs
    pulsars = hdul_v2[1].data[pulsar_mask]
    agns = hdul_v2[1].data[agn_mask]
    
    glat = data['GLAT']
    glon = data['GLON']
    ln_pivot_energy = np.log(data['Pivot_Energy'])
    unc_lp_index = data['Unc_LP_Index']
    lp_index = data['LP_Index']
    lp_beta = data['LP_beta']
    lp_sincurv = data['LP_SigCurv']
    ln_energy_flux100 = np.log(data['Energy_Flux100'])
    ln_unc_energy_flux100 = np.log(data['Unc_Energy_Flux100'])
    ln_var_index = np.log(data['Variability_Index'])
            
    in_data = np.vstack((glat, glon, ln_energy_flux100, ln_unc_energy_flux100, ln_var_index, ln_pivot_energy, lp_index, unc_lp_index, lp_beta, lp_sincurv))
    out_data = np.isin(data['CLASS1'], agn_classes).astype(float)
    np.savez_compressed(outfile, in_data=in_data, out_data=out_data)


with fits.open('data/gll_psc_v27.fit') as hdu:
    print(hdu[1].columns)
    data_dup2(hdu,'data/4fgl_simple_data.npz')
