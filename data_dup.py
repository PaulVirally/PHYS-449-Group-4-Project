import numpy as np
import argparse, random, os
from astropy.io import fits

def data_dup(hdul_fgl, outfile):
    global agn_mask, pulsar_mask, classes, data
    classes = hdul_fgl[1].data['CLASS1']
    agn_classes = ['psr  ', 'agn  ', 'FSRQ ', 'fsrq ', 'BLL  ', 'bll  ', 'BCU  ', 'bcu  ', 'RDG  ', 'rdg  ', 'NLSY1', 'nlsy1', 'ssrq ', 'sey  ']
    pulsar_classes = ['PSR  ', 'psr  ']
    no_class = '     '
    agn_mask = np.isin(classes, agn_classes)
    pulsar_mask = np.isin(classes, pulsar_classes)
    noclass_mask = classes == no_class
    bad_data_mask = hdul_fgl[1].data['Signif_Curve'] == 0.0

    oversample = sum(agn_mask) - sum(pulsar_mask)

    nrows1 = hdul_fgl[1].data.shape[0]
    nrows = nrows1 + oversample
    hdu = fits.BinTableHDU.from_columns(hdul_fgl[1].columns, nrows=nrows)

    pulsars = hdul_fgl[1].data[pulsar_mask & ~bad_data_mask]

    for colname in hdul_fgl[1].columns.names:
        hdu.data[colname][nrows1:] = random.choice(pulsars[colname])

    hdu.writeto('temptable.fits', overwrite=True)
    hdul_v2 = fits.open('temptable.fits')

    classes = hdul_v2[1].data['CLASS1']
    agn_mask = np.isin(classes, agn_classes)
    pulsar_mask = np.isin(classes, pulsar_classes)
    noclass_mask = classes == no_class
    bad_data_mask = hdul_v2[1].data['Signif_Curve'] == 0.0

    data = hdul_v2[1].data[(agn_mask | pulsar_mask) & ~bad_data_mask] # data is both pulsars and AGNs
    pulsars = hdul_v2[1].data[pulsar_mask & ~bad_data_mask]
    agns = hdul_v2[1].data[agn_mask & ~bad_data_mask]
    
    # The easy ones
    glat = data['GLAT']
    glon = data['GLON']
    ln_energy_flux100 = np.log(data['Energy_Flux100'])
    ln_unc_energy_flux100 = np.log(data['Unc_Energy_Flux100'])
    ln_signif_curve = np.log(data['Signif_Curve'])
    ln_var_index = np.log(data['Variability_Index'])

    # Hardness ratios
    ef1 = data['Flux100_300']
    ef2 = data['Flux300_1000']
    ef3 = data['Flux1000_3000']
    ef4 = data['Flux3000_10000']
    ef5 = data['Flux10000_100000']
    hr12 = (ef2 - ef1) / (ef2 + ef1)
    hr23 = (ef3 - ef2) / (ef3 + ef2)
    hr34 = (ef4 - ef3) / (ef4 + ef3)
    hr45 = (ef5 - ef4) / (ef5 + ef4)

    # 500 MeV index
    alpha = data['Spectral_Index']
    beta = data['beta']
    gamma = data['Spectral_Index']
    b = data['Exp_Index']
    E_c = data['Cutoff'] # In MeV
    E_0 = data['Pivot_Energy'] # In MeV
    mev_500_index = np.zeros(data.shape)
    for i, point in enumerate(data):
        if point['SpectrumType'] in ['PowerLaw', 'PLExpCutoff', 'PLSuperExpCutoff']:
            if b[i] == float('-inf'):
                b[i] = 1
            mev_500_index[i] = gamma[i] + b[i] * (500 / E_c[i])**b[i]
        else:
            mev_500_index[i] = alpha[i] + 2*beta[i] * np.log(500 / E_0[i])
            
    in_data = np.vstack((glat, glon, ln_energy_flux100, ln_unc_energy_flux100, ln_signif_curve, ln_var_index, hr12, hr23, hr34, hr45, mev_500_index))
    out_data = np.isin(data['CLASS1'], agn_classes).astype(float)
    np.savez_compressed(outfile, in_data=in_data, out_data=out_data)

    

if __name__ == '__main__':
    #python data_dup.py --data data/gll_psc_v16.fit --outfile data/over_3fgl
    parser = argparse.ArgumentParser(description='Duplicates pulsars in dataset to even out number of AGNs and pulsars for oversampling')
    parser.add_argument('--data', help='A file/path containing input data')
    parser.add_argument('--outfile', help='Desired name of output file (include file path if needed, ex. data/over_3fgl)')
    
    args = parser.parse_args()
    
    hdul_fgl = fits.open(args.data)
    
    data_dup(hdul_fgl, args.outfile)
    
    print("Oversampling complete")
