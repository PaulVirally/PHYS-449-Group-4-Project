import argparse
import random
import numpy as np
from astropy.io import fits

def extract_3fgl_features(data_path):
    # Open the fits file
    hdul = fits.open(data_path)

    # Extract the AGNs and the pulsars
    classes = hdul[1].data['CLASS1']
    agn_classes = ['psr  ', 'agn  ', 'FSRQ ', 'fsrq ', 'BLL  ', 'bll  ', 'BCU  ', 'bcu  ', 'RDG  ', 'rdg  ', 'NLSY1', 'nlsy1', 'ssrq ', 'sey  ']
    pulsar_classes = ['PSR  ', 'psr  ']
    agn_mask = np.isin(classes, agn_classes)
    pulsar_mask = np.isin(classes, pulsar_classes)

    # Some columns in the 3fgl dataset have bad data
    bad_data_mask = hdul[1].data['Signif_Curve'] == 0.0
    agn_mask = agn_mask & ~bad_data_mask
    pulsar_mask = pulsar_mask & ~bad_data_mask

    # Combine the AGNs and pulsars
    data = hdul[1].data[agn_mask | pulsar_mask]

    # Extract the 11 features we need for 3fgl
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
    out_data = np.isin(data['CLASS1'], agn_classes).astype(int)
    return in_data.T, out_data

def data_dup(data_path, out_path):
    # Get the features
    in_data, out_data = extract_3fgl_features(data_path)

    # Get the indices of the AGNs and pulsars within the data
    agn_idxs = np.where(out_data == 1)[0]
    pulsar_idxs = np.where(out_data == 0)[0]

    # There are more AGNs than pulsars, so we need to oversample the pulsars
    oversample = agn_idxs.size

    # Randomly sample pulsars to get the same number as AGNs
    rng = np.random.default_rng()
    new_pulsar_idxs = rng.choice(pulsar_idxs, size=oversample)
    new_pulsars = in_data[new_pulsar_idxs]

    # Remove the old pulsars from the data
    in_data = np.delete(in_data, new_pulsar_idxs, axis=0)
    out_data = np.delete(out_data, new_pulsar_idxs, axis=0)

    # Add the new pulsars to the data
    in_data = np.vstack((in_data, new_pulsars))
    out_data = np.hstack((out_data, np.zeros(oversample)))

    # Shuffle the data
    perm = rng.permutation(in_data.shape[0])
    in_data = in_data[perm, :]
    out_data = out_data[perm]

    # Save the data
    np.savez(out_path, in_data=in_data, out_data=out_data)

if __name__ == '__main__':
    #python data_dup.py --data data/gll_psc_v16.fit --outfile data/over_3fgl
    parser = argparse.ArgumentParser(description='Duplicates pulsars in dataset to even out number of AGNs and pulsars for oversampling')
    parser.add_argument('--data', help='A file/path containing input data')
    parser.add_argument('--outfile', help='Desired name of output file (include file path if needed, ex. data/over_3fgl)')
    
    args = parser.parse_args()
    
    data_dup(args.data, args.outfile)
    print("Oversampling complete")