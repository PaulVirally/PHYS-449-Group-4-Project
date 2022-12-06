import numpy as np
import argparse, random
from astropy.io import fits

def data_dup(infile):
    global agn_mask, pulsar_mask, classes, duplicated
    classes = hdul_fgl[1].data['CLASS1']
    agn_classes = ['psr  ', 'agn  ', 'FSRQ ', 'fsrq ', 'BLL  ', 'bll  ', 'BCU  ', 'bcu  ', 'RDG  ', 'rdg  ', 'NLSY1', 'nlsy1', 'ssrq ', 'sey  ']
    pulsar_classes = ['PSR  ', 'psr  ']
    no_class = '     '
    agn_mask = np.isin(classes, agn_classes)
    pulsar_mask = np.isin(classes, pulsar_classes)
    noclass_mask = classes == no_class

    oversample = sum(agn_mask) - sum(pulsar_mask)

    duplicated = []
    for i in range(classes.size):
        duplicated.append(hdul_fgl[1].data[i])

    pulsars = []
    for i in range(classes.size):
        #pulsars (classification labels in 3FGL: PSR, psr)
        if hdul_fgl[1].data[i]['CLASS1'] == 'PSR' or hdul_fgl[1].data[i]['CLASS1'] == 'psr':
            pulsars.append(hdul_fgl[1].data[i])

    for i in range(oversample):
        duplicated.append(random.choice(pulsars))

    duplicated = np.array(duplicated, dtype=object)
    

if __name__ == '__main__':
    #python data_dup.py --data data/gll_psc_v16.fit --outfile data/over_3fgl
    parser = argparse.ArgumentParser(description='Duplicates pulsars in dataset to even out number of AGNs and pulsars for oversampling')
    parser.add_argument('--data', help='A file/path containing input data')
    parser.add_argument('--outfile', help='Desired name of output file (include file path if needed, ex. data/over_3fgl)')
    
    args = parser.parse_args()
    
    hdul_fgl = fits.open(args.data)
    
    data_dup(hdul_fgl)
    np.save(args.outfile, duplicated)
    
    print(f'AGN: {sum(agn_mask)}, Pulsar: {sum(pulsar_mask)}, Difference: {sum(agn_mask) - sum(pulsar_mask)}')
    print(f'Data size: {classes.size}, Duplicated data size: {len(duplicated)}')
    print("Oversampling complete")
