# Group 4 Final Project

The goal of this project is to reproduce a subset of the paper "Machine learning methods for constructing probabilistic Fermi-LAT catalogs" by A. Bhat and D. Malyshev. The paper can be found [here](https://www.aanda.org/articles/aa/full_html/2022/04/aa40766-21/aa40766-21.html).

Group Members:
* [Joseph Elliott](https://github.com/Pojoe71)
* [Abdullah Maqsood](https://github.com/AbdullahMaqsood)
* [Paul Virally](https://github.com/PaulVirally)
* [Ronaldo Del Young](https://github.com/N4ldoDelly)

## Data
*The data provided by NASA, the data we've generated*

## Figures we are reproducing

## Description of files

The following files should be run in the following order (note that the generated plots can be found in `/figures/`):

### data_extr.py

```sh
python3 data_extr.py --data3fgl data/gll_psc_v16.fit --data4fgl data/gll_psc_v27.fit --outfile3fgl data/3fgl --outfile4fgl data/4fgl
```

This file extract the data in the fits files in `data/*.fit` and saves them as `.npz` files to make it easier to use in other parts of the code.

### data_dup.py

```sh
python3 data_dup.py --data3fgl data/gll_psc_v16.fit --data4fgl data/gll_psc_v27.fit --outfile3fgl data/over_3fgl --outfile4fgl data/over_4fgl
```

This file extracts the data in the fits files in `data/*.fit` and saves them as `.npz` files to make it easier to use in other parts of the code, but it oversamples the data.

### train.py

```sh
python3 train.py
```

Running this file will generate the training curves (accuracy & loss vs epochs) for various topologies as they are defined in the code. The files `params/params_3fgl.json`, `params/params_4fgl.json` are used for the hyperparameters.

### analyze_classification.py

```sh
python3 analyze_classification.py
```

This file generates the classification domains plot.

### analyze_optimizers.py

```sh
python3 analyze_optimizers.py
```

This file generates the optimizer comparison plot that compares SGD to Adam.

### analyze_precision_recall.py

```sh
python3 analyze_precision_recall.py
```

This file generates a plot of the precesion and recall as aq function of `abs(sin(GLAT))` where `GLAT` is the galactic latitude. 

### analyze_topology.py

```sh
python3 analyze_topology.py
```

This file generates a plot of the accuracy and loss as a function of the number of layers in the neural net as well as the number of neurons per hiddnel layer.

### analyze_training_curve.py

```sh
python3 analyze_training_curve.py
```

This file generates a plot of the accuracy and loss as a function of the number of epochs for the best topology.

### model.py
This file contains the code that defines the neural net. It is not to be run.