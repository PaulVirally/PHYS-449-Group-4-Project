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

### train.py
```sh
python3 train.py
```
Running this file will generate the training curves (accuracy & loss vs epochs) for various topologies as they are defined in the code. The files `params/params_3fgl.json`, `params/params_4fgl.json` are used for the hyperparameters.

### model.py
This file contains the code that defines the neural net. Its is not to be run.