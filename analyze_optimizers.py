from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from train import load_params, train_model, load_data
from model import Model

# Load the parameters
params = load_params('params/params_3fgl.json')
params.optim.num_epochs = 10000

params_adam = deepcopy(params)
params_adam.optim.optim = 'Adam'

params_sgd = deepcopy(params)
params_sgd.optim.optim = 'SGD'

# Load the data
dataset = load_data('data/3fgl.npz')

# Train the models
model_adam = Model(params_adam)
model_sgd = Model(params_sgd)
accuracies_adam, losses_adam = train_model(model_adam, params_adam, dataset)
accuracies_sgd, losses_sgd = train_model(model_sgd, params_sgd, dataset)

# Smooth the accuracies and losses
accuracies_adam_smooth = savgol_filter(accuracies_adam, 11, 3)
accuracies_sgd_smooth = savgol_filter(accuracies_sgd, 11, 3)
losses_adam_smooth = savgol_filter(losses_adam, 11, 3)
losses_sgd_smooth = savgol_filter(losses_sgd, 11, 3)

# Plot the accuracies and losses
fig, ax = plt.subplots(2, 1, figsize=(12, 4), tight_layout=True)
epochs = np.arange(len(accuracies_adam)) + 1
ax[0].plot(epochs, accuracies_adam, alpha=0.25, color='C0', label='Adam')
ax[0].plot(epochs, accuracies_adam_smooth, color='C0', label='Adam (smoothed)')
ax[0].plot(epochs, accuracies_sgd, alpha=0.25, color='C1', label='SGD')
ax[0].plot(epochs, accuracies_sgd_smooth, color='C1', label='Adam (smoothed)')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy [%]')
ax[0].legend()
ax[1].plot(epochs, losses_adam, alpha=0.25, color='C0', label='Adam')
ax[1].plot(epochs, losses_adam_smooth, color='C0', label='Adam (smoothed)')
ax[1].plot(epochs, losses_sgd, alpha=0.25, color='C1', label='SGD')
ax[1].plot(epochs, losses_sgd_smooth, color='C1', label='SGD (smoothed)')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_yscale('log')
ax[1].legend()
fig.savefig('figures/optimizers.pdf')
plt.show()