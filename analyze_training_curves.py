import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Define the parameters
topology = '[11, 6, 2]'
act_funcs = ['tanh', 'relu']
over_folders = ['results/oversampled_tanh_LR0p001_Adam', 'results/oversampled_relu_LR0p001_Adam']
under_folders = ['results/tanh_LR0p001_Adam', 'results/relu_LR0p001_Adam']

# Plot the accuracies
fig_over, ax_over = plt.subplots(1, 1, tight_layout=True)
for i, (over_folder, act_func) in enumerate(zip(over_folders, act_funcs)):
    data = np.load(os.path.join(over_folder, f'3fgl_{topology}.npz'))
    acc = data['accuracies']
    acc_smooth = savgol_filter(acc, 11, 3)
    epochs = np.arange(len(acc)) + 1
    ax_over.plot(epochs, acc, '-', alpha=0.25, color=f'C{i}', label=f'{act_func} Oversampled')
    ax_over.plot(epochs, acc_smooth, '--', color=f'C{i}', label=f'{act_func} Oversampled (Smoothed)')
    ax_over.set_xlabel('Epoch')
    ax_over.set_ylabel('Accuracy [%]')
    ax_over.set_ylim(70, 100)
    ax_over.legend()
fig_over.savefig('figures/[11, 6, 2]_oversampled.pdf')

# Plot the accuracies
fig_under, ax_under = plt.subplots(1, 1, tight_layout=True)
for i, (under_folder, act_func) in enumerate(zip(under_folders, act_funcs)):
    data = np.load(os.path.join(under_folder, f'3fgl_{topology}.npz'))
    acc = data['accuracies']
    acc_smooth = savgol_filter(acc, 11, 3)
    epochs = np.arange(len(acc)) + 1
    ax_under.plot(epochs, acc, '-', color=f'C{i}', alpha=0.25, label=f'{act_func}')
    ax_under.plot(epochs, acc_smooth, '--', color=f'C{i}', label=f'{act_func} (Smoothed)')
    ax_under.set_xlabel('Epoch')
    ax_under.set_ylabel('Accuracy [%]')
    ax_under.set_ylim(70, 100)
    ax_under.legend()
fig_under.savefig('figures/[11, 6, 2].pdf')

plt.show()