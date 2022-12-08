import os
import re
import numpy as np
import matplotlib.pyplot as plt

hidden_neurons = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 64, 128]
accuracies = np.zeros((2, 2, 3, len(hidden_neurons)))
losses = np.zeros((2, 2, 3, len(hidden_neurons)))

folders = os.listdir('results')
for folder in folders:
    if folder == '.DS_Store':
        continue

    folder = os.path.join('results', folder)
    files = os.listdir(folder)

    oversampled = int('oversampled' in folder) # 0 for not oversampled, 1 for oversampled
    activation = folder.split('_')[1] if oversampled else folder.split('_')[0]
    activation_idx = int(activation[-1] == 'h') # 0 for tanh, 1 for relu

    regex = re.compile('\[.*\]')
    for file in files:

        topology_str = regex.findall(file)[0]
        topology = list(map(int, topology_str[1:-1].split(', ')))

        num_hidden = len(topology) - 2
        num_neurons = topology[1]
        data = np.load(os.path.join(folder, file))
        acc = data['accuracies'][-1]
        loss = data['losses'][-1]

        neuron_idx = hidden_neurons.index(num_neurons)
        layer_idx = num_hidden - 1
        accuracies[oversampled, activation_idx, layer_idx, neuron_idx] = acc
        losses[oversampled, activation_idx, layer_idx, neuron_idx] = loss

fig, axs = plt.subplots(2, 3, figsize=(12, 4), tight_layout=True)
for oversampled in range(2):
    for activation_idx in range(2):
        for layer_idx in range(3):
            label = f'{"ReLU" if activation_idx else "$tanh$"}{", oversampled" if oversampled else ""}'

            ax = axs[0, layer_idx]
            acc = accuracies[oversampled, activation_idx, layer_idx, :]
            ax.plot(hidden_neurons, acc, '-o', label=label)
            ax.set_title(f'{layer_idx+1} Hidden Layer{"s" if layer_idx > 0 else ""}')
            ax.set_xlabel('Neurons per hidden layer')
            ax.set_ylabel('Accuracy [%]')

            ax = axs[1, layer_idx]
            loss = losses[oversampled, activation_idx, layer_idx, :]
            ax.plot(hidden_neurons, loss, '-o', label=label)
            ax.set_title(f'{layer_idx+1} Hidden Layer{"s" if layer_idx > 0 else ""}')
            ax.set_xlabel('Neurons per hidden layer')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')

for row in axs:
    for ax in row:
        ax.legend()
plt.show()
plt.savefig('figures/fig7.pdf')