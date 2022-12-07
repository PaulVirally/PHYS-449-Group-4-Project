import os
import re
import numpy as np
import matplotlib.pyplot as plt

hidden_neurons = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 64, 128]

folder = 'results/oversampled_tanh_LR0p001_Adam'
files = os.listdir(folder)
losses = np.zeros((3, len(files)))
accuracies = np.zeros((3, len(hidden_neurons)))

for file in files:
    regex = re.compile('\[.*\]')
    topology_str = regex.findall(file)[0]
    topology = list(map(int, topology_str[1:-1].split(', ')))

    num_hidden = len(topology) - 2
    num_neurons = topology[1]
    data = np.load(os.path.join(folder, file))
    acc = data['accuracies'][-1]
    loss = data['losses'][-1]

    neuron_idx = hidden_neurons.index(num_neurons)
    layer_idx = num_hidden - 1
    accuracies[layer_idx, neuron_idx] = acc
    losses[layer_idx, neuron_idx] = loss

fig, axs = plt.subplots(2, 3, figsize=(12, 4), tight_layout=True)
for i in range(3):
    ax = axs[0, i]
    ax.plot(hidden_neurons, accuracies[i])
    ax.set_title(f'{i+1} Hidden Layer(s)')
    ax.set_xlabel('Neurons per hidden layer')
    ax.set_ylabel('Accuracy [%]')

    ax = axs[1, i]
    ax.plot(hidden_neurons, losses[i])
    ax.set_title(f'{i+1} Hidden Layer(s)')
    ax.set_xlabel('Neurons per hidden layer')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')

plt.show()