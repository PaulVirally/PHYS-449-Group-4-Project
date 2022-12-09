import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
hidden_neurons = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]#, 32, 64, 128]
accuracies = np.zeros((2, 2, 3, len(hidden_neurons)))
losses = np.zeros((2, 2, 3, len(hidden_neurons)))

# Loop over the folders
folders = os.listdir('results')
for folder in folders:
    if folder == '.DS_Store':
        continue

    if not folder.endswith('Adam'):
        continue

    folder = os.path.join('results', folder)
    files = os.listdir(folder)

    oversampled = int('oversampled' in folder) # 0 for not oversampled, 1 for oversampled
    activation = folder.split('_')[1] if oversampled else folder.split('_')[0]
    activation_idx = int(activation[-1] == 'h') # 1 for tanh, 0 for relu

    regex = re.compile('\[.*\]')
    for file in files:
        # Extract the topology
        topology_str = regex.findall(file)[0]
        topology = list(map(int, topology_str[1:-1].split(', ')))

        # Extract the number of hidden layers and neurons
        num_hidden = len(topology) - 2
        num_neurons = topology[1]
        data = np.load(os.path.join(folder, file))
        acc = data['accuracies'][-1]
        loss = data['losses'][-1]

        if num_neurons not in hidden_neurons:
            continue
        neuron_idx = hidden_neurons.index(num_neurons)
        layer_idx = num_hidden - 1
        accuracies[oversampled, activation_idx, layer_idx, neuron_idx] = acc
        losses[oversampled, activation_idx, layer_idx, neuron_idx] = loss

# Plot the results
for layer_idx in range(3):
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    for oversampled in range(2):
        for activation_idx in range(2):
            label = f'{"$tanh$" if activation_idx else "$ReLU$"}{", oversampled" if oversampled else ""}'

            acc = accuracies[oversampled, activation_idx, layer_idx, :]
            ax.plot(hidden_neurons, acc, '-o', label=label)
            ax.set_title(f'{layer_idx+1} Hidden Layer{"s" if layer_idx > 0 else ""}')
            ax.set_xlabel('Neurons per hidden layer')
            ax.set_ylabel('Accuracy [%]')
            ax.legend()

    fig.savefig(f'figures/fig7_{layer_idx+1}_layers_acc.pdf')

plt.show()