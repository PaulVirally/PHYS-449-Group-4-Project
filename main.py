from types import SimpleNamespace
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from model import Model

def load_params(fpath):
    '''Load the parameters from a json file'''
    # Import the params from the json config file into a simple namespace
    with open(fpath) as json_file:
        params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    return params

def load_data(fpath):
    '''Load the data from an npz file'''
    # Load the data
    data = np.load(fpath)
    in_data = data['in_data']
    out_data = data['out_data']

    # Creaste a dataset
    in_tensor = torch.from_numpy(in_data).float()
    out_tensor = torch.from_numpy(out_data).long()
    out_onehot = F.one_hot(out_tensor, num_classes=2).float()
    dataset = torch.utils.data.TensorDataset(in_tensor, out_onehot)
    return dataset

def split_data(params, dataset, train_ratio=0.7):
    '''Split the data into train and test sets'''
    # Split the data into train and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_batch_size = train_size if params.optim.optim == 'LBFGS' else params.optim.batch_size
    test_batch_size = test_size if params.optim.optim == 'LBFGS' else params.optim.batch_size
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def train_model(model, params, dataset, quiet=False):
    '''Train the model and return the test accuracies and losses'''
    num_epochs = params.optim.num_epochs
    accuracies = np.zeros(num_epochs)
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        # The paper re-splits the train and test sets at each epoch
        train_dataloader, test_dataloader = split_data(params, dataset, train_ratio=0.7) # TODO: Should train_ratio be in the the json file?

        # Train one epoch
        model.train_epoch(train_dataloader, epoch, quiet)
        
        # Test the model and save the accuracy and loss
        acc, loss = model.test(test_dataloader)
        accuracies[epoch] = acc
        losses[epoch] = loss

    return accuracies, losses

def vary_topology(params, dataset, oversampled, fgl):
    '''Vary the topology of the model and save the results'''
    num_input = params.model.topology[0] 
    num_output = params.model.topology[-1]
    num_hidden_layers = 3 # TODO: Should this be a parameter to the function?
    num_neurons = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 64, 128]
    params.optim.num_epochs = 1000 # For consistency with the paper

    lr = str(params.optim.learning_rate).replace('.', 'p')
    results_dir = f'results/{"oversampled_" if oversampled else ""}{params.model.activation}_LR{lr}_{params.optim.optim}'
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = f'figures/{"oversampled_" if oversampled else ""}{params.model.activation}_LR{lr}_{params.optim.optim}'
    os.makedirs(fig_dir, exist_ok=True)

    for hidden_layers in range(1, num_hidden_layers+1):
        for neurons in num_neurons:
            # Create the topology
            topology = [num_input] + [neurons] * hidden_layers + [num_output]
            params.model.topology = topology

            # Create the model
            model = Model(params)

            # Train the model
            accuracies, losses = train_model(model, params, dataset, quiet=True)
            print(f'Topology: {topology}, Accuracy: {accuracies[-1]:.5f}, Loss: {losses[-1]:.5e}')

            # Save the results
            np.savez_compressed(f'{results_dir}/{fgl}fgl_{topology}.npz', accuracies=accuracies, losses=losses, params=params)

            # Plot the results and save the figure
            epochs = np.arange(len(accuracies)) + 1
            acc_smooth = savgol_filter(accuracies, 11, 3)
            loss_smooth = savgol_filter(losses, 11, 3)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
            ax[0].plot(epochs, accuracies, label='Raw', alpha=0.25)
            ax[0].plot(epochs, acc_smooth, '--', color='C0', label='Smoothed')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Accuracy [%]')
            ax[0].legend()
            ax[1].plot(epochs, losses, label='Raw', alpha=0.25)
            ax[1].plot(epochs, loss_smooth, '--', color='C0', label='Smoothed')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Loss')
            ax[1].set_yscale('log')
            ax[1].legend()
            fig.savefig(f'{fig_dir}/{fgl}fgl_{topology}.pdf')
            plt.close()

def vary_everything(params, dataset, oversampled, fgl):
    '''Vary the topology and activation functions model and save the results'''
    # ReLU
    params.model.activation = 'relu'
    vary_topology(params, dataset, oversampled, fgl)

    # tanh
    params.optim.activation = 'tanh'
    vary_topology(params, dataset, oversampled, fgl)

if __name__ == '__main__':
    # Load the data and params
    params = load_params('params/params_3fgl.json')
    # params = load_params('params/params_4fgl.json')
    dataset = load_data('data/3fgl.npz')
    # dataset = load_data('data/over_3fgl.npz')
    # dataset = load_data('data/4fgl.npz')
    # dataset = load_data('data/over_4fgl.npz')

    # vary_topology(params, dataset)
    vary_everything(params, dataset, oversampled=False, fgl=3)

    # # Create the model
    # model = Model(params)

    # # Train the model
    # accuracies, losses = train_model(model, params, dataset) # TODO: Run the model with different toplologies and log the results
    # epochs = np.arange(len(accuracies)) + 1

    # # Smooth out the accuracies and losses
    # acc_smooth = savgol_filter(accuracies, 11, 3)
    # loss_smooth = savgol_filter(losses, 11, 3)

    # # Plot the accuracy and loss
    # fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    # ax[0].plot(epochs, accuracies, label='Raw', alpha=0.25)
    # ax[0].plot(epochs, acc_smooth, '--', color='C0', label='Smoothed')
    # ax[0].set_xlabel('Epoch')
    # ax[0].set_ylabel('Accuracy [%]')
    # ax[0].legend()
    # ax[1].plot(epochs, losses, label='Raw', alpha=0.25)
    # ax[1].plot(epochs, loss_smooth, '--', color='C0', label='Smoothed')
    # ax[1].set_xlabel('Epoch')
    # ax[1].set_ylabel('Loss')
    # ax[1].set_yscale('log')
    # ax[1].legend()
    # plt.show()