from types import SimpleNamespace
import json
from model_3fgl import Model3FGL
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    in_data = data['in_data'].T
    out_data = data['out_data']

    # Creaste a dataset
    in_tensor = torch.from_numpy(in_data).float()
    out_tensor = torch.from_numpy(out_data.astype(int)).long()
    out_onehot = F.one_hot(out_tensor, num_classes=2).float()
    dataset = torch.utils.data.TensorDataset(in_tensor, out_onehot)
    return dataset

def split_data(dataset, train_ratio=0.7):
    '''Split the data into train and test sets'''
    # Split the data into train and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.optim.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params.optim.batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def train_model(model, params, dataset):
    '''Train the model and return the test accuracies and losses'''
    num_epochs = params.optim.num_epochs
    accuracies = np.zeros(num_epochs)
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        # The paper re-splits the train and test sets at each epoch
        train_dataloader, test_dataloader = split_data(dataset, train_ratio=0.7) # TODO: Should train_ratio in the the json file?

        # Train one epoch
        model.train_epoch(train_dataloader, epoch)
        
        # Test the model and save the accuracy and loss
        acc, loss = model.test(test_dataloader)
        accuracies[epoch] = acc
        losses[epoch] = loss

    return accuracies, losses

if __name__ == '__main__':
    # Load the data and params
    params = load_params('params/params_3fgl.json')
    dataset = load_data('data/3fgl_simple_data.npz')

    # Create the 3FGL model
    model3fgl = Model3FGL(params)

    # Train the 3FGL model
    accuracies, losses = train_model(model3fgl, params, dataset)
    epochs = np.arange(len(accuracies)) + 1

    # Plot the accuracy and loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    ax[0].plot(epochs, accuracies,)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy [%]')
    ax[1].plot(epochs, losses)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    plt.show()