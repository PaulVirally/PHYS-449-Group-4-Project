from types import SimpleNamespace
import json
from model_3fgl import Model3FGL
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Import the params from the json config file into a simple namespace
with open('params/params_3fgl.json') as json_file:
    params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))

# Create the 3FGL model
model3fgl = Model3FGL(params)

# Load the data
data = np.load('data/3fgl_simple_data.npz')
in_data = data['in_data'].T
out_data = data['out_data']

# Creaste a dataset
in_tensor = torch.from_numpy(in_data).float()
out_tensor = torch.from_numpy(out_data.astype(int)).long()
out_onehot = F.one_hot(out_tensor, num_classes=2).float()
dataset = torch.utils.data.TensorDataset(in_tensor, out_onehot)

# Split the data into train and test sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.optim.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params.optim.batch_size, shuffle=True)

# Train the 3FGL model
accuracies, losses = model3fgl.run_training(train_dataloader, test_dataloader)
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