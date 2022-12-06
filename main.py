from types import SimpleNamespace
import json
from model_3fgl import Model3FGL
import torch
import torch.nn.functional as F
import numpy as np

# Import the params from the json config file into a simple namespace
with open('params/params_3fgl.json') as json_file:
    params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))

# Create the 3FGL model
model3fgl = Model3FGL(params)

# Load the data
data = np.load('data/3fgl_simple_data.npz')
in_data = data['in_data'].T
out_data = data['out_data']

# Creaste a dataloader
in_tensor = torch.from_numpy(in_data).float()
out_tensor = torch.from_numpy(out_data.astype(int)).long()
out_onehot = F.one_hot(out_tensor, num_classes=2)
dataset = torch.utils.data.TensorDataset(in_tensor, out_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.optim.batch_size, shuffle=True)

# Train the 3FGL model
model3fgl.run_training(dataloader)