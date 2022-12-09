import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and sigmoid.
    '''

    def __init__(self, n_bits, n_hidden):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_bits, n_hidden)
        self.fc2= nn.Linear(n_hidden, 1)

    # Feedforward function
    def forward(self, x):
        h = func.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)

        obj_val= loss(self.forward(inputs).reshape(-1), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)

            cross_val= loss(self.forward(inputs).reshape(-1), targets)
        return cross_val.item()