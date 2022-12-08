import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()

        # The network is defined by the topology
        topology = params.model.topology
        activation = params.model.activation
        self.network = nn.Sequential()
        for neurons_in, neurons_out in zip(topology[:-1], topology[1:]):
            self.network.append(nn.Linear(neurons_in, neurons_out))
            self.network.append(nn.ReLU() if activation == 'relu' else nn.Tanh())

        # Cross entropy loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        self.lbfgs = params.optim.optim == 'LBFGS'
        if self.lbfgs:
            # self.optimizer = optim.LBFGS(self.network.parameters(), lr=params.optim.learning_rate)
            self.optimizer = optim.LBFGS(self.network.parameters())
            # self.optimizer = optim.LBFGS(self.network.parameters(), history_size=10, max_iter=4)
        elif params.optim.optim == 'SGD':
            self.optimizer = optim.SGD(self.network.parameters(), lr=params.optim.learning_rate, momentum=0.9, nesterov=True)
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=params.optim.learning_rate)

        # Number of epochs
        self.num_epochs = params.optim.num_epochs
        self.print_every = params.optim.print_every

    def loss(self, y_pred, y, train=False, x=None):
        if self.lbfgs and train:
            def closure():
                if torch.is_grad_enabled():
                    self.optimizer.zero_grad()
                y_ = self.forward(x)
                loss = self.loss_fn(y_pred, y_)
                if loss.requires_grad:
                    loss.backward()
                return loss
            self.optimizer.step(closure)
            y_ = self.forward(x)
            loss = closure()
            return loss

        # Simply return the output of the loss function if we are not using LBFGS
        return self.loss_fn(y_pred, y)

    def forward(self, x):
        return self.network(x)

    def train_epoch(self, dataloader, epoch, quiet=False):
        self.network.train() # Ensure the network is in training mode

        data_size = len(dataloader.dataset)
        running_loss = 0
        samples_done = 0
        num_correct = 0
        print_count = 0
        for x, y in dataloader:
            # Forward pass
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y, train=True, x=x)

            # Backprop is done in the self.loss function if we are using LBFGS
            if not self.lbfgs:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Keep track of the loss and accuracy
            running_loss += loss.item()
            num_correct += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            samples_done += len(x)
            print_count += len(x)

            # Report to the console
            if not quiet and print_count >= self.print_every:
                print_count -= self.print_every
                epoch_acc = 100 * num_correct / samples_done
                epoch_loss = running_loss / samples_done
                print(f'[Epoch {epoch+1:>3d}/{self.num_epochs:>3d}] [Epoch completion {samples_done:>4d}/{data_size:>4d}] Loss: {epoch_loss:>.5e} Epoch Accuracy: {epoch_acc:>.5f}%')

    def test(self, dataloader):
        self.network.eval() # Ensure the network is in evaluation mode

        # Compute the accuracy and loss on the test set
        data_size = len(dataloader.dataset)
        running_loss = 0
        num_correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = self.forward(x)
                loss = self.loss(y_pred, y)
                running_loss += loss.item()
                num_correct += (y_pred.argmax(1) == y.argmax(1)).sum().item()

        accuracy = 100 * num_correct / data_size
        loss = running_loss / data_size
        return accuracy, loss