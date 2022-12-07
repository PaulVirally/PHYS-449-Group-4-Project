import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Fully connected neural network with one hidden layer
class Model3FGL(nn.Module):
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
        self.loss = nn.CrossEntropyLoss()

        # Adam optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=params.optim.learning_rate)

        # Number of epochs
        self.num_epochs = params.optim.num_epochs
        self.print_every = params.optim.print_every

    def forward(self, x):
        return self.network(x)

    def _train_epoch(self, dataloader, epoch):
        self.network.train() # Ensure the network is in training mode

        data_size = len(dataloader.dataset)
        running_loss = 0
        samples_done = 0
        num_correct = 0
        print_count = 0
        for x, y in dataloader:
            # Forward pass
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Keep track of the loss and accuracy
            running_loss += loss.item()
            num_correct += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            samples_done += len(x)
            print_count += len(x)

            # Report to the console
            if print_count >= self.print_every:
                print_count -= self.print_every
                epoch_acc = 100 * num_correct / samples_done
                epoch_loss = running_loss / samples_done
                print(f'[Epoch {epoch+1:>3d}/{self.num_epochs:>3d}] [Epoch completion {samples_done:>4d}/{data_size:>4d}] Loss: {epoch_loss:>.5e} Epoch Accuracy: {epoch_acc:>.5f}%')

    def run_training(self, train_dataloader, test_dataloader):
        accurcies = np.zeros(self.num_epochs)
        losses = np.zeros(self.num_epochs)
        for epoch in range(self.num_epochs):
            self._train_epoch(train_dataloader, epoch)
            acc, loss = self._test(test_dataloader)
            accurcies[epoch] = acc
            losses[epoch] = loss

        return accurcies, losses 

    def _test(self, dataloader):
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

        epoch_acc = 100 * num_correct / data_size
        epoch_loss = running_loss / data_size
        return epoch_acc, epoch_loss