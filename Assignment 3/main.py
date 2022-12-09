# Name: Ronaldo Del Young

if __name__ == '__main__':
    print('hello')
    
# In[1]:


import json, argparse, torch, sys
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid") # plotting style
np.random.seed(seed=1) # seed for reproducibility


# In[2]:


# Hyperparameters

nb_train = 1000  # Number of training samples
nb_test = 200 # Number of testing samples
sequence_len = 8  # Length of the binary sequence
nb_hidden = 1
nb_inputs = 2
lr = 0.01
batch_size = 50
nb_epochs = 10
display_epochs = 1

# Create dataset

def dataset(nb_samples, sequence_len):
    max_int = 2**(sequence_len-1) # Maximum integer that can be added
     # Transform integer in binary format
    format_str = '{:0' + str(sequence_len) + 'b}'
    nb_inputs = 2  # Add 2 binary numbers
    nb_outputs = 1  # Result is 1 binary number
    
    X = np.zeros((nb_samples, sequence_len, nb_inputs))
    T = np.zeros((nb_samples, sequence_len, nb_outputs)) # Target samples
    # Fill up the input and target matrix
    for i in range(nb_samples):
        # Generation of random numbers to add
        nb1 = np.random.randint(0, max_int)
        nb2 = np.random.randint(0, max_int)
        #  Sequence is reversed because the binary numbers are added from right to left and our RNN reads from left to right
        X[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1)]))
        X[i,:,1] = list(
            reversed([int(b) for b in format_str.format(nb2)]))
        T[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1+nb2)]))
    return X, T

# Create training samples
X_train, T_train = dataset(nb_train, sequence_len)
print(f'X_train tensor shape: {X_train.shape}')
print(f'T_train tensor shape: {T_train.shape}')


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, nb_inputs, nb_hidden):
        super(RNN, self).__init__()
        self.nb_hidden = nb_hidden
        self.nb_inputs = nb_inputs
        self.rnn = nn.RNN(nb_inputs, nb_hidden, batch_first=True)
        self.fc = nn.Linear(nb_hidden*sequence_len, 2)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return (out)


# In[4]:


model = RNN(nb_inputs, nb_hidden).to(device)


# In[5]:


optimizer = optim.SGD(model.parameters(), lr)


# In[6]:


inputs = torch.from_numpy(X_train)
targets = torch.from_numpy(T_train)


# In[7]:


inputs = torch.autograd.Variable(inputs.float())
inputs = torch.autograd.Variable(targets.float())


# In[8]:


loss = torch.nn.BCELoss(reduction='mean')
loss(model.forward(inputs), targets) # Got an error here And for hours I tried and was not able to debug


# In[ ]:


obj_vals= []

for epoch in range(nb_epochs):   
    obj_val = loss(model.forward(inputs), targets)
    
    optimizer.zero_grad() # clear any previous gradients
    obj_val.backward() # backprop step, calculates gradient values
    optimizer.step() # apply gradients to model parameters
    
    obj_vals.append(obj_val.item())
    if (epoch+1) % display_epochs == 0:
        print ('Epoch [{}/{}]\tLoss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))


# In[ ]:


plt.plot(range(num_epochs), obj_vals)


# In[ ]:


X_test, T_test = dataset(nb_test, sequence_len)


# In[ ]:


with torch.no_grad():
    inputs = torch.from_numpy(X_test)
    targets = torch.from_numpy(T_test)
    targets = torch.reshape(targets, (-1, 1))
    cross_val = loss(model.forward(inputs), targets)
    
print(cross_val)


# In[ ]:


obj_vals= []
cross_vals= []

model.reset() # reset your parameters

for epoch in range(nb_epochs):
    inputs = torch.from_numpy(X_train)
    targets = torch.from_numpy(T_train)
    targets = torch.reshape(targets, (-1, 1))

    obj_val = loss(model.forward(inputs), targets)
    
    optimizer.zero_grad()
    obj_val.backward()
    optimizer.step()
    obj_vals.append(obj_val.item())

    if (epoch+1) % display_epochs == 0:
        print ('Epoch [{}/{}]\t Training Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))
            
    with torch.no_grad(): 
        # don't track calculations in the following scope for the purposes of gradients
        inputs = torch.from_numpy(X_test)
        targets = torch.from_numpy(T_test)
        targets = torch.reshape(targets, (-1, 1))
        cross_val = loss(model.forward(inputs), targets)
        cross_vals.append(cross_val)

    if (epoch+1) % display_epochs == 0:
        print ('Epoch [{}/{}]\t Test Loss: {:.4f}'.format(epoch+1, num_epochs, cross_val.item()))
        
print('Final training loss: {:.4f}'.format(obj_vals[-1]))
print('Final test loss: {:.4f}'.format(cross_vals[-1]))


# In[ ]:


plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="red")
plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
plt.legend()
