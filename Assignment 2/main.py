# Phys 449 -- Fall 2022
# Assignment 2 - Pytorch

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[19]:


learning_rate = 1e-2
num_epochs = 15
batch_size = 500
n_hidden = 512 
n_bits = 14*14


# In[6]:


data = np.genfromtxt('inputs/even_mnist.csv', delimiter=','


# In[20]:


train= data[:-3000]
x_train= train[:batch_size, :-1]
y_train= train[:batch_size, -1]
x_train= np.array(x_train, dtype= np.float32)
y_train= np.array(y_train, dtype= np.float32)


# In[ ]:


class NN(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and sigmoid.
    '''

    def __init__(self, n_bits, n_hidden):
        super(NN, self).__init__()
        self.fc1= nn.Linear(n_bits, n_hidden)
        self.fc2= nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y
    
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


# In[ ]:


model =  model = NN(param['n_bits'], param['n_hidden']).to(torch.device("cpu"))


# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
inputs = torch.from_numpy(x_train)


# In[ ]:


targets = torch.from_numpy(y_train) 
print(targets.size()) 
targets = torch.reshape(targets, (-1, 1))
print(targets.size()) 


# In[ ]:


loss = torch.nn.BCELoss(reduction='mean')
loss(model.forward(inputs), targets)


# In[ ]:


display_epochs = 1


# In[ ]:


obj_vals= []

for epoch in range(num_epochs):   
    obj_val = loss(model.forward(inputs), targets)
    
    optimizer.zero_grad() # clear any previous gradients
    obj_val.backward() # backprop step, calculates gradient values
    optimizer.step() # apply gradients to model parameters
    
    obj_vals.append(obj_val.item())
    if (epoch+1) % display_epochs == 0:
        print ('Epoch [{}/{}]\tLoss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# In[ ]:


plt.plot(range(num_epochs), obj_vals)


# In[ ]:


x_test= data[-3000:, :-1]
y_test= data[-3000:, :-1]
x_test= np.array(x_test, dtype= np.float32)
y_test= np.array(y_test, dtype= np.float32)

with torch.no_grad():
    inputs = torch.from_numpy(x_test)
    targets = torch.from_numpy(y_test)
    targets = torch.reshape(targets, (-1, 1))
    cross_val = loss(model.forward(inputs), targets)
    
print(cross_val)


# In[ ]:


obj_vals= []
cross_vals= []

model.reset() # reset your parameters

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    targets = torch.reshape(targets, (-1, 1))

    obj_val = loss(model.forward(inputs), targets)
    
    optimizer.zero_grad()
    obj_val.backward()
    optimizer.step()
    obj_vals.append(obj_val.item())

    if (epoch+1) % display_epochs == 0:
        print ('Epoch [{}/{}]\t Training Loss: {:.4f}'.format(epoch+1, num_epochs, obj_val.item()))
            
    with torch.no_grad(): 
        inputs = torch.from_numpy(x_test)
        targets = torch.from_numpy(y_test)
        targets = torch.reshape(targets, (-1, 1))
        cross_val = loss(model.forward(inputs), targets)
        cross_vals.append(cross_val)

    if (epoch+1) % display_epochs == 0:
        print ('Epoch [{}/{}]\t Test Loss: {:.4f}'.format(epoch+1, num_epochs, cross_val.item()))
        
print('Final training loss: {:.4f}'.format(obj_vals[-1]))
print('Final test loss: {:.4f}'.format(cross_vals[-1]))


# In[ ]:


plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="red")
plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "blue")
plt.legend()


# In[ ]:


if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='ML Assignment 2')
    parser.add_argument('--param', help='parameter file name')
    parser.add_argument('--res-path', help='path of results')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    model, data = prep_demo(param['data_hp'])
    obj_vals, cross_vals = run_demo(param['exec'], model, data)
    plot_results(obj_vals, cross_vals)

