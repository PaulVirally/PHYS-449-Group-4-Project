# Write your assignment here

if __name__ == '__main__':
    print('hello')
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing inputs
data_1 = pd.read_csv('data/1.in')

#Define variables
x = (x1, x2)
x1 = data_1.iloc[:,0]
x2 = data_1.iloc[:,1]
y = data_1.iloc[:,2]

plt.scatter(x,y)
plt.show()

# the model
# the gradient of a line can be found from the equation of a stright line y = mx+b

m = 0
b = 0

h_p = pd.read_csv('data/1.json')

n = len(x)

for i in range(n):
    y_pred = m*x + b
    w_m = (-2/n) * sum(x * (y- y_pred))
    w_c = (-2/n) * sum(y - y_pred)
    m = m-((h_p.int(learning rate))* w_m)
    c = c-((h_p.int(learning rate))* w_c)
    
print (m, c)

#Importing inputs for data set 2
data_2 = pd.read_csv('data/2.in')

#Define variables
x = (x1, x2)
x1 = data_2.iloc[:,0]
x2 = data_2.iloc[:,1]
y = data_2.iloc[:,2]

plt.scatter(x,y)
plt.show()

m = 0
b = 0

h_p = pd.read_csv('data/2.json')

n = len(x)

for i in range(n):
    y_pred = m*x + b
    w_m = (-2/n) * sum(x * (y- y_pred))
    w_c = (-2/n) * sum(y - y_pred)
    m = m-((h_p.int(learning rate))* w_m)
    c = c-((h_p.int(learning rate))* w_c)
    
print (m, c)
$ git add --all