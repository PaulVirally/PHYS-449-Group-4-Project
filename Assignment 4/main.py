# ----------------------------------------------------------------------------
#
# Phys 449 -- Fall 2022
# Assignment 4 - Fully visible Boltzmann machine
# Ronaldo Del Young 20808600
# ----------------------------------------------------------------------------


import numpy as np

def random_spin_field_data(N, M):
    return np.random.choice([-1, 1], size=(N, M))

random_spin_field_data(20, 4)


# In[ ]:


def ising_step(field, beta=0.4):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    ising_update(field, n, m, beta)
    return field

def ising_update(field, n, m, beta):
    total = 0
    N, M = field.shape
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
            if i == n and j == m:
                continue
            total += field[i % N, j % M]
    dE = 2 * field[n, m] * total
    if dE <= 0:
        field[n, m] *= -1
    elif np.exp(-dE * beta) > np.random.rand():
        field[n, m] *= -1


# In[ ]:


data = ising_step(random_spin_field_data(20, 4))
data


# In[ ]:


transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))]

def prior(x):
    if(x[1] <=0):
        return -1
    return 1

def manual_log_like_normal(x,data):
    return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))

def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        return (accept < (np.exp(x_new-x)))


def m_h(likelihood_computer,prior, transition_model, x,iterations,data,acceptance_rule):
   
    accepted = []
    rejected = []   
    for i in range(iterations):
        x_new =  transition_model(x)    
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data) 
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
                
    return np.array(accepted), np.array(rejected)


# In[ ]:


print(np.array(accepted), np.array(rejected))


# In[ ]:


num_epochs = 10

obj_vals= []
cross_vals= []

for epoch in range(1, num_epochs + 1):
    train_val= m_h(data=data, likelihood_computer=manual_log_like_normal, prior=prior, transition_model=transition_model, x=1, acceptance_rule=acceptance, iterations=100)
    obj_vals.append(train_val)
    
    test_val= m_h(data=data, likelihood_computer=manual_log_like_normal, prior=prior, transition_model=transition_model, x=1, acceptance_rule=acceptance, iterations=100)
    cross_vals.append(test_val)

    if (epoch+1) % (display_epochs) == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+                      '\tTraining Loss: {:.4f}'.format(train_val)+                      '\tTest Loss: {:.4f}'.format(test_val))


print('Final training loss: {:.4f}'.format(obj_vals[-1]))
print('Final test loss: {:.4f}'.format(cross_vals[-1]))   


# In[ ]:
if __name__ == '__main__':
    print('hello')



