import numpy as np

class Data():
    def __init__(self, n_bits, n_training_data, n_test_data):
        '''Data generation
        x_* attributes contain random bit strings of size n_bits
        y_* attributes are the parity of each bit string'''

        x_train= np.random.randint(2, size=(n_training_data, n_bits))
        y_train= np.array([np.sum(x) % 2 for x in x_train])
        x_train= np.array(x_train, dtype= np.float32)
        y_train= np.array(y_train, dtype= np.float32)

        x_test= np.random.randint(2, size=(n_test_data, n_bits))
        y_test= np.array([np.sum(x) % 2 for x in x_test])
        x_test= np.array(x_test, dtype= np.float32)
        y_test= np.array(y_test, dtype= np.float32)

        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test