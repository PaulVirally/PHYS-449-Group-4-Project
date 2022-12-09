# ----------------------------------------------------------------------------
#
# Phys 449 -- Fall 2021
# Tutorial 2 - Intro to ML with Pytorch
#
# ----------------------------------------------------------------------------

import json, argparse, torch, sys
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sys.path.append('src')
from nn_gen import Net
from data_gen import Data

def plot_results(obj_vals, cross_vals):
    assert len(obj_vals)==len(cross_vals), 'Length mismatch between the curves'
    num_epochs= len(obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    plt.legend()
    plt.savefig(args.res_path + '/fig.pdf')
    plt.close()


def prep_demo(param):
    # Construct a model and dataset
    model = Net(param['n_bits'], param['n_hidden'])
    data = Data(param['n_bits'],
               int(param['n_training_data']), int(param['n_test_data']))
    return model, data


def run_demo(param, model, data):

    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss= torch.nn.BCELoss(reduction= 'mean')

    obj_vals= []
    cross_vals= []
    num_epochs= int(param['num_epochs'])

    # Training loop
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(data, loss, epoch, optimizer)
        obj_vals.append(train_val)

        test_val= model.test(data, loss, epoch)
        cross_vals.append(test_val)

        if (epoch+1) % param['display_epochs'] == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(test_val))


    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    return obj_vals, cross_vals

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--param', help='parameter file name')
    parser.add_argument('--res-path', help='path of results')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    model, data = prep_demo(param['data'])
    obj_vals, cross_vals = run_demo(param['exec'], model, data)
    plot_results(obj_vals, cross_vals)