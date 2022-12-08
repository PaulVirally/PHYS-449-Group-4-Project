from main import load_params, train_model
from model import Model
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

param_path = 'params/params_fig9.json'
data_path = 'data/over_3fgl.npz'

## Train the model
# Load the params
params = load_params(param_path)

# Load the data
data = np.load(data_path)
in_data = data['in_data']
out_data = data['out_data']

# We only want two features from the whole dataset
features = ['glat', 'glon', 'ln_energy_flux100', 'ln_unc_energy_flux100', 'ln_signif_curve', 'ln_var_index', 'hr12', 'hr23', 'hr34', 'hr45', 'mev_500_index']
ln_signif_curve_idx = features.index('ln_signif_curve')
ln_var_index_idx = features.index('ln_var_index')
in_data = in_data[:, [ln_signif_curve_idx, ln_var_index_idx]]

# Creaste a dataset
in_tensor = torch.from_numpy(in_data).float()
out_tensor = torch.from_numpy(out_data).long()
out_onehot = F.one_hot(out_tensor, num_classes=2).float()
dataset = torch.utils.data.TensorDataset(in_tensor, out_onehot)

model = Model(params)
acc, loss = train_model(model, params, dataset)
final_acc = acc[-1]

## Plot the classification domains
ln_signif_curve = in_data[:, 0]
ln_var_index = in_data[:, 1]

xs = np.linspace(np.min(ln_var_index), np.max(ln_var_index), 10)
ys = np.linspace(np.min(ln_signif_curve), np.max(ln_signif_curve), 10)
X, Y = np.meshgrid(xs, ys)

X_flat = np.reshape(X, -1)
Y_flat = np.reshape(Y, -1)
nn_input = torch.from_numpy(np.stack((Y_flat, X_flat), axis=1)).float()
nn_output = torch.sigmoid(model(nn_input).detach()).numpy()
sgn = 2 * np.argmax(nn_output, axis=1) - 1
nn_output = ((np.max(nn_output, axis=1) * sgn) + 1) / 2
nn_output = np.reshape(nn_output, X.shape)

contour = plt.contourf(X, Y, nn_output, levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cmap='GnBu')
plt.scatter(ln_var_index, ln_signif_curve, s=1, c=out_data, cmap='GnBu')
plt.plot(ln_var_index[[out_data == 1][0]], ln_signif_curve[[out_data == 1][0]], '.', color='#183755', label='AGN')
plt.plot(ln_var_index[[out_data == 0][0]], ln_signif_curve[[out_data == 0][0]], '.', color='#5ea340', label='Pulsar')
plt.colorbar(contour)
plt.legend()
plt.xlabel('ln(Variability Index)')
plt.ylabel('ln(Significance Curvature)')
plt.savefig('figures/fig9.pdf')
plt.show()