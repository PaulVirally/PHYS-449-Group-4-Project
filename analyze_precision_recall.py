import numpy as np
import matplotlib.pyplot as plt
from main import load_params, load_data, train_model
from model import Model

# Load the data and params
params_3fgl = load_params('params/params_3fgl.json')
params_4fgl = load_params('params/params_4fgl.json')

dataset_3fgl = load_data('data/3fgl.npz')
dataset_4fgl = load_data('data/4fgl.npz')
dataset_3fgl_o = load_data('data/over_3fgl.npz')
dataset_4fgl_o = load_data('data/over_4fgl.npz')

# Create the models
model_3fgl = Model(params_3fgl)
model_4fgl = Model(params_4fgl)
model_3fgl_o = Model(params_3fgl)
model_4fgl_o = Model(params_4fgl)

# Train the models
accuracies_3fgl, losses_3fgl = train_model(model_3fgl, params_3fgl, dataset_3fgl, quiet=True)
accuracies_4fgl, losses_4fgl = train_model(model_4fgl, params_4fgl, dataset_4fgl, quiet=True)
accuracies_3fgl_o, losses_3fgl_o = train_model(model_3fgl_o, params_3fgl, dataset_3fgl_o, quiet=True)
accuracies_4fgl_o, losses_4fgl_o = train_model(model_4fgl_o, params_4fgl, dataset_4fgl_o, quiet=True)

# Which entries are predicted to be pulsars
pulsar_pred_3fgl = model_3fgl(dataset_3fgl.tensors[0]).detach().argmax(dim=1).numpy() == 1
pulsar_pred_4fgl = model_4fgl(dataset_4fgl.tensors[0]).detach().argmax(dim=1).numpy() == 1
pulsar_pred_3fgl_o = model_3fgl_o(dataset_3fgl_o.tensors[0]).detach().argmax(dim=1).numpy() == 1
pulsar_pred_4fgl_o = model_4fgl_o(dataset_4fgl_o.tensors[0]).detach().argmax(dim=1).numpy() == 1

# Which entries are actually pulsars
pulsar_true_3fgl = dataset_3fgl.tensors[1].argmax(dim=1).numpy() == 1
pulsar_true_4fgl = dataset_4fgl.tensors[1].argmax(dim=1).numpy() == 1
pulsar_true_3fgl_o = dataset_3fgl_o.tensors[1].argmax(dim=1).numpy() == 1
pulsar_true_4fgl_o = dataset_4fgl_o.tensors[1].argmax(dim=1).numpy() == 1

# Precision and recall
precision_3fgl = []
precision_4fgl = []
precision_3fgl_o = []
precision_4fgl_o = []
recall_3fgl = []
recall_4fgl = []
recall_3fgl_o = []
recall_4fgl_o = []

# glat
glat_3fgl = dataset_3fgl.tensors[0].numpy()[:, 0]
glat_4fgl = dataset_4fgl.tensors[0].numpy()[:, 0]
glat_3fgl_o = dataset_3fgl_o.tensors[0].numpy()[:, 0]
glat_4fgl_o = dataset_4fgl_o.tensors[0].numpy()[:, 0]

glat_edges = np.linspace(0, 1, num=8)
for left, right in zip(glat_edges[:-1], glat_edges[1:]):
    # Which entries are in the current glat bin
    curr_glat_3fgl_mask = (np.abs(np.sin(glat_3fgl)) >= left) & (np.abs(np.sin(glat_3fgl)) < right)
    curr_glat_4fgl_mask = (np.abs(np.sin(glat_4fgl)) >= left) & (np.abs(np.sin(glat_4fgl)) < right)
    curr_glat_3fgl_o_mask = (np.abs(np.sin(glat_3fgl_o)) >= left) & (np.abs(np.sin(glat_3fgl_o)) < right)
    curr_glat_4fgl_o_mask = (np.abs(np.sin(glat_4fgl_o)) >= left) & (np.abs(np.sin(glat_4fgl_o)) < right)

    # Which entries are predicted to be pulsars
    curr_pulsar_pred_3fgl = pulsar_pred_3fgl[curr_glat_3fgl_mask]
    curr_pulsar_pred_4fgl = pulsar_pred_4fgl[curr_glat_4fgl_mask]
    curr_pulsar_pred_3fgl_o = pulsar_pred_3fgl_o[curr_glat_3fgl_o_mask]
    curr_pulsar_pred_4fgl_o = pulsar_pred_4fgl_o[curr_glat_4fgl_o_mask]

    # Which entries are actually pulsars
    curr_pulsar_true_3fgl = pulsar_true_3fgl[curr_glat_3fgl_mask]
    curr_pulsar_true_4fgl = pulsar_true_4fgl[curr_glat_4fgl_mask]
    curr_pulsar_true_3fgl_o = pulsar_true_3fgl_o[curr_glat_3fgl_o_mask]
    curr_pulsar_true_4fgl_o = pulsar_true_4fgl_o[curr_glat_4fgl_o_mask]

    # True positives
    curr_tp_3fgl = np.sum(curr_pulsar_pred_3fgl & curr_pulsar_true_3fgl)
    curr_tp_4fgl = np.sum(curr_pulsar_pred_4fgl & curr_pulsar_true_4fgl)
    curr_tp_3fgl_o = np.sum(curr_pulsar_pred_3fgl_o & curr_pulsar_true_3fgl_o)
    curr_tp_4fgl_o = np.sum(curr_pulsar_pred_4fgl_o & curr_pulsar_true_4fgl_o)

    # True
    curr_t_3fgl = np.sum(curr_pulsar_true_3fgl)
    curr_t_4fgl = np.sum(curr_pulsar_true_4fgl)
    curr_t_3fgl_o = np.sum(curr_pulsar_true_3fgl_o) 
    curr_t_4fgl_o = np.sum(curr_pulsar_true_4fgl_o)

    # Positive
    curr_p_3fgl = np.sum(curr_pulsar_pred_3fgl)
    curr_p_4fgl = np.sum(curr_pulsar_pred_4fgl)
    curr_p_3fgl_o = np.sum(curr_pulsar_pred_3fgl_o)
    curr_p_4fgl_o = np.sum(curr_pulsar_pred_4fgl_o)

    # Precision
    curr_precision_3fgl = curr_tp_3fgl / curr_p_3fgl
    curr_precision_4fgl = curr_tp_4fgl / curr_p_4fgl
    curr_precision_3fgl_o = curr_tp_3fgl_o / curr_p_3fgl_o
    curr_precision_4fgl_o = curr_tp_4fgl_o / curr_p_4fgl_o

    # Recall
    curr_recall_3fgl = curr_tp_3fgl / curr_t_3fgl
    curr_recall_4fgl = curr_tp_4fgl / curr_t_4fgl
    curr_recall_3fgl_o = curr_tp_3fgl_o / curr_t_3fgl_o
    curr_recall_4fgl_o = curr_tp_4fgl_o / curr_t_4fgl_o

    # Append to lists
    precision_3fgl.append(curr_precision_3fgl)
    precision_4fgl.append(curr_precision_4fgl)
    precision_3fgl_o.append(curr_precision_3fgl_o)
    precision_4fgl_o.append(curr_precision_4fgl_o)
    recall_3fgl.append(curr_recall_3fgl)
    recall_4fgl.append(curr_recall_4fgl)
    recall_3fgl_o.append(curr_recall_3fgl_o)
    recall_4fgl_o.append(curr_recall_4fgl_o)

# Plot the precision and recall against the glat
fig, ax = plt.subplots()
ax.set_xlabel('abs(sin(GLAT))')
ax.set_ylabel('Precision (Solid) or Recall (Dashed)')
ax.plot(glat_edges[1:], precision_3fgl, '-o', label='3FGL Precision', color='C0')
ax.plot(glat_edges[1:], precision_4fgl, '-d', label='4FGL Precision', color='C0')
ax.plot(glat_edges[1:], precision_3fgl_o, '-^', label='3FGL Precision Oversampled', color='C0')
ax.plot(glat_edges[1:], precision_4fgl_o, '-s', label='4FGL Precision Oversampled', color='C0')
ax.plot(glat_edges[1:], recall_3fgl, '--o', label='3FGL Recall', color='C1')
ax.plot(glat_edges[1:], recall_4fgl, '--d', label='4FGL Recall', color='C1')
ax.plot(glat_edges[1:], recall_3fgl_o, '--^', label='4FGL Recall Oversampled', color='C1')
ax.plot(glat_edges[1:], recall_4fgl_o, '--s', label='4FGL Recall Oversampled', color='C1')
ax.legend()
ax.set_ylim(0.3, 1.5)
fig.tight_layout()
fig.savefig('figures/fig12.pdf')
plt.show()