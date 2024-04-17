import matplotlib.pyplot as plt
import numpy as np

def visualize_2freq(X_data, y_data, X_NN=None, y_NN=None):
    if (X_NN is not None) and (y_NN is not None):
        N_axes = 2
    else:
        N_axes = 1

    fig, axes = plt.subplots(1, N_axes)
    # Make sure axes subscriptable

    try:
        axes[0].scatter(X_data[:,-2], X_data[:,-1], c=y_data, cmap='viridis')
        #axes[0].colorbar(label = 'Infidelity')
        axes[0].set_title("Simulation Data")
    except TypeError:
        axes.scatter(X_data[:,-2], X_data[:,-1], c=y_data, cmap='viridis')
        #axes[0].colorbar(label = 'Infidelity')
        axes.set_title("Simulation Data")


    if (X_NN is not None) and (y_NN is not None):
        axes[1].scatter(X_NN[:,-2], X_NN[:,-1], c=y_NN, cmap='viridis')
        #axes[1].colorbar(label = 'Infidelity')
        axes[1].set_title("NN Predictions")

    return fig


