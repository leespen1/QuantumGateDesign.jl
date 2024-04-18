import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def visualize_2freq(X_data, y_data, X_NN, y_NN):

    fig, axes = plt.subplots(1, 2)

    min_y = min(min(y_data), min(y_NN))
    max_y = max(max(y_data), max(y_NN))
    norm = Normalize(vmin=min_y, vmax=max_y)
    # Make sure axes subscriptable

    axes[0].scatter(X_data[:,-2], X_data[:,-1], c=y_data, cmap='viridis', norm=norm)
    axes[0].set_title("Simulation Data")
    axes[0].set_xlabel("Frequency 1")
    axes[0].set_ylabel("Frequency 2")
    #axes[0].colorbar(label = 'Infidelity')


    scatter_ret = axes[1].scatter(X_NN[:,-2], X_NN[:,-1], c=y_NN, cmap='viridis', norm=norm)
    axes[1].set_title("NN Predictions")
    axes[1].set_xlabel("Frequency 1")
    axes[1].set_ylabel("Frequency 2")
    cbar = fig.colorbar(scatter_ret, ax=axes[1], location='right', pad=0.1)
    cbar.set_label('Infidelity')

    return fig


