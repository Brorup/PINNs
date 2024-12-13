import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import rc

from models.networks import netmap
from utils.plotting import (
    save_fig,
    #log_plot,
    get_plot_variables
)


_CLEVELS = 401
_FONTSIZE = 40
_LABELSIZE = 40
_TICKLABELSIZE = 25



def plot_loss(
    loss_arr: jax.Array,
    loss_map: dict,
    *,
    fig_dir,
    name,
    epoch_step = None,
    extension="png",
    figsize = (35, 30),
    dpi = 100
) -> None:
    """
    Plots losses from array in different subplots according to the specified dict.
    """
    
    num_plots = len(loss_map.keys())
    fig, ax = plt.subplots(num_plots, 1, figsize=figsize)
    plot_split = list(loss_map.keys())

    if epoch_step is not None:
        epochs = epoch_step*np.arange(loss_arr.shape[0])
        for i in range(num_plots):
            ax[i].semilogy(epochs, loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
            ax[i].tick_params(axis='x', labelsize=_FONTSIZE)
            ax[i].tick_params(axis='t', labelsize=_FONTSIZE)
            # ax[i].fill_between(epochs[epochs % 10000 >= 5000], 0, facecolor='gray', alpha=.5)
    else:
        for i in range(num_plots):
            ax[i].semilogy(loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
            ax[i].tick_params(axis='x', labelsize=_FONTSIZE)
            ax[i].tick_params(axis='t', labelsize=_FONTSIZE)
        
    save_fig(fig_dir, name, extension, fig=fig, dpi=dpi)
    plt.clf()
    return


def plot_results(prediction, data_true, fig_dir, save=True, dpi=50):
    
    if save:
        plot_prediction(prediction, data_true, fig_dir=fig_dir, name="Prediction", dpi=dpi)

    return

def plot_prediction(prediction, data_true, *, fig_dir, name,
                   extension="png", dpi=100):
    """
    Function for plotting potential function.
    """    
    for i in range(prediction.shape[0]):
        plt.plot(prediction[i], label='Prediction')
        plt.plot(data_true[i], c='r', label='True')
        plt.legend()
        save_fig(fig_dir, f"test_data_{i}", extension, dpi=dpi)
        plt.clf()
    return
