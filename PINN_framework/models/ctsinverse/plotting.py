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


def plot_results(prediction, data_true, labels, fig_dir, name: str = "", varying_params=None, save=True, dpi=50):
    
    if save:
        plot_prediction(prediction, data_true, labels, varying_params=varying_params, fig_dir=fig_dir, name=name, dpi=dpi)

    return

def plot_prediction(prediction, data_true, labels, *, fig_dir, name, varying_params=None,
                   extension="png", dpi=100):
    """
    Function for plotting potential function.
    """    
    if varying_params is None:
        varying_params = [True for i in range(9)]

    fig, axes = plt.subplots(3, 3, figsize=(20, 20))

    for i, ax in enumerate(fig.axes):
        if not varying_params[i]:
            continue
        ax.plot(prediction[:, i] - data_true[:, i], label=f"{labels[i]} error")
        ax.legend()
        ax.set_ylim((-1, 1))

    save_fig(fig_dir, name, extension, dpi=dpi)
    plt.clf()

    return
