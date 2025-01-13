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
from models.loss import maxabse


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
    else:
        for i in range(num_plots):
            ax[i].semilogy(loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
            ax[i].tick_params(axis='x', labelsize=_FONTSIZE)
            ax[i].tick_params(axis='t', labelsize=_FONTSIZE)
        
    save_fig(fig_dir, name, extension, fig=fig, dpi=dpi)
    plt.clf()
    return


def plot_results(prediction, data_true, fig_dir, type: str | None = None, save=True, dpi=50):
    if type is None:
        type = "eval"

    if save:
        plot_prediction(prediction, data_true, fig_dir=fig_dir, type=type, dpi=dpi)

    return

def plot_prediction(prediction, data_true, *, fig_dir, type,
                   extension="png", dpi=100):
    """
    Function for plotting potential function.
    """    
    num_plots = 20
    if type.lower() == "train":
        plot_names = "train_prediction"
    else:
        plot_names = "test_prediction"

    errors = jax.vmap(maxabse, in_axes=(0, 0))(prediction, data_true)
    sorted_errors_idx = jnp.argsort(errors)
    idx = np.round(np.linspace(0, len(sorted_errors_idx) - 1, num_plots)).astype(int)
    error_lim_lower = 1e-6
    error_lim_upper = jnp.max(jnp.abs(prediction - data_true))

    for plot_num, i in enumerate(sorted_errors_idx[idx]):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(prediction[i], label='Prediction')
        ax[0].plot(data_true[i], c='r', label='True')
        ax[0].legend()
        ax[1].semilogy(jnp.abs(prediction[i] - data_true[i]), label='Error')
        ax[1].set_ylim(error_lim_lower, error_lim_upper)
        save_fig(fig_dir, f"{plot_names}_{plot_num}", extension, dpi=dpi)
        plt.clf()
    return
