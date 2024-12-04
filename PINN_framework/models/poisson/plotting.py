import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import rc

from . import analytic
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
            ax[i].tick_params(axis='y', labelsize=_FONTSIZE)
            # ax[i].fill_between(epochs[epochs % 10000 >= 5000], 0, facecolor='gray', alpha=.5)
    else:
        for i in range(num_plots):
            ax[i].semilogy(loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
            ax[i].tick_params(axis='x', labelsize=_FONTSIZE)
            ax[i].tick_params(axis='y', labelsize=_FONTSIZE)
        
    save_fig(fig_dir, name, extension, fig=fig, dpi=dpi)
    plt.clf()
    return

def get_plot_data(geometry_settings, forward, params, grid):
    
    xlim = geometry_settings["domain"]["rectangle"]["xlim"]
    ylim = geometry_settings["domain"]["rectangle"]["ylim"]

    X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=grid)

    # Prediction
    Z = netmap(forward)(params, plotpoints).reshape(X.shape)
    Z_true = (jnp.sin(jnp.pi*plotpoints[:, 0])*jnp.sin(jnp.pi*plotpoints[:, 1])).reshape(X.shape)

    return X, Y, Z, Z_true


def plot_results(geometry_settings, forward, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50):

    X, Y, Z, Z_true = get_plot_data(geometry_settings, forward, params, grid=grid)

    if save:
        plot_prediction(X, Y, Z, Z_true, fig_dir=fig_dir, name="Prediction", dpi=dpi)
    # if log:        
        # log_stress(X, Y, sigma_cart_list, sigma_cart_true_list, log_dir=log_dir, name="Cart_stress", varnames="XY", step=step, dpi=dpi)

    return

def plot_prediction(X, Y, Z, Z_true, *, fig_dir, name,
                   extension="png", dpi=100):
    """
    Function for plotting potential function.
    """    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("Prediction", fontsize=20)
    p = ax[0].contourf(X, Y, Z, levels=_CLEVELS)
    plt.colorbar(p, ax=ax[0])
    
    
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title("Reference", fontsize=20)
    p = ax[1].contourf(X, Y, Z_true, levels=_CLEVELS)
    plt.colorbar(p, ax=ax[1])
    
    ax[2].set_aspect('equal', adjustable='box')
    ax[2].set_title("Error", fontsize=20)
    p = ax[2].contourf(X, Y, jnp.abs(Z - Z_true), levels=_CLEVELS)
    plt.colorbar(p, ax=ax[2])

    save_fig(fig_dir, name, extension, dpi=dpi)
    plt.clf()
    return
