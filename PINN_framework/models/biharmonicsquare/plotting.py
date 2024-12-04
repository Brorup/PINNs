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

def get_plot_data(geometry_settings, forward, hessian, params, grid, **kwargs):
    
    xlim = geometry_settings["domain"]["rectangle"]["xlim"]
    ylim = geometry_settings["domain"]["rectangle"]["ylim"]

    X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=grid)

    # Hessian prediction
    phi = netmap(forward)(params, plotpoints).reshape(X.shape)
    phi_true = (jnp.sin(plotpoints[:, 0])*jnp.sin(plotpoints[:, 1])).reshape(X.shape)
    
    phi_pp = netmap(hessian)(params, plotpoints).reshape(-1, 4)

    # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
    sigma_cart = phi_pp

    # List and reshape the four components
    sigma_cart_list = [sigma_cart[:, i].reshape(X.shape) for i in range(4)]

    # Calculate true stresses (cartesian and polar)
    sigma_cart_true = jax.vmap(analytic.cart_stress_true)(plotpoints, **kwargs)
    sigma_cart_true_list = [sigma_cart_true.reshape(-1, 4)[:, i].reshape(X.shape) for i in range(4)]

    return X, Y, phi, phi_true, sigma_cart_list, sigma_cart_true_list


def plot_results(geometry_settings, forward, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, **kwargs):

    X, Y, phi, phi_true, sigma_cart_list, sigma_cart_true_list = get_plot_data(geometry_settings, forward, hessian, params, grid=grid, **kwargs)

    if save:
        plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir=fig_dir, name="Cart_stress", dpi=dpi)
        plot_potential(X, Y, phi, phi_true, fig_dir=fig_dir, name="Potential", dpi=dpi)
    if log:        
        log_stress(X, Y, sigma_cart_list, sigma_cart_true_list, log_dir=log_dir, name="Cart_stress", varnames="XY", step=step, dpi=dpi)

    return

def plot_potential(X, Y, Z, Z_true, *, fig_dir, name,
                   extension="png", dpi=100):
    """
    Function for plotting potential function.
    """    
    fig, ax = plt.subplots(1, 2, figsize=(15, 30))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("Prediction", fontsize=20)
    p = ax[0].contourf(X, Y, Z, levels=_CLEVELS)
    plt.colorbar(p, ax=ax[0])
    
    
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title("Prediction", fontsize=20)
    p = ax[1].contourf(X, Y, Z_true, levels=_CLEVELS)
    plt.colorbar(p, ax=ax[1])
    
    save_fig(fig_dir, name, extension, dpi=dpi)
    plt.clf()
    return


def plot_stress(X, Y, Z, Z_true, *, fig_dir, name,
                extension="png",
                figsize = (25, 30), dpi = 100):
    """
    Function for plotting stresses in cartesian coordinates.
    """
    rc("text", usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    hess_idx = [0, 1, 3]

    vmins = [min(jnp.min(Z_true[i]), jnp.min(Z[i])) for i in hess_idx]
    vmaxs = [max(jnp.max(Z_true[i]), jnp.max(Z[i])) for i in hess_idx]
    
    u = [
        [Z[i] for i in hess_idx],
        [Z_true[i] for i in hess_idx],
        [jnp.abs(Z[i]-Z_true[i]) for i in hess_idx]
    ]
    all_titles = [[
        r"Prediction ($\sigma_{xx}$)",
        r"Prediction ($\sigma_{xy}$)",
        r"Prediction ($\sigma_{yy}$)"
    ], [
        r"True solution ($\sigma_{xx}$)",
        r"True solution ($\sigma_{xy}$)",
        r"True solution ($\sigma_{yy}$)"
    ], [
        r"Absolute error ($\sigma_{xx}$)",
        r"Absolute error ($\sigma_{xy}$)",
        r"Absolute error ($\sigma_{yy}$)"
    ]]

    include = [0, 2]

    fig, ax = plt.subplots(len(hess_idx), len(include), figsize=figsize)
    for r, h in enumerate(hess_idx):
        for c, cc in enumerate(include):
            ax[r, c].set_aspect('equal', adjustable='box')
            ax[r, c].set_title(all_titles[cc][r], fontsize=_FONTSIZE, pad=20)
            if cc < 2:
                p = ax[r, c].contourf(X , Y, u[cc][r], levels=_CLEVELS, cmap="jet")
            else:
                p = ax[r, c].contourf(X , Y, u[cc][r], levels=_CLEVELS, cmap="jet")
            p.set_edgecolor("face")
            ax[r, c].set_xlabel(r"$x$", fontsize=_LABELSIZE)
            ax[r, c].set_ylabel(r"$y$", rotation=0, fontsize=_LABELSIZE)
            ax[r, c].set_xticks([-10., -5., 0., 5., 10.])
            ax[r, c].set_yticks([-10., -5., 0., 5., 10.])
            ax[r, c].set_xticklabels([r"$-10$", r"$-5$", r"$0$", r"$5$", r"$10$"], fontsize=_TICKLABELSIZE)
            ax[r, c].set_yticklabels([r"$-10$", r"$-5$", r"$0$", r"$5$", r"$10$"], fontsize=_TICKLABELSIZE)
            if cc < 2:
                cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(jnp.min(u[cc][r]), jnp.max(u[cc][r]), 11))
            else:
                cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(0, jnp.max(u[cc][r]), 11))
            cbar.ax.tick_params(labelsize=_TICKLABELSIZE)

    fig.tight_layout(pad=3.0)
    save_fig(fig_dir, name, extension, dpi=dpi)
    plt.clf()
    return
    
def log_stress(X, Y, Z, Z_true, *, log_dir, name, step=None, varnames="XY", dpi=50):
    pass
    # # Log plots
    # log_plot(X, Y, Z[0], name=name+"/Surrogate/"+varnames[0]+varnames[0], log_dir=log_dir, step=step,
    #         vmin=min(jnp.min(Z_true[0]),jnp.min(Z[0])), 
    #         vmax=max(jnp.max(Z_true[0]),jnp.max(Z[0])), dpi=dpi)
    
    # log_plot(X, Y, Z[1], name=name+"/Surrogate/"+varnames[0]+varnames[1], log_dir=log_dir, step=step,
    #         vmin=min(jnp.min(Z_true[1]),jnp.min(Z[1])), 
    #         vmax=max(jnp.max(Z_true[1]),jnp.max(Z[1])), dpi=dpi)
            
    # log_plot(X, Y, Z[3], name=name+"/Surrogate/"+varnames[1]+varnames[1], log_dir=log_dir, step=step,
    #         vmin=min(jnp.min(Z_true[3]),jnp.min(Z[3])), 
    #         vmax=max(jnp.max(Z_true[3]),jnp.max(Z[3])), dpi=dpi)
    
    
    # log_plot(X, Y, jnp.abs(Z_true[0] - Z[0]), name=name+"/Error/"+varnames[0]+varnames[0], log_dir=log_dir, step=step, logscale=True, dpi=dpi)
    
    # log_plot(X, Y, jnp.abs(Z_true[1] - Z[1]), name=name+"/Error/"+varnames[0]+varnames[1], log_dir=log_dir, step=step, logscale=True, dpi=dpi)

    # log_plot(X, Y, jnp.abs(Z_true[3] - Z[3]), name=name+"/Error/"+varnames[1]+varnames[1], log_dir=log_dir, step=step, logscale=True, dpi=dpi)
            
    
    #     # These are redundant after first time being logged
    # if step == 0:
    #     log_plot(X, Y, Z_true[0], name=name+"/True/"+varnames[0]+varnames[0], log_dir=log_dir, step=step,
    #             vmin=min(jnp.min(Z_true[0]),jnp.min(Z[0])), 
    #             vmax=max(jnp.max(Z_true[0]),jnp.max(Z[0])), dpi=dpi)
        
    #     log_plot(X, Y, Z_true[1], name=name+"/True/"+varnames[0]+varnames[1], log_dir=log_dir, step=step,
    #             vmin=min(jnp.min(Z_true[1]),jnp.min(Z[1])), 
    #             vmax=max(jnp.max(Z_true[1]),jnp.max(Z[1])), dpi=dpi)
                
    #     log_plot(X, Y, Z_true[3], name=name+"/True/"+varnames[1]+varnames[1], log_dir=log_dir, step=step,
    #             vmin=min(jnp.min(Z_true[3]),jnp.min(Z[3])), 
    #             vmax=max(jnp.max(Z_true[3]),jnp.max(Z[3])), dpi=dpi)
            
            
            
            
def plot_boundaries(geometry_settings, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, **kwargs):
        xlim = geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = geometry_settings["domain"]["rectangle"]["ylim"]

        b0x = jnp.linspace(xlim[0], xlim[1], grid).reshape(-1,1)
        b0y = jnp.full_like(b0x, ylim[0])
        b0 = jnp.hstack((b0x, b0y))
        
        b1y = jnp.linspace(ylim[0], ylim[1], grid).reshape(-1,1)
        b1x = jnp.full_like(b1y, xlim[1])
        b1 = jnp.hstack((b1x, b1y))
        
        b2x = jnp.linspace(xlim[1], xlim[0], grid).reshape(-1,1)
        b2y = jnp.full_like(b2x, ylim[1])
        b2 = jnp.hstack((b2x, b2y))

        b3y = jnp.linspace(ylim[1], ylim[0], grid).reshape(-1,1)
        b3x = jnp.full_like(b3y, xlim[0])
        b3 = jnp.hstack((b3x, b3y))

        x = jnp.concatenate((b0x, b1x, b2x, b3x))
        y = jnp.concatenate((b0y, b1y, b2y, b3y))
        xy = jnp.concatenate((x, y),axis=1)
        u = netmap(hessian)(params, jnp.concatenate((b0, b1, b2, b3))).reshape(-1, 4)
        
        u_true = jax.vmap(analytic.cart_stress_true)(xy, **kwargs).reshape(-1, 4)
        
        fig, ax = plt.subplots(2, 3, figsize=(30, 20))
        ax[0, 0].set_title("XX stress", fontsize=_FONTSIZE)
        ax[0, 0].plot(u_true[:, 0], linewidth=2)
        ax[0, 0].plot(u[:, 0], linewidth=2)
        ax[0, 0].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 0]), min(u[:, 0])), ymax = max(max(u_true[:, 0]), max(u[:, 0])), colors='black')
        
        ax[0, 1].set_title("XY stress", fontsize=_FONTSIZE)
        ax[0, 1].plot(u_true[:, 1], linewidth=2)
        ax[0, 1].plot(u[:, 1], linewidth=2)
        ax[0, 1].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 1]), min(u[:, 1])), ymax = max(max(u_true[:, 1]), max(u[:, 1])), colors='black')
                
        ax[0, 2].set_title("YY stress", fontsize=_FONTSIZE)
        ax[0, 2].plot(u_true[:, 3], linewidth=2)
        ax[0, 2].plot(u[:, 3], linewidth=2)
        ax[0, 2].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 3]), min(u[:, 3])), ymax = max(max(u_true[:, 3]), max(u[:, 3])), colors='black')
        
        ax[1, 0].set_title("XX error", fontsize=_FONTSIZE)
        ax[1, 0].semilogy(jnp.abs(u_true[:, 0] - u[:, 0]), linewidth=2)
        ax[1, 0].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 0] - u[:, 0])), colors='black')
         
        ax[1, 1].set_title("XY error", fontsize=_FONTSIZE)
        ax[1, 1].semilogy(jnp.abs(u_true[:, 1] - u[:, 1]), linewidth=2)
        ax[1, 1].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 1] - u[:, 1])), colors='black')
        
        ax[1, 2].set_title("YY error", fontsize=_FONTSIZE)
        ax[1, 2].semilogy(jnp.abs(u_true[:, 3] - u[:, 3]), linewidth=2)
        ax[1, 2].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 3] - u[:, 3])), colors='black')
        
        for i in ax.ravel():
            for item in ([i.xaxis.label, i.yaxis.label] + i.get_xticklabels() + i.get_yticklabels()):
                item.set_fontsize(20)
        
        save_fig(fig_dir, "boundaries", "png")