import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import jax
import jax.numpy as jnp

from . import analytic
from models.networks import netmap
from utils.plotting import (
    save_fig,
    plot_circle,
    log_plot,
    get_plot_variables
)
from utils.transforms import (
    cart2polar_tensor,
    xy2r,
    rtheta2xy,
    vrtheta2xy,
    vxy2rtheta
)



_DEFAULT_RADIUS = 2
_DEFAULT_CIRCLE_RES = 100
_CLEVELS = 381
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
    figsize = (35, 30)
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

    save_fig(fig_dir, name, extension, fig=fig)
    plt.clf()
    return

def get_plot_data(geometry_settings, hessian, params, grid, **kwargs):

    radius = geometry_settings["domain"]["circle"]["radius"]
    xlim = geometry_settings["domain"]["rectangle"]["xlim"]
    ylim = geometry_settings["domain"]["rectangle"]["ylim"]
    angle = geometry_settings["domain"]["circle"].get("angle")
    if angle is None:
        angle = [0, 2*jnp.pi]

    X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=grid)
    R, THETA, plotpoints_polar = get_plot_variables([radius, max(xlim[1], ylim[1])], angle, grid=grid)
    plotpoints2 = jax.vmap(rtheta2xy)(plotpoints_polar)

    assert(jnp.allclose(plotpoints, vrtheta2xy(vxy2rtheta(plotpoints)), atol=1e-4))

    # Hessian prediction
    phi_pp = netmap(hessian)(params, plotpoints).reshape(-1, 4)

    # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
    sigma_cart = phi_pp[:, [3, 1, 2, 0]]
    sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

    # List and reshape the four components
    sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]
    # To plot without circle mask
    # sigma_cart_list = [sigma_cart[:, i].reshape(X.shape) for i in range(4)]

    # Repeat for the other set of points (polar coords converted to cartesian coords)
    phi_pp2 = netmap(hessian)(params, plotpoints2).reshape(-1, 4)

    # Calculate stress from phi function
    sigma_cart2 = phi_pp2[:, [3, 1, 2, 0]]
    sigma_cart2 = sigma_cart2.at[:, [1, 2]].set(-phi_pp2[:, [1, 2]])

    # Convert these points to polar coordinates before listing and reshaping
    sigma_polar = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigma_cart2.reshape(-1, 2, 2), plotpoints2).reshape(-1, 4)
    sigma_polar_list = [sigma_polar[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]

    # Calculate true stresses (cartesian and polar)
    sigma_cart_true = jax.vmap(analytic.cart_stress_true)(plotpoints, **kwargs)
    sigma_cart_true_list = [sigma_cart_true.reshape(-1, 4)[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]
    sigma_polar_true = jax.vmap(analytic.polar_stress_true)(plotpoints_polar, **kwargs)
    sigma_polar_true_list = [sigma_polar_true.reshape(-1, 4)[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]

    return X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list


def plot_results(geometry_settings, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, **kwargs):

    X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list = get_plot_data(geometry_settings, hessian, params, grid=grid, **kwargs)
    radius = geometry_settings["domain"]["circle"]["radius"]
    angle = geometry_settings["domain"]["circle"].get("angle")

    if save:
        plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir=fig_dir, name="Cart_stress", radius=radius, angle=angle)
        plot_vm_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir=fig_dir, name="VM_stress", radius=radius, angle=angle)
        plot_polar_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, fig_dir=fig_dir, name="Polar_stress")
    if log:        
        log_stress(X, Y, sigma_cart_list, sigma_cart_true_list, log_dir=log_dir, name="Cart_stress", varnames="XY", step=step, dpi=dpi)
        log_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, log_dir=log_dir, name="Polar_stress", varnames="RT", step=step, dpi=dpi)

    return

def plot_potential(X, Y, Z, *, fig_dir, name,
                   extension="png",
                   radius = _DEFAULT_RADIUS,
                   angle = None,
                   circle_res = _DEFAULT_CIRCLE_RES):
    """
    Function for plotting potential function.
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Prediction", fontsize=20)
    p = ax.contourf(X, Y, Z, levels=_CLEVELS)
    plt.colorbar(p, ax=ax)
    
    plot_circle(plt, radius, circle_res, angle=angle, color="red")
    save_fig(fig_dir, name, extension)
    plt.clf()
    return


def plot_stress(X, Y, Z, Z_true, *, fig_dir, name,
                extension = "pdf",
                radius = _DEFAULT_RADIUS,
                circle_res = _DEFAULT_CIRCLE_RES,
                angle = None,
                figsize = (25, 30)):
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
        [jnp.abs(Z[i]-Z_true[i])*(xy2r(X, Y) >= radius) for i in hess_idx]
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
            ax[r, c].add_patch(plt.Circle((0, 0), radius=radius+0.05, color="#777777"))
            if cc < 2:
                cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(jnp.min(u[cc][r]), jnp.max(u[cc][r]), 11))
            else:
                cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(0, jnp.max(u[cc][r]), 11))
            cbar.ax.tick_params(labelsize=_TICKLABELSIZE)

    fig.tight_layout(pad=3.0)
    save_fig(fig_dir, name, extension)
    plt.clf()
    return


def plot_vm_stress(X, Y, Z, Z_true, *, fig_dir, name,
                extension = "pdf",
                radius = _DEFAULT_RADIUS,
                circle_res = _DEFAULT_CIRCLE_RES,
                angle = None,
                figsize = (25, 10)):
    """
    Function for plotting stresses in cartesian coordinates.
    """
    rc("text", usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')

    vm_stress = jnp.sqrt(Z[0]**2 + Z[3]**2 - Z[0]*Z[3] + 3*Z[1]**2)
    vm_stress_true = jnp.sqrt(Z_true[0]**2 + Z_true[3]**2 - Z_true[0]*Z_true[3] + 3*Z_true[1]**2)
    
    u = [[vm_stress], [vm_stress_true], [jnp.abs(vm_stress - vm_stress_true)]]
    
    all_titles = [[
        r"Prediction ($\sigma_{\text{v}}$)"
    ], [
        r"True solution ($\sigma_{\text{v}}$)"
    ], [
        r"Absolute error ($\sigma_{\text{v}}$)"
    ]]

    include = [0, 2]

    fig, ax = plt.subplots(1, len(include), figsize=figsize, squeeze=False)
    r = 0
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
        ax[r, c].add_patch(plt.Circle((0, 0), radius=radius+0.05, color="#777777"))
        if cc < 2:
            cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(jnp.min(u[cc][r]), jnp.max(u[cc][r]), 11))
        else:
            cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(0, jnp.max(u[cc][r]), 11))
        cbar.ax.tick_params(labelsize=_TICKLABELSIZE)

    fig.tight_layout(pad=3.0)
    save_fig(fig_dir, name, extension)
    plt.clf()
    return
    

def plot_polar_stress(X, Y, Z, Z_true, *, fig_dir, name, 
                      extension = "pdf", 
                      figsize = (30, 30)):
    """
    Function for plotting stresses in polar coordinates.
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
        r"Prediction ($\sigma_{rr}$)",
        r"Prediction ($\sigma_{r\theta}$)",
        r"Prediction ($\sigma_{\theta\theta}$)"
    ], [
        r"True solution ($\sigma_{rr}$)",
        r"True solution ($\sigma_{r\theta}$)",
        r"True solution ($\sigma_{\theta\theta}$)"
    ], [
        r"Absolute error ($\sigma_{rr}$)",
        r"Absolute error ($\sigma_{r\theta}$)",
        r"Absolute error ($\sigma_{\theta\theta}$)"
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
            ax[r, c].set_xlabel(r"$r$", fontsize=_LABELSIZE)
            ax[r, c].set_ylabel(r"$\theta$", rotation=0, fontsize=_LABELSIZE)
            ax[r, c].set_xticks([2., 4., 6., 8., 10.])
            ax[r, c].set_yticks([0., 0.5*jnp.pi, jnp.pi, 1.5*jnp.pi, 2.*jnp.pi])
            ax[r, c].set_xticklabels([r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=_TICKLABELSIZE)
            ax[r, c].set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=_TICKLABELSIZE)
            if cc < 2:
                cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(jnp.min(u[cc][r]), jnp.max(u[cc][r]), 11))
            else:
                cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(0, jnp.max(u[cc][r]), 11))
            cbar.ax.tick_params(labelsize=_TICKLABELSIZE)

    fig.tight_layout(pad=3.0)
    save_fig(fig_dir, name, extension)

    plt.clf()
    return



def plot_true_sol(geometry_settings, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, extension = "pdf", figsize = (30, 30), **kwargs,):
    
    X, Y, R, THETA, sigma_cart_list, Z_true, sigma_polar_list, Zp_true = get_plot_data(geometry_settings, hessian, params, grid=grid, **kwargs)
    radius = geometry_settings["domain"]["circle"]["radius"]
    angle = geometry_settings["domain"]["circle"].get("angle")
    
    rc("text", usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')

    hess_idx = [0, 1, 3]

    u = [
        [Z_true[i] for i in hess_idx],
        [Zp_true[i] for i in hess_idx],
    ]
    all_titles = [[        
        r"True solution ($\sigma_{xx}$)",
        r"True solution ($\sigma_{xy}$)",
        r"True solution ($\sigma_{yy}$)"
    ], [
        r"True solution ($\sigma_{rr}$)",
        r"True solution ($\sigma_{r\theta}$)",
        r"True solution ($\sigma_{\theta\theta}$)"
    ]]

    include = [0, 1]

    fig, ax = plt.subplots(len(hess_idx), len(include), figsize=figsize)
    for r, h in enumerate(hess_idx):
        for c, cc in enumerate(include):
            ax[r, c].set_aspect('equal', adjustable='box')
            ax[r, c].set_title(all_titles[cc][r], fontsize=_FONTSIZE, pad=20)
            if cc == 0:
                p = ax[r, c].contourf(X , Y, u[cc][r], levels=_CLEVELS, cmap="jet")
            else:
                p = ax[r, c].contourf(R , THETA, u[cc][r], levels=_CLEVELS, cmap="jet")
            p.set_edgecolor("face")
            if cc == 0:
                ax[r, c].set_xlabel(r"$x$", fontsize=_LABELSIZE)
                ax[r, c].set_ylabel(r"$y$", rotation=0, fontsize=_LABELSIZE)
                ax[r, c].set_xticks([-10., -5., 0., 5., 10.])
                ax[r, c].set_yticks([-10., -5., 0., 5., 10.])
                ax[r, c].set_xticklabels([r"$-10$", r"$-5$", r"$0$", r"$5$", r"$10$"], fontsize=_TICKLABELSIZE)
                ax[r, c].set_yticklabels([r"$-10$", r"$-5$", r"$0$", r"$5$", r"$10$"], fontsize=_TICKLABELSIZE)
                ax[r, c].add_patch(plt.Circle((0, 0), radius=radius+0.05, color="#777777"))
            else:
                ax[r, c].set_xlabel(r"$r$", fontsize=_LABELSIZE)
                ax[r, c].set_ylabel(r"$\theta$", rotation=0, fontsize=_LABELSIZE)
                ax[r, c].set_xticks([2., 4., 6., 8., 10.])
                ax[r, c].set_yticks([0., 0.5*jnp.pi, jnp.pi, 1.5*jnp.pi, 2.*jnp.pi])
                ax[r, c].set_xticklabels([r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"], fontsize=_TICKLABELSIZE)
                ax[r, c].set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=_TICKLABELSIZE)
            
            cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(jnp.min(u[cc][r]), jnp.max(u[cc][r]), 11))
            cbar.ax.tick_params(labelsize=_TICKLABELSIZE)

    fig.tight_layout(pad=3.0)
    save_fig(fig_dir, "Cart_Polar_true", extension)

    plt.clf()
    
    vm_stress_true = jnp.sqrt(Z_true[0]**2 + Z_true[3]**2 - Z_true[0]*Z_true[3] + 3*Z_true[1]**2)
    
    u = [[vm_stress_true]]
         
    all_titles = [[r"True solution ($\sigma_{\text{v}}$)"]]

    fig, ax = plt.subplots(1, 1, figsize=(15, 10), squeeze=False)
    r = 0
    include = [0]
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
        ax[r, c].add_patch(plt.Circle((0, 0), radius=radius+0.05, color="#777777"))
        if cc < 2:
            cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(jnp.min(u[cc][r]), jnp.max(u[cc][r]), 11))
        else:
            cbar = plt.colorbar(p, ax=ax[r, c], ticks=np.linspace(0, jnp.max(u[cc][r]), 11))
        cbar.ax.tick_params(labelsize=_TICKLABELSIZE)

    fig.tight_layout(pad=3.0)
    save_fig(fig_dir, "VM_true", extension)
    
    plt.clf()
    return


def log_stress(X, Y, Z, Z_true, *, log_dir, name, step=None, varnames="XY", dpi=50):
        
    # Log plots
    log_plot(X, Y, Z[0], name=name+"/Surrogate/"+varnames[0]+varnames[0], log_dir=log_dir, step=step,
            vmin=min(jnp.min(Z_true[0]),jnp.min(Z[0])), 
            vmax=max(jnp.max(Z_true[0]),jnp.max(Z[0])), dpi=dpi)
    
    log_plot(X, Y, Z[1], name=name+"/Surrogate/"+varnames[0]+varnames[1], log_dir=log_dir, step=step,
            vmin=min(jnp.min(Z_true[1]),jnp.min(Z[1])), 
            vmax=max(jnp.max(Z_true[1]),jnp.max(Z[1])), dpi=dpi)
            
    log_plot(X, Y, Z[3], name=name+"/Surrogate/"+varnames[1]+varnames[1], log_dir=log_dir, step=step,
            vmin=min(jnp.min(Z_true[3]),jnp.min(Z[3])), 
            vmax=max(jnp.max(Z_true[3]),jnp.max(Z[3])), dpi=dpi)
    
    
    log_plot(X, Y, jnp.abs(Z_true[0] - Z[0]), name=name+"/Error/"+varnames[0]+varnames[0], log_dir=log_dir, step=step, logscale=True, dpi=dpi)
    
    log_plot(X, Y, jnp.abs(Z_true[1] - Z[1]), name=name+"/Error/"+varnames[0]+varnames[1], log_dir=log_dir, step=step, logscale=True, dpi=dpi)

    log_plot(X, Y, jnp.abs(Z_true[3] - Z[3]), name=name+"/Error/"+varnames[1]+varnames[1], log_dir=log_dir, step=step, logscale=True, dpi=dpi)
            
    
        # These are redundant after first time being logged
    if step == 0:
        log_plot(X, Y, Z_true[0], name=name+"/True/"+varnames[0]+varnames[0], log_dir=log_dir, step=step,
                vmin=min(jnp.min(Z_true[0]),jnp.min(Z[0])), 
                vmax=max(jnp.max(Z_true[0]),jnp.max(Z[0])), dpi=dpi)
        
        log_plot(X, Y, Z_true[1], name=name+"/True/"+varnames[0]+varnames[1], log_dir=log_dir, step=step,
                vmin=min(jnp.min(Z_true[1]),jnp.min(Z[1])), 
                vmax=max(jnp.max(Z_true[1]),jnp.max(Z[1])), dpi=dpi)
                
        log_plot(X, Y, Z_true[3], name=name+"/True/"+varnames[1]+varnames[1], log_dir=log_dir, step=step,
                vmin=min(jnp.min(Z_true[3]),jnp.min(Z[3])), 
                vmax=max(jnp.max(Z_true[3]),jnp.max(Z[3])), dpi=dpi)
            
            
def plot_boundaries(geometry_settings, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, **kwargs):
        # xlim = geometry_settings["domain"]["rectangle"]["xlim"]
        # ylim = geometry_settings["domain"]["rectangle"]["ylim"]

        # b0x = jnp.linspace(xlim[0], xlim[1], grid).reshape(-1,1)
        # b0y = jnp.full_like(b0x, ylim[0])
        # b0 = jnp.hstack((b0x, b0y))
        
        # b1y = jnp.linspace(ylim[0], ylim[1], grid).reshape(-1,1)
        # b1x = jnp.full_like(b1y, xlim[1])
        # b1 = jnp.hstack((b1x, b1y))
        
        # b2x = jnp.linspace(xlim[1], xlim[0], grid).reshape(-1,1)
        # b2y = jnp.full_like(b2x, ylim[1])
        # b2 = jnp.hstack((b2x, b2y))

        # b3y = jnp.linspace(ylim[1], ylim[0], grid).reshape(-1,1)
        # b3x = jnp.full_like(b3y, xlim[0])
        # b3 = jnp.hstack((b3x, b3y))

        # x = jnp.concatenate((b0x, b1x, b2x, b3x))
        # y = jnp.concatenate((b0y, b1y, b2y, b3y))
        # xy = jnp.concatenate((x, y),axis=1)
        # u = netmap(hessian)(params, jnp.concatenate((b0, b1, b2, b3))).reshape(-1, 4)
        
        # u_true = jax.vmap(analytic.cart_stress_true)(xy, **kwargs).reshape(-1, 4)
        
        # fig, ax = plt.subplots(2, 3, figsize=(30, 20))
        # ax[0, 0].set_title("XX stress", fontsize=_FONTSIZE)
        # ax[0, 0].plot(u_true[:, 0], linewidth=2)
        # ax[0, 0].plot(u[:, 0], linewidth=2)
        # ax[0, 0].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 0]), min(u[:, 0])), ymax = max(max(u_true[:, 0]), max(u[:, 0])), colors='black')
        
        # ax[0, 1].set_title("XY stress", fontsize=_FONTSIZE)
        # ax[0, 1].plot(u_true[:, 1], linewidth=2)
        # ax[0, 1].plot(u[:, 1], linewidth=2)
        # ax[0, 1].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 1]), min(u[:, 1])), ymax = max(max(u_true[:, 1]), max(u[:, 1])), colors='black')
                
        # ax[0, 2].set_title("YY stress", fontsize=_FONTSIZE)
        # ax[0, 2].plot(u_true[:, 3], linewidth=2)
        # ax[0, 2].plot(u[:, 3], linewidth=2)
        # ax[0, 2].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 3]), min(u[:, 3])), ymax = max(max(u_true[:, 3]), max(u[:, 3])), colors='black')
        
        # ax[1, 0].set_title("XX error", fontsize=_FONTSIZE)
        # ax[1, 0].semilogy(jnp.abs(u_true[:, 0] - u[:, 0]), linewidth=2)
        # ax[1, 0].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 0] - u[:, 0])), colors='black')
         
        # ax[1, 1].set_title("XY error", fontsize=_FONTSIZE)
        # ax[1, 1].semilogy(jnp.abs(u_true[:, 1] - u[:, 1]), linewidth=2)
        # ax[1, 1].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 1] - u[:, 1])), colors='black')
        
        # ax[1, 2].set_title("YY error", fontsize=_FONTSIZE)
        # ax[1, 2].semilogy(jnp.abs(u_true[:, 3] - u[:, 3]), linewidth=2)
        # ax[1, 2].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 3] - u[:, 3])), colors='black')
        
        # for i in ax.ravel():
        #     for item in ([i.xaxis.label, i.yaxis.label] + i.get_xticklabels() + i.get_yticklabels()):
        #         item.set_fontsize(20)
        
        # save_fig(fig_dir, "boundaries", "png")
        
        
        
        rc("text", usetex=True)
        rc('text.latex', preamble=r'\usepackage{amsmath}')
        
        xlim = geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = geometry_settings["domain"]["rectangle"]["ylim"]

        b0x = jnp.linspace(xlim[0], xlim[1], grid).reshape(-1,1)
        b0y = jnp.full_like(b0x, ylim[0])
        b0 = jnp.hstack((b0x, b0y))
        
        b1y = jnp.linspace(ylim[0], ylim[1], grid).reshape(-1,1)
        b1x = jnp.full_like(b1y, xlim[1])
        b1 = jnp.hstack((b1x, b1y))

        x = jnp.concatenate((b0x, b1x))
        y = jnp.concatenate((b0y, b1y))
        xy = jnp.concatenate((x, y),axis=1)
        
        fig, ax = plt.subplots(2, 2, figsize=(20, 15))
        ax[0, 0].set_title(r"$\sigma_{xx}$ on $\partial \Omega_{x_\ell}$", fontsize=_FONTSIZE)
        ax[0, 0].plot(b1y, jax.vmap(analytic.cart_stress_true)(b1, **kwargs)[:, 0, 0], linewidth=2, label='Analytical')
        ax[0, 0].plot(b1y, jax.vmap(analytic.cart_stress_true)(b1, **kwargs)[:, 0, 0]*0 + 10, '--', linewidth=2, label='Idealized')
        ax[0, 0].set_ylim(-2, 12)
        
        ax[0, 1].set_title(r"$\sigma_{xy}$ on $\partial \Omega_{x_\ell}$", fontsize=_FONTSIZE)
        ax[0, 1].plot(b1y, jax.vmap(analytic.cart_stress_true)(b1, **kwargs)[:, 0, 1], linewidth=2, label='Analytical')
        ax[0, 1].plot(b1y, jax.vmap(analytic.cart_stress_true)(b1, **kwargs)[:, 0, 1]*0, '--', linewidth=2, label='Idealized')
        ax[0, 1].set_ylim(-2, 12)

        ax[1, 0].set_title(r"$\sigma_{yy}$ on $\partial \Omega_{y_\ell}$", fontsize=_FONTSIZE)
        ax[1, 0].plot(b0x, jax.vmap(analytic.cart_stress_true)(b0, **kwargs)[:, 1, 1], linewidth=2, label='Analytical')
        ax[1, 0].plot(b0x, jax.vmap(analytic.cart_stress_true)(b0, **kwargs)[:, 1, 1]*0, '--', linewidth=2, label='Idealized')
        ax[1, 0].set_ylim(-6, 6)
        
        ax[1, 1].set_title(r"$\sigma_{xy}$ on $\partial \Omega_{y_\ell}$", fontsize=_FONTSIZE)
        ax[1, 1].plot(b0x, jax.vmap(analytic.cart_stress_true)(b0, **kwargs)[:, 1, 0], linewidth=2, label='Analytical')
        ax[1, 1].plot(b0x, jax.vmap(analytic.cart_stress_true)(b0, **kwargs)[:, 1, 0]*0, '--', linewidth=2, label='Idealized')
        ax[1, 1].set_ylim(-6, 6)
        
        
        ax[1, 0].set_xlabel(r"$x$", fontsize=_LABELSIZE)
        ax[1, 0].set_ylabel(r"$y$", rotation=0, fontsize=_LABELSIZE)
        ax[1, 1].set_xlabel(r"$x$", fontsize=_LABELSIZE)
        ax[0, 0].set_ylabel(r"$y$", rotation=0, fontsize=_LABELSIZE)
        for r in range(2):
            for c in range(2):
                ax[r, c].set_aspect('equal')
                ax[r, c].tick_params(labelsize=_TICKLABELSIZE)
                # ax[r, c].legend(fontsize=_TICKLABELSIZE)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.14, 0.9), fontsize=_LABELSIZE)
        
        save_fig(fig_dir, "boundaries", "pdf")