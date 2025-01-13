import os

import jax.numpy as jnp
import jax.tree_util as jtu
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from matplotlib import rc

_CLEVELS = 101

def save_fig(dir: str, file_name: str, format: str = "png",
             fig: Figure | None = None, clear = True, close = True, dpi=100):
    if not file_name.endswith("." + format):
        file_name += ("." + format)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(dir, file_name), format=format, bbox_inches="tight", dpi=dpi)
    if clear:
        fig.clf()
    if close:
        plt.close(fig)
    return


def multi_scatter(xy, t, kwargs: dict[dict]):
    xyflat = jtu.tree_flatten(xy)[0]
    tflat = jtu.tree_flatten(t)[0]
    for xy, tt in zip(xyflat, tflat):
        plt.scatter(xy[:, 0], xy[:, 1], **kwargs[tt])


def plot_polygon(ax, vertices: jnp.ndarray, *args, **kwargs):
    v = jnp.squeeze(vertices)
    if v.ndim != 2:
        raise ValueError(f"Input must only have 2 dimensions with length > 1, "
                         f"but there were {v.ndim}.")
    x = jnp.append(v[:, 0].ravel(), v[0, 0])
    y = jnp.append(v[:, 1].ravel(), v[0, 1])
    ax.plot(x, y, *args, **kwargs)
    return


def plot_circle(ax, radius: float, resolution: int, *args, angle = None, **kwargs) -> None:
    if angle is None:
        angle = [0, 2*jnp.pi]
    theta = jnp.linspace(*angle, resolution+1)
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    ax.plot(x, y, *args, **kwargs)
    return


def get_plot_variables(xlim, ylim, grid = 201):
    x = jnp.linspace(xlim[0], xlim[1], grid)
    y = jnp.linspace(ylim[0], ylim[1], grid)
    X, Y = jnp.meshgrid(x, y)
    plotpoints = jnp.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
    return X, Y, plotpoints


def plot_loss_history(train_loss_history, dir: str, file_name: str, format: str = "png", validation_loss_history = None, stepsize = 1):
    
    rc("text", usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')

    fig = plt.figure(figsize=(18,10))
    plt.grid(True)
    ll = jnp.arange(0, stepsize*jnp.size(train_loss_history), stepsize)
    llval = jnp.arange(0, stepsize*jnp.size(validation_loss_history), stepsize)

    if validation_loss_history is None:
        plt.semilogy(ll, train_loss_history)
    else:
        plt.semilogy(ll, train_loss_history, label='Training loss')
        plt.semilogy(llval, validation_loss_history, c='r', label='Validation loss')
    
    ax = fig.gca()
    ax.set_xticks(np.linspace(0, max(max(ll), max(llval)), 11, dtype=np.int32))
    ax.set_xticklabels(np.linspace(0, max(max(ll), max(llval)), 11, dtype=np.int32), fontsize=25)
    ax.set_xlabel(r"\textbf{Epochs}", fontsize=30)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_ylabel(r"\textbf{MSE}", fontsize=30, rotation=0, ha="right", labelpad=5)
    
    save_fig(dir, file_name, format)
    return