import os

import jax.numpy as jnp
import jax.tree_util as jtu
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

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

# def log_img(dir: str, file_name: str, image_tensor, step=None):
#     writer = SummaryWriter(log_dir=dir)
#     writer.add_image(file_name, image_tensor, global_step=step)
#     writer.close()
#     return


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


# def log_plot(X, Y, Z, name, log_dir, step=None, vmin=None, vmax=None, logscale=False, dpi=50):
#     fig = plt.figure(figsize=(10,10), dpi=dpi)
#     p = plt.contourf(X, Y, Z, vmin=vmin, vmax=vmax, levels=_CLEVELS)
#     plt.colorbar(p, ax=plt.gca())
#     log_figure(fig=plt.gcf(), name=name, log_dir=log_dir, step=step)
#     plt.close()
#     return

# def log_figure(fig, name, log_dir, step=None):
#     fig.canvas.draw()
    
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     w, h = fig.canvas.get_width_height()
#     data2 = data.reshape((int(h), int(w), -1))
#     im = data2.transpose((2, 0, 1))
    
#     log_img(log_dir, name, im, step=step)
#     return