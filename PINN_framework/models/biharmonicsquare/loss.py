from typing import Callable

import jax
import jax.numpy as jnp

from models.loss import mse
from utils.transforms import xy2theta


def loss_rect(*output: jax.Array, true_val: dict[str, jax.Array] | None = None, loss_fn: Callable = mse):
    """
    Computes loss of the BC residuals on the four sides of the rectangle.
    
    Layout of rectangle sides:
    ```
                        ^
                      y |
                        |
                        *------>
                                x
                            
                                     2
                             _________________
            xx stress  <-   |                 |   -> xx stress
                            |                 |
                         3  |        O        |  1
                            |                 |
            xx stress  <-   |_________________|   -> xx stress
                            
                                     0
    ```
    """
    # Unpack outputs
    out0, out1, out2, out3 = output

    # Return all losses
    if true_val is None:
        return loss_fn(out0[:, 0]), loss_fn(out0[:, 1]), \
               loss_fn(out1[:, 3]), loss_fn(out1[:, 2]), \
               loss_fn(out2[:, 0]), loss_fn(out2[:, 1]), \
               loss_fn(out3[:, 3]), loss_fn(out3[:, 2])
               
    return loss_fn(out0[:, 0], true_val.get("xx0")), loss_fn(out0[:, 1], true_val.get("xy0")), \
           loss_fn(out1[:, 3], true_val.get("yy1")), loss_fn(out1[:, 2], true_val.get("xy1")), \
           loss_fn(out2[:, 0], true_val.get("xx2")), loss_fn(out2[:, 1], true_val.get("xy2")), \
           loss_fn(out3[:, 3], true_val.get("yy3")), loss_fn(out3[:, 2], true_val.get("xy3"))


def loss_rect_extra(*output: jax.Array, true_val: dict[str, jax.Array] | None = None, loss_fn: Callable = mse):

    # Unpack outputs
    out0, out1, out2, out3 = output

    # Return all losses
    if true_val is None:
        return loss_fn(out0[:, 3]), loss_fn(out1[:, 0]), \
               loss_fn(out2[:, 3]), loss_fn(out3[:, 0])
    return loss_fn(out0[:, 3], true_val.get("yy0")), loss_fn(out1[:, 0], true_val.get("xx1")), \
           loss_fn(out2[:, 3], true_val.get("yy2")), loss_fn(out3[:, 0], true_val.get("xx3"))


def loss_dirichlet(*output: jax.Array, true_val: dict[str, jax.Array] | None = None, loss_fn: Callable = mse):

    # Unpack outputs
    out0, out1, out2, out3 = output
    
    # Return all losses
    if true_val is None:
        return loss_fn(out0), loss_fn(out1), loss_fn(out2), loss_fn(out3)
    return loss_fn(out0, true_val.get("di0")), \
           loss_fn(out1, true_val.get("di1")), \
           loss_fn(out2, true_val.get("di2")), \
           loss_fn(out3, true_val.get("di3"))


def loss_data(output: jax.Array, true_val: dict[str, jax.Array] | None = None, loss_fn: Callable = mse):

    if true_val is None:
        return loss_fn(output[:, 3]), loss_fn(output[:, 1]), loss_fn(output[:, 0])
    return loss_fn( output[:, 0], true_val.get("true_xx")), \
           loss_fn( output[:, 1], true_val.get("true_xy")), \
           loss_fn( output[:, 3], true_val.get("true_yy")),