from collections.abc import Sequence

import jax
import jax.numpy as jnp

from utils.transforms import xy2r, xy2theta, xy2rtheta, polar2cart_tensor, cart2polar_tensor

_TENSION = 10
_RADIUS = 2

def cart_stress_true(xy, **kwargs):

    x = xy[0]
    y = xy[1]
    diff_xx = -jnp.sin(x)*jnp.sin(y)
    diff_xy = jnp.cos(x)*jnp.cos(y)
    diff_yy = -jnp.sin(x)*jnp.sin(y)
    
    return jnp.array([diff_xx, diff_xy, diff_xy, diff_yy]).reshape(2, 2)


def get_true_vals(points: dict[str, jax.Array | tuple[dict[str, jax.Array]] | None],
                  *,
                  exclude: Sequence[str] | None = None,
                  noise: float | None = None,
                  key = None
                  ) -> dict[str, jax.Array | dict[str, jax.Array] | None]:
    vals = {}
    if exclude is None:
        exclude = []
    
    # Homogeneous PDE ==> RHS is zero
    if "coll" not in exclude:
        vals["coll"] = 4*jnp.sin(points["coll"][:, 0])*jnp.sin(points["coll"][:, 1])

    # True stresses in domain
    if "data" not in exclude:
        true_data = jax.vmap(cart_stress_true)(points["data"])
        vals["data"] = {}
        if noise is None:
            # Exact data
            vals["data"]["true_xx"] = true_data[:, 0, 0]
            vals["data"]["true_xy"] = true_data[:, 0, 1]
            vals["data"]["true_yy"] = true_data[:, 1, 1]
        else:
            # Noisy data
            if key is None:
                raise ValueError(f"PRNGKey must be specified.")
            keys = jax.random.split(key, 3)

            xx_noise = jax.random.normal(keys[0], true_data[:, 0, 0].shape)
            vals["data"]["true_xx"] = true_data[:, 0, 0] + \
                noise * (jnp.linalg.norm(true_data[:, 0, 0]) / jnp.linalg.norm(xx_noise)) * xx_noise
            xy_noise = jax.random.normal(keys[1], true_data[:, 0, 1].shape)
            vals["data"]["true_xy"] = true_data[:, 0, 1] + \
                noise * (jnp.linalg.norm(true_data[:, 0, 1]) / jnp.linalg.norm(xy_noise)) * xy_noise
            yy_noise = jax.random.normal(keys[2], true_data[:, 1, 1].shape)
            vals["data"]["true_yy"] = true_data[:, 1, 1] + \
                noise * (jnp.linalg.norm(true_data[:, 1, 1]) / jnp.linalg.norm(yy_noise)) * yy_noise
    
    # Only inhomogeneous BCs at two sides of rectangle
    if "rect" not in exclude:
        
        true_rect = [jax.vmap(cart_stress_true)(points["rect"][i]) for i in range(4)]

        rect = {
                "xx0":  true_rect[0][:, 0, 0],
                "xy0":  true_rect[0][:, 0, 1],
                "yy1":  true_rect[1][:, 1, 1],
                "xy1":  true_rect[1][:, 0, 1],
                "xx2":  true_rect[2][:, 0, 0],
                "xy2":  true_rect[2][:, 0, 1],
                "yy3":  true_rect[3][:, 1, 1],
                "xy3":  true_rect[3][:, 0, 1],
                "yy0":  true_rect[0][:, 1, 1], # extra
                "xx1":  true_rect[1][:, 0, 0], # extra
                "yy2":  true_rect[2][:, 1, 1], # extra
                "xx3":  true_rect[3][:, 0, 0]  # extra
                }


        vals["rect"] = rect
    
        
    if "diri" not in exclude:
        diri = {"di0": jnp.sin(points["rect"][0][:, 0])*jnp.sin(points["rect"][0][:, 1]),
                "di1": jnp.sin(points["rect"][1][:, 0])*jnp.sin(points["rect"][1][:, 1]),
                "di2": jnp.sin(points["rect"][2][:, 0])*jnp.sin(points["rect"][2][:, 1]),
                "di3": jnp.sin(points["rect"][3][:, 0])*jnp.sin(points["rect"][3][:, 1])}
        vals["diri"] = diri


    return vals