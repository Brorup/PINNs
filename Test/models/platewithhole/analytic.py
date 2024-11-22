from collections.abc import Sequence

import jax
import jax.numpy as jnp

from utils.transforms import xy2r, xy2theta, xy2rtheta, polar2cart_tensor, cart2polar_tensor

_TENSION = 10
_RADIUS = 2


def sigma_rr_true(r, theta, S = _TENSION, a = _RADIUS):
    a2 = jnp.array(a*a)
    r2 = jnp.square(r)
    a2divr2 = jnp.divide(a2, r2)
    return 0.5*S * ((1 + 3 * jnp.square(a2divr2) - 4 * a2divr2) * jnp.cos(2*theta) + (1 - a2divr2))


def sigma_tt_true(r, theta, S = _TENSION, a = _RADIUS):
    a2 = jnp.array(a*a)
    r2 = jnp.square(r)
    a2divr2 = jnp.divide(a2, r2)
    return 0.5*S * ((1 + a2divr2) - (1 + 3 * jnp.square(a2divr2)) * jnp.cos(2*theta))


def sigma_rt_true(r, theta, S = _TENSION, a = _RADIUS):
    a2 = jnp.array(a*a)
    r2 = jnp.square(r)
    a2divr2 = jnp.divide(a2, r2)
    return -0.5*S * (1 - 3 * jnp.square(a2divr2) + 2 * a2divr2) * jnp.sin(2*theta)


def polar_stress_true(rtheta, S = _TENSION, a = _RADIUS):

    # Compute polar stresses from analytical solutions
    rr_stress = sigma_rr_true(rtheta[0], rtheta[1], S=S, a=a)
    rt_stress = sigma_rt_true(rtheta[0], rtheta[1], S=S, a=a)
    tt_stress = sigma_tt_true(rtheta[0], rtheta[1], S=S, a=a)

    # Format as stress tensor
    return jnp.array([[rr_stress, rt_stress], [rt_stress, tt_stress]])


def cart_stress_true(xy, S = _TENSION, a = _RADIUS):

    # Map cartesian coordinates to polar coordinates
    rtheta = xy2rtheta(xy)
    
    # Compute true stress tensor in polar coordinates
    polar_stress_hessian = polar_stress_true(rtheta, S=S, a=a)

    # Convert polar stress tensor to cartesian stress tensor
    return polar2cart_tensor(polar_stress_hessian, rtheta)


def principal_stress_true(coords, polar = False, S = _TENSION, a = _RADIUS):
    if polar:
        p = polar_stress_true(coords, S=S, a=a)
    else:
        p = cart_stress_true(coords, S=S, a=a)
    return jnp.linalg.eigvalsh(p)


def von_mises_stress_true(coords, S = _TENSION, a = _RADIUS):
    p = cart_stress_true(coords, S=S, a=a)
    return jnp.sqrt(p[0, 0]**2 + p[1, 1]**2 - p[0, 0]*p[1, 1] + 3*p[0, 1]**2)


def get_true_vals(points: dict[str, jax.Array | tuple[dict[str, jax.Array]] | None],
                  *,
                  exclude: Sequence[str] | None = None,
                  ylim = None,
                  noise: float | None = None,
                  key = None,
                  radius = _RADIUS,
                  tension = _TENSION,
                  **kwargs
                  ) -> dict[str, jax.Array | dict[str, jax.Array] | None]:
    vals = {}
    if exclude is None:
        exclude = []
    
    # Homogeneous PDE ==> RHS is zero
    if "coll" not in exclude:
        coll = None
        # coll = jnp.zeros((points["coll"].shape[0], 1))
        vals["coll"] = coll

    # True stresses in domain
    if "data" not in exclude:
        true_data = jax.vmap(cart_stress_true, in_axes=(0, None, None))(points["data"], tension, radius)
        vals["data"] = {}
        if noise is None:
            # Exact data
            vals["data"]["true_sxx"] = true_data[:, 0, 0]
            vals["data"]["true_sxy"] = true_data[:, 0, 1]
            vals["data"]["true_syy"] = true_data[:, 1, 1]
        else:
            # Noisy data
            if key is None:
                raise ValueError(f"PRNGKey must be specified.")
            keys = jax.random.split(key, 3)

            xx_noise = jax.random.normal(keys[0], true_data[:, 0, 0].shape)
            vals["data"]["true_sxx"] = true_data[:, 0, 0] + \
                noise * (jnp.linalg.norm(true_data[:, 0, 0]) / jnp.linalg.norm(xx_noise)) * xx_noise
            xy_noise = jax.random.normal(keys[1], true_data[:, 0, 1].shape)
            vals["data"]["true_sxy"] = true_data[:, 0, 1] + \
                noise * (jnp.linalg.norm(true_data[:, 0, 1]) / jnp.linalg.norm(xy_noise)) * xy_noise
            yy_noise = jax.random.normal(keys[2], true_data[:, 1, 1].shape)
            vals["data"]["true_syy"] = true_data[:, 1, 1] + \
                noise * (jnp.linalg.norm(true_data[:, 1, 1]) / jnp.linalg.norm(yy_noise)) * yy_noise
    
    # Only inhomogeneous BCs at two sides of rectangle
    if "rect" not in exclude:
        
        true_rect = [jax.vmap(cart_stress_true, in_axes=(0, None, None))(points["rect"][i], tension, radius) for i in range(4)]
        
        # xx0 corresponds to the true value of phi_xx at the lower boundary
        rect = {
                "xx0":  true_rect[0][:, 1, 1],
                "xy0": -true_rect[0][:, 0, 1],
                "yy1":  true_rect[1][:, 0, 0],
                "xy1": -true_rect[1][:, 1, 0],
                "xx2":  true_rect[2][:, 1, 1],
                "xy2": -true_rect[2][:, 0, 1],
                "yy3":  true_rect[3][:, 0, 0],
                "xy3": -true_rect[3][:, 1, 0],
                "yy0":  true_rect[0][:, 0, 0], # extra
                "xx1":  true_rect[1][:, 1, 1], # extra
                "yy2":  true_rect[2][:, 0, 0], # extra
                "xx3":  true_rect[3][:, 1, 1]  # extra
                }


        vals["rect"] = rect
    
    # Homogeneous BC at inner circle
    if "circ" not in exclude:
        true_data = jax.vmap(cart_stress_true, in_axes=(0, None, None))(points["circ"], tension, radius)
        vals["circ"] = {}
        # Exact data
        vals["circ"]["true_scxx"] = true_data[:, 0, 0]
        vals["circ"]["true_scxy"] = true_data[:, 0, 1]
        vals["circ"]["true_scyy"] = true_data[:, 1, 1]
        
        polar_points = jax.vmap(xy2rtheta)(points["circ"])
        true_tt = jax.vmap(sigma_tt_true, in_axes=(0, 0, None, None))(polar_points[:, 0], polar_points[:, 1], tension, radius)
        vals["circ"]["stt"] = true_tt
        
    
    # If used, constrains the solution to a specific one, namely
    # the one where the terms of order 0 and 1 vanish.
    """
                    _..._
                ..**     **..
              .*             *.
            .*                 *.
           /                   /
          /    _..._          /
         / ..**     **..     /
        /.*             *.  /
       /*                 */
    
    """
    if "diri" not in exclude:
        if ylim is None:
            print("Y-limit defaults to [-10.0, 10.0]")
            ylim = [-10.0, 10.0]
        diri = {"di1": 0.5*_TENSION*(points["rect"][1][:, 1]-ylim[0])*(points["rect"][1][:, 1]-ylim[1]),
                "di3": 0.5*_TENSION*(points["rect"][3][:, 1]-ylim[0])*(points["rect"][3][:, 1]-ylim[1])}
        vals["diri"] = diri

    return vals