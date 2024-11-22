from functools import partial

import jax
import jax.numpy as jnp


def sq(r: jax.Array) -> jax.Array:
    return jnp.square(r)


def sqe(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return sq(u)
    return sq(jnp.subtract(u.ravel(), u_true.ravel()))


def ms(r: jax.Array) -> float:
    return jnp.mean(jnp.square(r))


def mse(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return ms(u)
    return ms(jnp.subtract(u.ravel(), u_true.ravel()))


def ma(r: jax.Array) -> float:
    return jnp.mean(jnp.abs(r))


def mae(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return ma(u)
    return ma(jnp.subtract(u.ravel(), u_true.ravel()))


def maxabse(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return jnp.max(jnp.abs(u))
    return jnp.max(jnp.abs(jnp.subtract(u.ravel(), u_true.ravel())))


def Lp_rel(u: jax.Array, u_true: jax.Array, p: int = 2):
    return jnp.linalg.norm(jnp.subtract(u.ravel(), u_true.ravel()), ord=p) / jnp.linalg.norm(u_true.ravel(), ord=p)
    #return jnp.sqrt(jnp.divide(mse(u, u_true), ms(u_true))) # means of num/denom cancel

def L2rel(u: jax.Array, u_true: jax.Array):
    return jax.jit(partial(Lp_rel, p=2))(u, u_true)
    

def _norme(u: jax.Array, u_true: jax.Array | None = None, order = 2):
    if u_true is None:
        return jnp.linalg.norm(u, ord = order)
    return jnp.linalg.norm(jnp.subtract(u.ravel(), u_true.ravel()), ord = order)

def pnorm(order = 2):
    return jax.jit(partial(_norme, order=order))

def running_average(new_weights: jax.Array, old_weights: jax.Array, alpha: float | jax.Array, normalized: bool = False):
    alpha = jnp.clip(alpha, a_min=0, a_max=1)
    avg_weights = alpha*new_weights + (1-alpha)*old_weights
    if normalized:
        return avg_weights / jnp.sum(avg_weights)
    return avg_weights