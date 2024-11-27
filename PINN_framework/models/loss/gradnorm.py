from functools import wraps
from collections.abc import Callable

from scipy.special import gamma
import jax
import jax.numpy as jnp

from setup.settings import DefaultSettings, GradNormSettings

def gradnorm(sett: GradNormSettings, grads, num_losses: int, loss_weights: jax.Array | None = None) -> jax.Array:
    grad_per_loss = jnp.concatenate([i.reshape(num_losses, -1) for i in jax.tree.leaves(grads)], axis=1)
    grad_norms = jnp.linalg.norm(grad_per_loss, axis=1)
    grad_sum = jnp.mean(grad_norms)
    
    weights = jax.tree_util.tree_map(lambda x: (grad_sum / x), grad_norms)
    
    if sett.loss_weighted:
        if loss_weights is not None:
            weights = jnp.divide(w:=jnp.multiply(loss_weights, weights), jnp.sum(w))

    if sett.normalized:
        weights = jnp.divide(weights, jnp.sum(weights))
    
    return jax.lax.stop_gradient(weights)