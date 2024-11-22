from functools import wraps
from collections.abc import Callable

from scipy.special import gamma
import jax
import jax.numpy as jnp

from setup.settings import DefaultSettings, SoftAdaptSettings



def get_fdm_coeff(order: int,
                  backward: bool = True
                  ) -> jax.Array:
    """
    Function for calculating finite difference coefficients.
    """
    if backward:
        lins = jnp.linspace(-order, 0, order+1)
    else:
        lins = jnp.linspace(0, order, order+1)
    exponents = jnp.linspace(0, order, order+1)
    powers = jnp.power(lins.reshape(-1, 1), exponents)
    factorials = jnp.array([gamma(i+1) for i in range(order+1)])
    
    # Set up linear system
    RHS = jnp.zeros((order+1,)).at[1].set(1.) # Set position of 1st derivative to 1.
    M = jnp.divide(powers, factorials) # Taylor expansion terms
    return jnp.linalg.solve(M.T, RHS)


def softmax(input: jax.Array,
            beta: float = 0.1,
            numerator_weights: jax.Array | None = None,
            shift_by_max_value: bool = True
            ) -> jax.Array:
    if shift_by_max_value:
        exp_of_input = jnp.exp(beta * (input - jnp.max(input)))
    else:
        exp_of_input = jnp.exp(beta * input)
        
    if numerator_weights is not None:
        exp_of_input = jnp.multiply(numerator_weights, exp_of_input)
        
    return exp_of_input / jnp.sum(exp_of_input + DefaultSettings.SOFTMAX_TOLERANCE)

def softadapt(sett: SoftAdaptSettings,
              prevlosses = None,
              fdm_coeff = None
              ) -> jax.Array:

    if prevlosses is None:
        return None
    
    if fdm_coeff is None:
        fdm_coeff = get_fdm_coeff(sett.order, backward=True)
    
    # Calculate loss slopes using finite difference
    rates_of_change = jnp.matmul(fdm_coeff, prevlosses)
    
    # Normalize slopes
    if sett.normalized_rates:
        rates_of_change = jnp.divide(rates_of_change, jnp.sum(jnp.abs(rates_of_change)))
    elif sett.delta_time is not None:
        rates_of_change = jnp.divide(rates_of_change, sett.delta_time)

    # Call custom SoftMax function
    weights = jnp.clip(softmax(rates_of_change, beta=sett.beta, shift_by_max_value=sett.shift_by_max_val), 0.01, 100)

    # Weight by loss values
    if sett.loss_weighted:
        avg_weights = jnp.multiply(jnp.mean(prevlosses, axis=0), weights)
        weights = jnp.divide(avg_weights, jnp.sum(avg_weights))
    
    if sett.normalized:
        weights = jnp.divide(weights, jnp.sum(weights))

    return jax.lax.stop_gradient(weights)