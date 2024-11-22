from collections.abc import Sequence, Callable
from functools import wraps
import os
import json
from time import perf_counter
import subprocess

import jax
import jax.numpy as jnp

import flax.linen as nn


def cyclic_diff(x: jax.Array) -> jax.Array:
    return jnp.subtract(x, jnp.roll(x, -1))


def remove_points(arr: jax.Array, fun: Callable) -> jax.Array:
    return arr[jnp.invert(fun(arr))].copy()


def keep_points(arr: jax.Array, fun: Callable) -> jax.Array:
    return arr[fun(arr)].copy()


def out_shape(fun, *args):
    print(f"{fun.__name__:<15}    {jax.eval_shape(fun, *args).shape}")


def limits2vertices(xlim: Sequence, ylim: Sequence) -> list[tuple[list]]:
    """
    Works for rectangles only. Returns pairs of end points for
    each of the four sides in counterclockwise order.
    """
    
    v = [
         ([xlim[0], ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
         ([xlim[1], ylim[0]], [xlim[1], ylim[1]]), # Right vertical
         ([xlim[1], ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
         ([xlim[0], ylim[1]], [xlim[0], ylim[0]])  # Left vertical
    ]
    return v


def normal_eq(x: jax.Array, y: jax.Array, base: list[Callable], ridge: float | None = None):
    X = jnp.concatenate([b(x).reshape(-1, 1) for b in base], axis=-1)
    XT_X = X.T @ X
    XT_y = X.T @ y
    if ridge is None:
        return jnp.linalg.solve(XT_X, XT_y).ravel()
    return jnp.linalg.solve(XT_X+ridge*jnp.identity(len(base)), XT_y).ravel()


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = perf_counter()
        v = func(*args, **kwargs)
        t2 = perf_counter()
        print(f"Time for function '{func.__name__}': {t2-t1:.6f} seconds")
        return v
    return wrapper


def find_first_integer(s: str):
    for i, c in enumerate(s):
        if c.isdigit():
            start = i
            while i < len(s) and s[i].isdigit():
                i += 1
            break
        
    return int(s[start:i])


def get_gpu_model():
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    line = line_as_bytes.decode("ascii")
    line = line.split("\n", 1)[0]
    print(line)
    _, line = line.split(":", 1)
    line, _ = line.split("(")
    return line.strip()


def find_nearest(array, value):
    array = jnp.asarray(array)
    idx = (jnp.abs(array - value)).argmin()
    return array[idx]


class WaveletActivation(nn.Module):
    param_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        c = self.param(
            "coeff", nn.initializers.normal(self.param_scale), (2,)
        )
        return jnp.add(jnp.multiply(c[0], jnp.cos(x)), jnp.multiply(c[1], jnp.sin(x)))