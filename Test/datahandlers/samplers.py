from collections.abc import Sequence
import math

from scipy.stats.qmc import Sobol
import jax
import jax.numpy as jnp


def sample_line(key,
                end_points: Sequence[Sequence], # Outer sequence length should be 2 (there are 2 end points)
                *args,
                ref_scale: float = 1.0,
                ref_offset: float = 0.0,
                distribution: str = "uniform",
                round_up: bool = False,
                **kwargs) -> jax.Array:
    if len(end_points) != 2:
        raise ValueError(f"Length of argument 'end_points' should be 2 but was {len(end_points)}.")
    if distribution == "uniform":
        sample_fun = jax.random.uniform
    elif distribution == "sobol":
        def sobol_fun(key, *args, **kwargs):
            s = Sobol(1, seed=int(jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
                ).random_base2(math.ceil(jnp.log2(kwargs["shape"][0])))
            if not round_up:
                return s[:kwargs["shape"][0]]
            return s
        sample_fun = sobol_fun
    else:
        raise ValueError("Unknown sampling distribution.")
    sample_points = sample_fun(key, *args, **kwargs)
    ref_points = (sample_points - ref_offset) / ref_scale
    p1 = jnp.array(end_points[0])
    p2 = jnp.array(end_points[1])
    return p1*ref_points + p2*(1-ref_points)