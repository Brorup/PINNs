from collections.abc import Sequence

import jax
import jax.numpy as jnp

def get_true_vals(points: dict[str, jax.Array | tuple[dict[str, jax.Array]] | None],
                  *,
                  exclude: Sequence[str] | None = None,
                  noise: float | None = None,
                  key = None
                  ) -> dict[str, jax.Array | dict[str, jax.Array] | None]:
    vals = {}
    if exclude is None:
        exclude = []
    
    point_types = ["coll", "bc", "data"]
    [exclude.append(i) if points.get(i) is None else None for i in point_types]
    
    # Homogeneous PDE ==> RHS is zero
    if "coll" not in exclude:
        vals["coll"] = 2*jnp.pi**2*jnp.sin(jnp.pi*points["coll"][:, 0])*jnp.sin(jnp.pi*points["coll"][:, 1])

        
    if "bc" not in exclude:
        bc = jnp.sin(jnp.pi*points["bc"][:, 0])*jnp.sin(jnp.pi*points["bc"][:, 1])
        vals["bc"] = bc

    if "data" not in exclude:
        data = jnp.sin(jnp.pi*points["data"][:, 0])*jnp.sin(jnp.pi*points["data"][:, 1])
        vals["data"] = data

    return vals