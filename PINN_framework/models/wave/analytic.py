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
    
    point_types = ["coll", "bc", "ic0", "ic1", "data"]
    [exclude.append(i) if points.get(i) is None else None for i in point_types]
    
    # Homogeneous PDE ==> RHS is zero
    if "coll" not in exclude:
        vals["coll"] = 0*points["coll"][:, 0]
        
    if "bc" not in exclude:
        bc = (points["bc"][:, 1] - points["bc"][:, 0] + 1)*jnp.sin(points["bc"][:, 1] - points["bc"][:, 0]) + jnp.sin(points["bc"][:, 1] + points["bc"][:, 0])*(points["bc"][:, 1] + points["bc"][:, 0] + 1)
        vals["bc"] = bc

    if "ic0" not in exclude:
        ic = 2*jnp.sin(points["ic0"][:, 0])*points["ic0"][:, 0]
        vals["ic0"] = ic

    if "ic1" not in exclude:
        ic = 2*jnp.cos(points["ic1"][:, 0])
        vals["ic1"] = ic

    if "data" not in exclude:
        data = (points["data"][:, 1] - points["data"][:, 0] + 1)*jnp.sin(points["data"][:, 1] - points["data"][:, 0]) + jnp.sin(points["data"][:, 1] + points["data"][:, 0])*(points["data"][:, 1] + points["data"][:, 0] + 1)
        vals["data"] = data

    return vals