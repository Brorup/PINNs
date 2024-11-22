import jax
import jax.numpy as jnp


def xy2r(x, y):
    return jnp.sqrt(jnp.add(jnp.square(x), jnp.square(y)))


def xy2theta(x, y):
    return jnp.arctan2(y, x)


def rtheta2x(r, theta):
    return jnp.multiply(r, jnp.cos(theta))


def rtheta2y(r, theta):
    return jnp.multiply(r, jnp.sin(theta))


def xy2rtheta(xy: jax.Array) -> jax.Array:
    return jnp.array([xy2r(xy[0], xy[1]), xy2theta(xy[0], xy[1])])


def rtheta2xy(rtheta: jax.Array) -> jax.Array:
    return jnp.array([rtheta[0]*jnp.cos(rtheta[1]), rtheta[0]*jnp.sin(rtheta[1])])


def vxy2rtheta(xy: jax.Array) -> jax.Array:
    return jax.vmap(xy2rtheta)(xy)


def vrtheta2xy(rtheta: jax.Array) -> jax.Array:
    return jax.vmap(rtheta2xy)(rtheta)


def cart2polar_tensor(stresses: jax.Array, xy: jax.Array) -> jax.Array:
    theta = xy2theta(xy[0], xy[1])
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    P = jnp.array([[costheta, sintheta], [-sintheta, costheta]])
    return jnp.matmul(P, jnp.matmul(stresses, P.T))


def polar2cart_tensor(stresses: jax.Array, rt: jax.Array) -> jax.Array:
    costheta = jnp.cos(rt[1])
    sintheta = jnp.sin(rt[1])
    P = jnp.array([[costheta, sintheta], [-sintheta, costheta]])
    return jnp.matmul(P.T, jnp.matmul(stresses, P))


# def cart2polar(sigma_xx: jnp.ndarray,
#                sigma_xy: jnp.ndarray,
#                sigma_yy: jnp.ndarray,
#                theta: jnp.ndarray
#                ):
    
#     costheta = jnp.cos(theta)
#     sintheta = jnp.sin(theta)
#     costheta2 = jnp.square(costheta)
#     sintheta2 = jnp.square(sintheta)
#     sincos = jnp.multiply(costheta, sintheta)

#     sigma_rr = jnp.multiply(sigma_xx, costheta2) + \
#                jnp.multiply(sigma_yy, sintheta2) + \
#                jnp.multiply(sigma_xy, sincos) * 2
    
#     sigma_rt = jnp.multiply(sincos, jnp.subtract(sigma_yy, sigma_xx)) + \
#                jnp.multiply(sigma_xy, jnp.subtract(costheta2, sintheta2))
    
#     sigma_tt = jnp.multiply(sigma_xx, sintheta2) + \
#                jnp.multiply(sigma_yy, costheta2) - \
#                jnp.multiply(sigma_xy, sincos) * 2
    
#     return sigma_rr, sigma_rt, sigma_tt