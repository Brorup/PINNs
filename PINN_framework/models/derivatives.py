from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp


def gradient(model: Callable, argnums=1, rev = True) -> Callable:
    if rev:
        return jax.jacrev(model, argnums=argnums)
    return jax.jacfwd(model, argnums=argnums)


def hessian(model: Callable, argnums=1) -> Callable:
    return jax.hessian(model, argnums=argnums)


def laplacian(model: Callable, axis1=-2, axis2=-1, argnums=1) -> Callable:
    hess = jax.hessian(model, argnums=argnums)
    tr = lambda *args: jnp.trace(hess(*args).reshape(2, 2), axis1=axis1, axis2=axis2)
    return tr


def biharmonic(model: Callable, axis1=-2, axis2=-1, argnums=1) -> Callable:
    lap = laplacian(model, axis1=axis1, axis2=axis2, argnums=argnums)
    lap2 = laplacian(lap, axis1=axis1, axis2=axis2, argnums=argnums)
    return lap2


def diff_xx(model) -> Callable:
    def xx(params, input):
        return jax.vmap(jax.hessian(model, argnums=1), in_axes=(None, 0))(params, input)
    return xx


def diff_yy(model) -> Callable:
    def yy(params, input):
        return jax.vmap(hessian(model), in_axes=(None, 0))(params, input)[:, 1, 1]
    return yy


def diff_xy(model) -> Callable:
    def xy(params, input):
        return jax.vmap(hessian(model), in_axes=(None, 0))(params, input)[:, 0, 1]
    return xy


def finite_diff_x(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/2.0)
    
    def fdmx1d(params, input):
        dx = jnp.array([[h, 0]])
        xx = (-0.5*model(params, jnp.subtract(input, dx)) + 0.5*model(params, jnp.add(input, dx))) / h
        return xx
    return fdmx1d


def finite_diff_y(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/2.0)
    
    def fdmy1d(params, input):
        dy = jnp.array([[0, h]])
        xx = (-0.5*model(params, jnp.subtract(input, dy)) + 0.5*model(params, jnp.add(input, dy))) / h
        return xx
    return fdmy1d


def finite_diff_xx(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/3.0)
    
    h2 = h**2
    def fdmx2d(params, input):
        dx = jnp.array([[h, 0]])
        xx = (model(params, jnp.subtract(input, dx)) - 2*model(params, input) + model(params, jnp.add(input, dx))) / h2
        return xx
    return fdmx2d


def finite_diff_yy(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/3.0)
        
    h2 = h**2
    def fdmy2d(params, input):
        dy = jnp.array([[0, h]])
        xx = (model(params, jnp.subtract(input, dy)) - 2*model(params, input) + model(params, jnp.add(input, dy))) / h2
        return xx
    return fdmy2d


def finite_diff_xxx(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/4.0)
    
    h3 = h**3
    def fdmx3d(params, input):
        dx = jnp.array([[h, 0]])
        dx2 = jnp.array([[2*h, 0]])
        xx = (-0.5*model(params, jnp.subtract(input, dx2)) + model(params, jnp.subtract(input, dx)) + model(params, jnp.add(input, dx)) - 0.5*model(params, jnp.add(input, dx2))) / h3
        return xx
    return fdmx3d


def finite_diff_yyy(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/4.0)
        
    h3 = h**3
    def fdmy3d(params, input):
        dy = jnp.array([[0, h]])
        dy2 = jnp.array([[0, 2*h]])
        xx = (-0.5*model(params, jnp.subtract(input, dy2)) + model(params, jnp.subtract(input, dy)) + model(params, jnp.add(input, dy)) - 0.5*model(params, jnp.add(input, dy2))) / h3
        return xx
    return fdmy3d


def finite_diff_xxxx(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/6.0)
    
    h4 = h**4
    def fdmx4d(params, input):
        dx = jnp.array([[h, 0]])
        dx2 = jnp.array([[2*h, 0]])
        xx = (model(params, jnp.subtract(input, dx2)) - 4*model(params, jnp.subtract(input, dx)) + 6*model(params, input) - 4*model(params, jnp.add(input, dx)) + model(params, jnp.add(input, dx2))) / h4
        return xx
    return fdmx4d


def finite_diff_yyyy(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/6.0)
        
    h4 = h**4
    def fdmy4d(params, input):
        dy = jnp.array([[0, h]])
        dy2 = jnp.array([[0, 2*h]])
        xx = (model(params, jnp.subtract(input, dy2)) - 4*model(params, jnp.subtract(input, dy)) + 6*model(params, input) - 4*model(params, jnp.add(input, dy)) + model(params, jnp.add(input, dy2))) / h4
        return xx
    return fdmy4d


def finite_diff_xxyy(model, h=None) -> Callable:
    if h is None:
        h = pow(np.finfo(np.float32).eps, 1.0/6.0)
        
    h4 = h**4
    def fdmxy4d(params, input):
        dx = jnp.array([[h, 0]])
        dy = jnp.array([[0, h]])
        xx = ( model(params, jnp.subtract(jnp.add(input, dy), dx))      - 2*model(params, jnp.add(input, dy))      +   model(params, jnp.add(jnp.add(input, dy), dx)) + \
            -2*model(params, jnp.subtract(input, dx))                   + 4*model(params, input)                   - 2*model(params, jnp.add(input, dx)) + \
               model(params, jnp.subtract(jnp.subtract(input, dy), dx)) - 2*model(params, jnp.subtract(input, dy)) +   model(params, jnp.add(jnp.subtract(input, dy), dx))) / h4
        return xx
    return fdmxy4d


def finite_diff_biharmonic(model, h=None) -> Callable:
    if h is None: 
        h = pow(np.finfo(np.float32).eps, 1.0/5.0)
    
    def fdm(params, input):
        dx = jnp.array([[h, 0]])
        dy = jnp.array([[0, h]])
        dxm = -dx
        dym = -dy
        dx2 = jnp.array([[2*h, 0]])
        dy2 = jnp.array([[0, 2*h]])
        dx2m = -dx2
        dy2m = -dy2
        xx = (                                                                                           model(params, jnp.add(input, dy2)) + \
                                                  2*model(params, jnp.add(input, jnp.add(dy, dxm))) -  8*model(params, jnp.add(input, dy))  + 2*model(params, jnp.add(input, jnp.add(dy, dx))) + \
            model(params, jnp.add(input, dx2m)) - 8*model(params, jnp.add(input, dxm))              + 20*model(params, input)               - 8*model(params, jnp.add(input, dx)) + model(params, jnp.add(input, dx2)) + \
                                                  2*model(params, jnp.add(input, jnp.add(dxm, dym))) - 8*model(params, jnp.add(input, dym)) + 2*model(params, jnp.add(input, jnp.add(dx, dym))) + \
                                                                                                         model(params, jnp.add(input, dy2m)))
        return xx
    return fdm   