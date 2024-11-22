from functools import partial, wraps
from typing import override
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from models.derivatives import laplacian, hessian
from models.loss import ms, mse, sq, sqe
from models.networks import netmap, setup_network
import models.platewithhole.loss as pwhloss
from . import PlateWithHolePINN

class DoubleLaplacePINN(PlateWithHolePINN):
    """
    Implementation of a PINN for solving the biharmonic
    equation as a system of two Laplace problems.
    """
    net: nn.Module
    optimizer: optax.GradientTransformation

    def __init__(self, settings: dict):
        super().__init__(settings)
        # Call init_model() which must be implemented and set the params
        self.init_model(settings["model"]["pinn"]["network"])
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")

        # Only use one network
        self.net = self.net[0]
        self.opt_state = self.optimizer.init(self.params)
        return
    
    def forward(self, params, input: jax.Array):
        """
        Function defining the forward pass of the model.
        """
        return self.phi(params, input)
    
    def phi(self, params, input: jax.Array) -> jax.Array:
        return self.net.apply(params["net0"], input)[0]

    def psi(self, params, input: jax.Array) -> jax.Array:
        return self.net.apply(params["net0"], input)[1]
    
    def hessian(self, params, input: jax.Array) -> jax.Array:
        return hessian(self.phi)(params, input)
    
    def loss_coll(self, params, input: jax.Array, true_val: jax.Array | None = None):
        """
        Computes the loss of the PDE residual on the domain.
        """
        
        # Compute Laplacian values for both networks
        lap_phi = netmap(laplacian(self.phi))(params, input)
        lap_psi = netmap(laplacian(self.psi))(params, input)

        # Return both losses
        if true_val is None:
            return self.loss_fn(lap_phi, netmap(self.psi)(params, input)), self.loss_fn(lap_psi)
        return self.loss_fn(lap_phi, netmap(self.psi)(params, input)), self.loss_fn(lap_psi, true_val)

    def resample_eval(self,
                      params,
                      input: jax.Array,
                      true_val: jax.Array | None = None
                      ) -> jax.Array:
        """
        Function for evaluating loss in resampler.
        Should return the loss for each input point, i.e.
        not aggregating them with sum, mean, etc.
        """

        # Compute Laplacian values for both networks
        lap_phi = netmap(laplacian(self.phi))(params, input)
        lap_psi = netmap(laplacian(self.psi))(params, input)

        # Return both losses
        if true_val is None:
            return self.loss_fn(lap_phi, self.psi(params, input)) + self.loss_fn(lap_psi)
        return self.loss_fn(lap_phi, self.psi(params, input)) + self.loss_fn(lap_psi, true_val)