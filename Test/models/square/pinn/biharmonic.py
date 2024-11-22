from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from models.derivatives import biharmonic, hessian
from models.loss import ms, mse, sq, sqe
from models.networks import netmap
import models.square.loss as sqloss
from . import SquarePINN

class BiharmonicPINN(SquarePINN):
    net_bi: nn.Module
    optimizer: optax.GradientTransformation

    def __init__(self, settings: dict):
        super().__init__(settings)
        # Call init_model() which must be implemented and set the params
        self.init_model(settings["model"]["pinn"]["network"])
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")

        # Only use one network
        self.net_bi = self.net[0]
        self.opt_state = self.optimizer.init(self.params)

        return
    
    def forward(self, params, input: jax.Array):
        """
        Function defining the forward pass of the model. The same as phi in this case.
        """
        return self.phi(params, input)
    
    def phi(self, params, input: jax.Array) -> jax.Array:
        return self.net_bi.apply(params["net0"], input)

    def biharmonic(self, params, input: jax.Array) -> jax.Array:
        return biharmonic(self.phi)(params, input)
    
    def loss_coll(self, params, input: jax.Array, true_val: jax.Array | None = None):
        """
        Computes the loss of the PDE residual on the domain.
        """
        
        # Compute biharmonic values
        bi_out = netmap(self.biharmonic)(params, input)

        # Return loss
        if true_val is None:
            return ms(bi_out)
        return mse(bi_out, true_val)

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

        # Compute biharmonic values
        bi_out = netmap(self.biharmonic)(params, input)

        # Return loss
        if true_val is None:
            return sq(bi_out)
        return sqe(bi_out, true_val)