from typing import override
import sys

import jax
import jax.numpy as jnp
import optax

from .base import Model
from .networks import setup_network
from models.loss import L2rel, mse, maxabse, ms, sq, sqe
from models.networks import netmap

class NN(Model):
    """
    General NN model:

    MANDATORY methods to add:
    
        self.forward():
            A forward method, i.e. a forward pass through a network.
        
        self.loss_terms():
            A method that evaluates each loss term. These terms can
            be specified in separate methods if desired. This method
            should return a jax.Array of the loss terms.
        
        self.update():
             A method for updating parameters during training.
        
        self.train():
            A method for training the model. Typically calling the
            'update' method in a loop.
        
        self.eval():
            A method for evaluating the model.        

    
    OPTIONAL methods to add:

        self.total_loss():
            A function summing the loss terms. Can
            be rewritten to do something else.
        
        self.save_state():
            A method for saving the model state.
        
        self.load_state():
            A method for loading the model state.
    
    
    """

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        
        # self._parse_geometry_settings(settings["geometry"])
        return

    @override
    def init_model(self, network_settings: list[dict]) -> None:
        
        # Number of networks in model
        num_nets = len(network_settings)

        # Initialize network classes
        self.nets = [setup_network(net) for net in network_settings]
        
        # Initialize network parameters
        self._key, *net_keys = jax.random.split(self._key, num_nets+1)
        params = [net.init(net_keys[i], jnp.ones((net.input_dim,))) for i, net in enumerate(self.nets)]
        self.params = {"net"+str(i): par for i, par in enumerate(params)}

        # Set optimizer if relevant
        if self.do_train:
            self.schedule = optax.exponential_decay(self.train_settings.learning_rate,
                                                    self.train_settings.decay_steps,
                                                    self.train_settings.decay_rate)
            self.optimizer = self.train_settings.optimizer(learning_rate=self.schedule)
            
        return
    
    def predict(self, *args, **kwargs):
        """
        A basic method for the forward pass without inputting
        parameters. For external use.
        """
        return self.forward(self.params, *args, **kwargs)

    def _eval(self, u, u_true, metric: str = "all", verbose = None):
        """
        Evaluates the error using the specified metric.
        """
        if verbose is None:
            verbose = self._verbose.evaluation

        if verbose:
            print("\nEvaluation:\n")

        do_all_metrics = False
        if isinstance(metric, str):
            match metric.lower():
                case "all":
                    do_all_metrics = True
                    def unjitted_L2rel(u, u_true): return jax.vmap(L2rel, in_axes=(0, 0))(u, u_true)
                    def unjitted_mse(u, u_true): return jax.vmap(mse, in_axes=(0, 0))(u, u_true)
                    def unjitted_maxabse(u, u_true): return jax.vmap(maxabse, in_axes=(0, 0))(u, u_true)
                case "l2-rel":
                    def unjitted_metric_fun(u, u_true): return jax.vmap(L2rel, in_axes=(0, 0))(u, u_true)
                    metric_description = "L2 relative error"
                case "l2rel":
                    def unjitted_metric_fun(u, u_true): return jax.vmap(L2rel, in_axes=(0, 0))(u, u_true)
                    metric_description = "L2 relative error"
                case "mse":
                    def unjitted_metric_fun(u, u_true): return jax.vmap(mse, in_axes=(0, 0))(u, u_true)
                    metric_description = "Mean squared error"
                case "maxabse":
                    def unjitted_metric_fun(u, u_true): return jax.vmap(maxabse, in_axes=(0, 0))(u, u_true)
                    metric_description = "Max abs error"
                case _:
                    print(f"Unknown metric: '{metric}'. Default ('L2-rel') is used for evaluation.")
                    def unjitted_metric_fun(u, u_true): return jax.vmap(L2rel, in_axes=(0, 0))(u, u_true)
                    

        if do_all_metrics:
            metric_funs = [jax.jit(unjitted_L2rel), jax.jit(unjitted_mse), jax.jit(unjitted_maxabse)]
            metric_descriptions = ["L2 relative error", "Mean squared error", "Max abs error"]
        else:
            metric_funs = [jax.jit(unjitted_metric_fun)]
            metric_descriptions = [metric_description]
        
        for (metric_fun, metric_description) in zip(metric_funs, metric_descriptions):
            err = metric_fun(u, u_true)

            err_max = jnp.max(err)
            err_mean = jnp.mean(err)
            err_min = jnp.min(err)
            err_25 = jnp.percentile(err, 25)
            err_median = jnp.median(err)
            err_75 = jnp.percentile(err, 75)
            
            
            if verbose:
                print(f"\n\nMin  {metric_description} of model: {err_min:.2g}")
                print(f"Mean {metric_description} of model: {err_mean:.2g}")
                print(f"Max  {metric_description} of model: {err_max:.2g}\n")
                print(f"25th percentile {metric_description} of model: {err_25:.2g}")
                print(f"Median          {metric_description} of model: {err_median:.2g}")
                print(f"75th percentile {metric_description} of model: {err_75:.2g}\n")
        
        if verbose:
            print("\n###############################################################\n\n")
            sys.stdout.flush()

        return err_max