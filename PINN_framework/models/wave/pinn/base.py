from collections.abc import Callable
from typing import override
from functools import partial
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from datahandlers.generators import (
    generate_rectangle,
    generate_collocation_points,
    resample,
    resample_idx,
    JaxDataset
)
from models import PINN
from models.derivatives import laplacian, gradient
from models.networks import netmap
from models.wave import analytic
from models.wave import plotting as wplot
from models.loss import L2rel, mse, maxabse, ms, sq, sqe
from utils.utils import timer

class WavePINN(PINN):
    """
    PINN class.

    The methods in this class are common for all experiments made.
    They include:
    
        self.predict():
            The method use for external calls of the models.
        
        self.sample_points():
            The method used for sampling the points on the
            boundaries and in the domain.
        
        self.resample():
            The method used for resampling.
        
        self.plot_results():
            The method used for plotting the prediction.
    
    The methods are NOT general methods for any PINN (see the PINN class),
    but relates to this specific problem. The inheritance is as follows:
    ```
                    Model   ________________________
                                                    |
                      |                             |
                      V                             V

                    PINN             << Other models, e.g. DeepONet >>

                      |
                      V

                   WavePINN

                      |
                      V
                
        << Specific implementations >>
    
    ```
    """

    net: nn.Module
    optimizer: optax.GradientTransformation

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)

        self.init_model(settings["model"]["pinn"]["network"])
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")

        # Only use one network
        self.net = self.nets[0]
        self.opt_state = self.optimizer.init(self.params)
        
        if self._verbose.init:
            print(f"Initialized network:")
            print(f"\tInput dim: {self.net.input_dim}")
            print(f"\tOutput dim: {self.net.output_dim}")
            print(f"\tHidden layers: {self.net.hidden_dims}")
            print(f"\tActivations: {[i._fun.__name__ for i in self.net.activation]}")
            print(f"\tWeight Initialization: {[i.__name__ for i in self.net.initialization]}")
            print("\n###############################################################\n\n")
            sys.stdout.flush()
        return

    def forward(self, params, input: jax.Array):
        """
        Function defining the forward pass of the model. 
        """
        return self.net.apply(params["net0"], input)
    
    def laplacian(self, params, input: jax.Array):
        # Laplacian w.r.t. spatial coordinates, i.e. x in (x,t) or (x,y) in (x,y,t). Hardcoded to 1 spatial dimension
        model = lambda p, x1, x2: self.forward(p, jnp.stack([x1, x2]))
        return laplacian(model, argnums=1, dims=1)(params, input[0], input[1])
    
    def dtt(self, params, input: jax.Array):
        model = lambda p, x1, x2: self.forward(p, jnp.stack([x1, x2]))
        return laplacian(model, argnums=2, dims=1)(params, input[0], input[1])
    
    def dt(self, params, input: jax.Array):
        model = lambda p, x1, x2: self.forward(p, jnp.stack([x1, x2]))
        return gradient(model, argnums=2)(params, input[0], input[1])

    def diff_yy_m_diff_xx(self, params, input: jax.Array):
        hess = jax.hessian(self.forward, argnums=1)(params, input)
        return hess[:, 1, 1] - hess[:, 0, 0]


    @override
    def predict(self,
                input: jax.Array,
                ) -> jax.Array:
        """
        Function for predicting without inputting parameters.
        For external use.
        """

        output = self.forward(self.params, input)

        return output
    
    def loss_coll(self, params, input: jax.Array, true_val: jax.Array | None = None):
        """
        Computes the loss of the PDE residual on the domain.
        """

        wave = lambda params, input: self.dtt(params, input) - self.laplacian(params, input)
        # Compute laplacian values
        out = netmap(wave)(params, input)
        
        # out = netmap(self.diff_yy_m_diff_xx)(params, input)

        # Return loss
        if true_val is None:
            return ms(out)
        return mse(out, true_val)
    
    def loss_bc(self, params, input: jax.Array, true_val: jax.Array | None = None):
        out = netmap(self.forward)(params, input)

        # Compute loss
        if self.loss_fn is None:
            loss_fn = mse
        else:
            loss_fn = self.loss_fn

        if true_val is None:
            return loss_fn(out)
        return loss_fn(out, true_val)

    def loss_ic0(self, params, input: jax.Array, true_val: jax.Array | None = None):
        return self.loss_bc(params, input, true_val)

    def loss_ic1(self, params, input: jax.Array, true_val: jax.Array | None = None):
        out = netmap(self.dt)(params, input)

        # Compute loss
        if self.loss_fn is None:
            loss_fn = mse
        else:
            loss_fn = self.loss_fn

        if true_val is None:
            return loss_fn(out)
        return loss_fn(out, true_val)
    
    def loss_data(self, params, input: jax.Array, true_val: jax.Array | None = None):
        
        # Compute model output
        out = netmap(self.forward)(params, input)

        # Compute loss
        if self.loss_fn is None:
            loss_fn = mse
        else:
            loss_fn = self.loss_fn

        if true_val is None:
            return loss_fn(out)
        return loss_fn(out, true_val)
    
    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """
        if self._verbose.sampling:
            print("\nSampling:\n")

        self._key, train_key, eval_key, dataset_key = jax.random.split(self._key, 4)
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        tlim = self.geometry_settings["domain"]["rectangle"]["tlim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        # Sampling points in domain and on boundaries
        #TODO implement way to switch between sampling methods
        if self._verbose.sampling:
            print(f"Training on domain:")
            print(f"\tx in {xlim}")
            print(f"\tt in {tlim}")
            print()
            print("Sampling training points using sobol sampling:")
            print(f"\tCollocation points: {train_sampling["coll"]}")
            print(f"\tBoundary points: {[train_sampling["rect"][i] for i in [1, 3]]}")
            print(f"\tInitial points: {train_sampling["rect"][0]}")
            print()

        self.train_points = generate_rectangle(train_key, xlim, tlim,
                                                         train_sampling["coll"],
                                                         train_sampling["rect"])
        self.train_points["bc"] = jnp.concatenate([self.train_points["rect"][i] for i in [1, 3]])
        self.train_points["ic0"] = self.train_points["rect"][0]
        self.train_points["ic1"] = self.train_points["rect"][0]
        self.train_points.pop("rect")

        if self._verbose.sampling:
            print("Sampling evaluation points using sobol sampling:")
            print(f"\tCollocation points: {eval_sampling["coll"]}")
            print(f"\tBoundary points: {[eval_sampling["rect"][i] for i in [1, 3]]}")
            print(f"\tInitial points: {eval_sampling["rect"][0]}")
            print()
        self.eval_points = generate_rectangle(eval_key, xlim, tlim,
                                                        eval_sampling["coll"],
                                                        eval_sampling["rect"])
        self.eval_points["bc"] = jnp.concatenate([self.eval_points["rect"][i] for i in [1, 3]])
        self.eval_points["ic0"] = self.eval_points["rect"][0]
        self.eval_points.pop("rect")
        self.eval_points["data"] = self.eval_points["coll"]
        self.eval_points.pop("coll")

        # Generate data points
        self._key, data_train_key = jax.random.split(self._key, 2)
        self.train_points["data"] = generate_collocation_points(data_train_key,
                                                                          xlim, tlim, train_sampling.get("data"))

        # Get corresponding function values
        self.train_true_val = analytic.get_true_vals(self.train_points)
        self.eval_true_val = analytic.get_true_vals(self.eval_points)
        
        self.full_batch_dataset = JaxDataset(key=dataset_key, xy=self.train_points["coll"], u=self.train_true_val["coll"], batch_size=self.train_settings.batch_size)
        
        self.train_points_batch = self.train_points.copy()
        self.train_true_val_batch = self.train_true_val.copy()
        
        if self.sample_plots.do_plots:
            self.plot_training_points()    
        
        if self._verbose.sampling:
            print("###############################################################\n\n")
            sys.stdout.flush()
        return
    
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
        out = netmap(self.laplacian)(params, input)

        # Return loss
        if true_val is None:
            return sq(out)
        return sqe(out, true_val)

    def eval(self, point_type: str = "all", metric: str  = "all", verbose = None, **kwargs):
        """
        Evaluates the error using the specified metric.
        """

        if point_type == "all":
            #TODO maybe make sure dictionarires are sorted the same way
            points = jnp.concatenate(list(self.eval_points.values()))
            u_true = jnp.concatenate(list(self.eval_true_val.values()))
        else:
            points = self.eval_points[point_type]
            u_true = self.eval_true_val[point_type]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
        err = self._eval(u, u_true, metric, verbose)
        
        return err

    def plot_results(self, save=True, log=False, step=None):
        wplot.plot_results(self.geometry_settings, self.forward, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])
        
    def do_every(self, epoch: int | None = None, loss_term_fun: Callable | None = None, **kwargs):
        
        # plot_every = self.result_plots.plot_every
        # log_every = self.logging.log_every
        # do_log = self.logging.do_logging
        checkpoint_every = self.train_settings.checkpoint_every

        # if do_log and epoch % log_every == log_every-1:
        #     loss_terms = loss_term_fun(**kwargs)
        #     if epoch // log_every == 0:
        #         self.all_losses = jnp.zeros((0, loss_terms.shape[0]))
        #     self.all_losses = self.log_scalars(loss_terms, self.loss_names, all_losses=self.all_losses, log=False)

        # if plot_every and epoch % plot_every == plot_every-1:
        #     self.plot_results(save=False, log=True, step=epoch)

        if epoch % checkpoint_every == checkpoint_every-1:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_losses"):
                with open(self.dir.log_dir.joinpath('all_losses.npy'), "wb") as f:
                    jnp.save(f, self.all_losses)
        
        return