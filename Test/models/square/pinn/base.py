from collections.abc import Callable
from typing import override
from functools import partial

import jax
import jax.numpy as jnp

from datahandlers.generators import (
    generate_rectangle,
    generate_collocation_points,
    resample,
    resample_idx,
    JaxDataset
)
from models import PINN
from models.derivatives import hessian
from models.networks import netmap
from models.square import analytic
from models.square import loss as sqloss
from models.square import plotting as sqplot
from models.loss import L2rel, mse, maxabse
from utils.utils import timer

class SquarePINN(PINN):
    """
    PINN class specifically for solving the square problem.

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
            The method used for plotting the potential and
            the cartesian/polar stresses.
    
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

                  SquarePINN

                      |
                      V
                
        << Specific implementations >>
    
    ```
    """

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        return
    
    def hessian(self, params, input: jax.Array) -> jax.Array:
        return hessian(self.forward)(params, input)
    
    @partial(jax.jit, static_argnums=(0,))
    def jitted_hessian(self, params, input: jax.Array) -> jax.Array:
        return self.hessian(params, input)

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

    def loss_rect(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):
        """
        Computes the loss of the BC residuals on the four sides of the rectangle.
        """
        """
        Layout of rectangle sides:

                            ^
                          y |
                            |
                            *------>
                                 x
                                
                                        2
                                _________________
               xx stress  <-   |                 |   -> xx stress
                               |                 |
                            3  |        O        |  1
                               |                 |
               xx stress  <-   |_________________|   -> xx stress
                                
                                        0
        """

        # Compute Hessian values for each of the four sides
        out0 = netmap(self.hessian)(params, input[0]).reshape(-1, 4) # horizontal lower
        out1 = netmap(self.hessian)(params, input[1]).reshape(-1, 4) # vertical right
        out2 = netmap(self.hessian)(params, input[2]).reshape(-1, 4) # horizontal upper
        out3 = netmap(self.hessian)(params, input[3]).reshape(-1, 4) # vertical left

        # Compute losses for all four sides of rectangle
        losses  = sqloss.loss_rect(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def loss_diri(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):

        # Compute potential output
        out0 = netmap(self.forward)(params, input[0])
        out1 = netmap(self.forward)(params, input[1])
        out2 = netmap(self.forward)(params, input[2])
        out3 = netmap(self.forward)(params, input[3])

        # Compute losses for all four sides of rectangle
        losses  = sqloss.loss_dirichlet(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_data(self, params, input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        
        # Compute model output
        out = netmap(self.hessian)(params, input).reshape(-1, 4)

        # Compute losses
        losses = sqloss.loss_data(out, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_rect_extra(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):
        # Compute Hessian values for each of the four sides
        out0 = netmap(self.hessian)(params, input[0]).reshape(-1, 4) # horizontal lower
        out1 = netmap(self.hessian)(params, input[1]).reshape(-1, 4) # vertical right
        out2 = netmap(self.hessian)(params, input[2]).reshape(-1, 4) # horizontal upper
        out3 = netmap(self.hessian)(params, input[3]).reshape(-1, 4) # vertical left

        # Compute losses for all four sides of rectangle
        losses  = sqloss.loss_rect_extra(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_key, eval_key, dataset_key = jax.random.split(self._key, 4)
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        # Sampling points in domain and on boundaries
        self.train_points = generate_rectangle(train_key, xlim, ylim,
                                                         train_sampling["coll"],
                                                         train_sampling["rect"])
        self.eval_points = generate_rectangle(eval_key, xlim, ylim,
                                                        eval_sampling["coll"],
                                                        eval_sampling["rect"])
        
        # Generate data points
        self._key, data_train_key, data_eval_key = jax.random.split(self._key, 3)
        self.train_points["data"] = generate_collocation_points(data_train_key,
                                                                          xlim, ylim, train_sampling.get("data"))
        self.eval_points["data"] = generate_collocation_points(data_eval_key,
                                                                         xlim, ylim, eval_sampling.get("data"))

        # Get corresponding function values
        self.train_true_val = analytic.get_true_vals(self.train_points)
        self.eval_true_val = analytic.get_true_vals(self.eval_points)
        
        self.full_batch_dataset = JaxDataset(key=dataset_key, xy=self.train_points["coll"], u=self.train_true_val["coll"], batch_size=self.train_settings.batch_size)
        
        self.train_points_batch = self.train_points.copy()
        self.train_true_val_batch = self.train_true_val.copy()
        
        if self.sample_plots.do_plots:
            self.plot_training_points()    
            
        return

    def eval(self, point_type: str = "coll", metric: str  = "L2-rel", **kwargs):
        """
        Evaluates the Cartesian stresses using the specified metric.
        """
        match metric.lower():
            case "l2-rel":
                metric_fun = jax.jit(L2rel)
            case "l2rel":
                metric_fun = jax.jit(L2rel)
            case "mse":
                metric_fun = jax.jit(mse)
            case "maxabse":
                metric_fun = jax.jit(maxabse)
            case _:
                print(f"Unknown metric: '{metric}'. Default ('L2-rel') is used for evaluation.")
                metric_fun = jax.jit(L2rel)

        u = jnp.squeeze(netmap(self.jitted_hessian)(self.params, self.eval_points[point_type]))
        u_true = jax.vmap(partial(analytic.cart_stress_true, **kwargs))(self.eval_points[point_type])

        err = jnp.array([[metric_fun(u[:, i, j], u_true[:, i, j]) for i in range(2)] for j in range(2)])

        attr_name = "eval_result"

        if hasattr(self, attr_name):
            if isinstance(self.eval_result, dict):
                self.eval_result[metric] = err
            else:
                raise TypeError(f"Attribute '{attr_name}' is not a dictionary. "
                                f"Evaluation error cannot be added.")
        else:
            self.eval_result = {metric: err}
        
        return err

    def plot_results(self, save=True, log=False, step=None):
        sqplot.plot_results(self.geometry_settings, self.forward, self.jitted_hessian, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])
        
        
    def plot_boundaries(self, save=True, log=False, step=None):
        sqplot.plot_boundaries(self.geometry_settings, self.jitted_hessian, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])
        
    def do_every(self, epoch: int | None = None, loss_term_fun: Callable | None = None, **kwargs):
        
        plot_every = self.result_plots.plot_every
        log_every = self.logging.log_every
        do_log = self.logging.do_logging
        checkpoint_every = self.train_settings.checkpoint_every

        if do_log and epoch % log_every == log_every-1:
            loss_terms = loss_term_fun(**kwargs)
            if epoch // log_every == 0:
                self.all_losses = jnp.zeros((0, loss_terms.shape[0]))
            self.all_losses = self.log_scalars(loss_terms, self.loss_names, all_losses=self.all_losses, log=False)

        if plot_every and epoch % plot_every == plot_every-1:
            self.plot_results(save=False, log=True, step=epoch)

        if epoch % checkpoint_every == checkpoint_every-1:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_losses"):
                with open(self.dir.log_dir.joinpath('all_losses.npy'), "wb") as f:
                    jnp.save(f, self.all_losses)
        
        return