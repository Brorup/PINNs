from collections.abc import Callable
from typing import override
from functools import partial

import jax
import jax.numpy as jnp

from datahandlers.generators import (
    generate_rectangle_with_hole,
    generate_collocation_points_with_hole,
    generate_extra_points,
    resample,
    resample_idx,
    JaxDataset
)
from models import PINN
from models.derivatives import hessian
from models.networks import netmap
from models.platewithhole import analytic
from models.platewithhole import loss as pwhloss
from models.platewithhole import plotting as pwhplot
from models.loss import L2rel, mse, maxabse
from utils.transforms import (
    vrtheta2xy,
    vxy2rtheta
)
from utils.utils import timer

class PlateWithHolePINN(PINN):
    """
    PINN class specifically for solving the plate-with-hole problem.

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

               PlateWithHolePINN

                      |
                      V
                
        << Specific implementations >>
    
    ```
    """

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self._register_static_loss_arg("update_key")
        self.forward_input_coords = "cartesian"
        self.forward_output_coords = "cartesian"
        return
    
    def hessian(self, params, input: jax.Array) -> jax.Array:
        return hessian(self.forward)(params, input)
    
    @partial(jax.jit, static_argnums=(0,))
    def jitted_hessian(self, params, input: jax.Array) -> jax.Array:
        return self.hessian(params, input)

    @override
    def predict(self,
                input: jax.Array,
                in_coords: str = "cartesian",
                out_coords: str = "cartesian"
                ) -> jax.Array:
        """
        Function for predicting without inputting parameters.
        For external use.
        """
        in_forward = self.forward_input_coords.lower()
        out_forward = self.forward_output_coords.lower()
        in_coords = in_coords.lower()
        out_coords = out_coords.lower()

        if in_coords == in_forward:
            output = self.forward(self.params, input)
        elif in_coords == "cartesian" and in_forward == "polar":
            output = self.forward(self.params, vxy2rtheta(input))
        elif in_coords == "polar" and in_forward == "cartesian":
            output = self.forward(self.params, vrtheta2xy(input))
        else:
            raise NotImplementedError("Unknown type of coordinates.")

        if out_coords == out_forward:
            return output
        elif out_coords == "cartesian" and out_forward == "polar":
            return vrtheta2xy(output)
        elif out_coords == "polar" and out_forward == "cartesian":
            return vxy2rtheta(output)
        else:
            raise NotImplementedError("Unknown type of coordinates.")

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
        losses  = pwhloss.loss_rect(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def loss_circ(self, params, input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        """
        Computes the loss of the residuals on the circle.
        """

        # Compute cartesian output
        output = netmap(self.hessian)(params, input).reshape(-1, 4)

        # Compute polar stresses and return loss
        losses = pwhloss.loss_circ_rr_rt(input, output, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_circ_extra(self, params, input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        """
        Extra loss on the circle for the theta-theta stress.
        """

        # Compute cartesian output
        output = netmap(self.hessian)(params, input).reshape(-1, 4)
        
        # Compute polar stresses and return loss
        losses = pwhloss.loss_circ_tt(input, output, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def loss_diri(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):

        # Compute potential output
        out0 = netmap(self.forward)(params, input[0])
        out1 = netmap(self.forward)(params, input[1])
        out2 = netmap(self.forward)(params, input[2])
        out3 = netmap(self.forward)(params, input[3])

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_dirichlet(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_data(self, params, input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        
        # Compute model output
        out = netmap(self.hessian)(params, input).reshape(-1, 4)

        # Compute losses
        losses = pwhloss.loss_data(out, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_rect_extra(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):
        # Compute Hessian values for each of the four sides
        out0 = netmap(self.hessian)(params, input[0]).reshape(-1, 4) # horizontal lower
        out1 = netmap(self.hessian)(params, input[1]).reshape(-1, 4) # vertical right
        out2 = netmap(self.hessian)(params, input[2]).reshape(-1, 4) # horizontal upper
        out3 = netmap(self.hessian)(params, input[3]).reshape(-1, 4) # vertical left

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_rect_extra(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_key, eval_key, dataset_key = jax.random.split(self._key, 4)
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        dt = self.geometry_settings["domain"]["type"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        self.define_plot_geometry()

        # Sampling points in domain and on boundaries
        self.train_points = generate_rectangle_with_hole(train_key, radius, xlim, ylim,
                                                        train_sampling["coll"],
                                                        train_sampling["rect"],
                                                        train_sampling["circ"],
                                                        domain_type=dt)
        self.eval_points = generate_rectangle_with_hole(eval_key, radius, xlim, ylim,
                                                        eval_sampling["coll"],
                                                        eval_sampling["rect"],
                                                        eval_sampling["circ"],
                                                        domain_type=dt)
        
        # Keys for data points
        self._key, data_train_key, data_eval_key = jax.random.split(self._key, 3)
        self.train_points["data"] = generate_collocation_points_with_hole(data_train_key, radius,
                                                                          xlim, ylim, train_sampling.get("data"),
                                                                          domain_type=dt)
        self.eval_points["data"] = generate_collocation_points_with_hole(data_eval_key, radius,
                                                                         xlim, ylim, eval_sampling.get("data"),
                                                                         domain_type=dt)

        # Get corresponding function values
        self.train_true_val = analytic.get_true_vals(self.train_points, ylim=ylim)
        self.eval_true_val = analytic.get_true_vals(self.eval_points, ylim=ylim)


        if train_sampling.get("separate_coll"):

            # Find splitting point (number of standard samples vs. extra samples)
            train_coll_points = c[0] if isinstance(c:=train_sampling["coll"], list) else c
            train_data_points = c[0] if isinstance(c:=train_sampling["data"], list) else c

            # Split into seperate dictionary entries
            self.train_points["coll_extra"] = self.train_points["coll"][ train_coll_points:]
            self.train_points["coll"]       = self.train_points["coll"][:train_coll_points ]
            # self.train_points["data_extra"] = self.train_points["data"][:,  train_data_points:]
            # self.train_points["data"]       = self.train_points["data"][:, :train_data_points ]

            # Split into seperate dictionary entries
            self.train_true_val["coll_extra"] = None if (c:=self.train_true_val["coll"]) is None else c[ train_coll_points:]
            self.train_true_val["coll"]       = None if (c:=self.train_true_val["coll"]) is None else c[:train_coll_points ]
            # self.train_true_val["data_extra"] = None if (c:=self.train_true_val["data"]) is None else c[:,  train_data_points:]
            # self.train_true_val["data"]       = None if (c:=self.train_true_val["data"]) is None else c[:, :train_data_points ]
            

        if eval_sampling.get("separate_coll"):
            
            # Find splitting point (number of standard samples vs. extra samples)
            eval_coll_points  = c[0] if isinstance(c:=eval_sampling["coll"], list) else c
            eval_data_points  = c[0] if isinstance(c:=eval_sampling["data"], list) else c

            # Split into seperate dictionary entries
            self.eval_points["coll_extra"] = self.eval_points["coll"][ eval_coll_points:]
            self.eval_points["coll"]       = self.eval_points["coll"][:eval_coll_points ]
            # self.eval_points["data_extra"] = self.eval_points["data"][:,  eval_data_points:]
            # self.eval_points["data"]       = self.eval_points["data"][:, :eval_data_points ]

            # Split into seperate dictionary entries
            self.eval_true_val["coll_extra"] = None if (c:=self.eval_true_val["coll"]) is None else c[ eval_coll_points:]
            self.eval_true_val["coll"]       = None if (c:=self.eval_true_val["coll"]) is None else c[:eval_coll_points ]
            # self.eval_true_val["data_extra"] = None if (c:=self.eval_true_val["data"]) is None else c[:,  eval_coll_points:]
            # self.eval_true_val["data"]       = None if (c:=self.eval_true_val["data"]) is None else c[:, :eval_coll_points ]
                
        self.full_batch_dataset = JaxDataset(key=dataset_key, xy=self.train_points["coll"], u=self.train_true_val["coll"], batch_size=self.train_settings.batch_size)
        
        self.train_points_batch = self.train_points.copy()
        self.train_true_val_batch = self.train_true_val.copy()
        
        if self.sample_plots.do_plots:
            self.plot_training_points()
            
        return
    
    def sample_eval_points(self):
        self._key, train_key, eval_key, dataset_key = jax.random.split(self._key, 4)
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        dt = self.geometry_settings["domain"]["type"]
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        self.define_plot_geometry()

        self.eval_points = generate_rectangle_with_hole(eval_key, radius, xlim, ylim,
                                                        [5000, 200],
                                                        [50, 50, 50, 50],
                                                        200,
                                                        domain_type=dt)
        
        # Keys for data points
        self._key, data_train_key, data_eval_key = jax.random.split(self._key, 3)

        self.eval_points["data"] = generate_collocation_points_with_hole(data_eval_key, radius,
                                                                         xlim, ylim, eval_sampling.get("data"),
                                                                         domain_type=dt)

        # Get corresponding function values
        self.eval_true_val = analytic.get_true_vals(self.eval_points, ylim=ylim)

        return

    def define_plot_geometry(self):
        dt = self.geometry_settings["domain"]["type"].lower()

        self.geometry_settings_plotting = self.geometry_settings.copy()

        match dt:
            case "full":
                pass

            case "half-upper":
                self.geometry_settings_plotting["domain"]["rectangle"]["ylim"][0] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [0, jnp.pi]

            case "half-lower":
                self.geometry_settings_plotting["domain"]["rectangle"]["ylim"][1] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [jnp.pi, 2*jnp.pi]

            case "half-left":
                self.geometry_settings_plotting["domain"]["rectangle"]["xlim"][1] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [0.5*jnp.pi, 1.5*jnp.pi]

            case "half-right":
                self.geometry_settings_plotting["domain"]["rectangle"]["xlim"][0] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [-0.5*jnp.pi, 0.5*jnp.pi]
            
            case "quarter-1":
                self.geometry_settings_plotting["domain"]["rectangle"]["xlim"][0] = 0.0
                self.geometry_settings_plotting["domain"]["rectangle"]["ylim"][0] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [0, 0.5*jnp.pi]
            
            case "quarter-2":
                self.geometry_settings_plotting["domain"]["rectangle"]["xlim"][1] = 0.0
                self.geometry_settings_plotting["domain"]["rectangle"]["ylim"][0] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [0.5*jnp.pi, jnp.pi]

            case "quarter-3":
                self.geometry_settings_plotting["domain"]["rectangle"]["xlim"][1] = 0.0
                self.geometry_settings_plotting["domain"]["rectangle"]["ylim"][1] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [jnp.pi, 1.5*jnp.pi]
            
            case "quarter-4":
                self.geometry_settings_plotting["domain"]["rectangle"]["xlim"][0] = 0.0
                self.geometry_settings_plotting["domain"]["rectangle"]["ylim"][1] = 0.0
                self.geometry_settings_plotting["domain"]["circle"]["angle"] = [-0.5*jnp.pi, 0]
            
            case _:
                raise ValueError(f"Unknown domain type: '{dt}'.")
        

        return


    def resample(self, loss_fun: Callable):
        """
        Method for resampling training points

        input:
            loss_fun(params, inputs, true_val):
                A function for calculating the loss of each point.
                This could be the square of some residual, for example.
        """
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        self._key, resample_key, perm_key = jax.random.split(self._key, 3)
        
        # Get resample parameters
        loss_emphasis = self.train_settings.resampling["loss_emphasis"]
        resample_num = self.train_settings.resampling["resample_num"]
        if isinstance(resample_num, int):
            resample_num = [resample_num]
        
        coll_num = sum(self.train_settings.sampling["coll"])

        print(f"Resampling {sum(resample_num)} out of {coll_num} points with loss emphasis = {loss_emphasis} ...")

        resample_points = [int(loss_emphasis*r) for r in resample_num]

        # Generate points and choose the ones with highest loss
        new_coll = generate_collocation_points_with_hole(resample_key, radius, xlim, ylim, resample_points)
        
        # Get true values of new collocation points
        new_true = analytic.get_true_vals({"coll": new_coll}, exclude=["rect", "circ", "diri", "data"])["coll"]
        
        # Calculate values
        new_loss = loss_fun(self.params, new_coll, true_val=new_true)
        
        # Choose subset of sampled points to keep
        new_coll = resample(new_coll, new_loss, sum(resample_num))

        # Find indices to swap out new training points with
        old_coll = self.train_points["coll"]
        old_true = self.train_true_val["coll"]
        old_loss = loss_fun(self.params, old_coll, true_val=old_true)
        replace_idx = resample_idx(old_coll, old_loss, sum(resample_num))

        # Set new training points
        self.train_points["coll"] = self.train_points["coll"].at[replace_idx].set(new_coll)
        
        # Recalculate true values for collocation points
        self.train_true_val["coll"] = analytic.get_true_vals(self.train_points, exclude=["rect", "circ", "diri", "data"])["coll"]
        return
    
    def eval(self, cartesian: bool = True, point_type: str = "coll", metric: str  = "L2-rel", **kwargs):
        """
        Evaluates the Cartesian stresses using the specified metric.
        """
        def phi2sigma(u):
            return jnp.array([[u[1, 1], -u[1, 0]], [-u[0, 1], u[0, 0]]])

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
        u = jax.vmap(phi2sigma)(u)
        
        if cartesian:
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
        else:
            vm_stress_true = jax.vmap(partial(analytic.von_mises_stress_true, **kwargs))(self.eval_points[point_type])
            vm_stress = jnp.sqrt(u[:, 0, 0]**2 + u[:, 1, 1]**2 - u[:, 0, 0]*u[:, 1, 1] + 3*u[:, 0, 1]**2)

            err = metric_fun(vm_stress, vm_stress_true)
        
        return err
    
    def plot_results(self, save=True, log=False, step=None):
        pwhplot.plot_results(self.geometry_settings_plotting, self.jitted_hessian, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])
        
        
    def plot_boundaries(self, save=True, log=False, step=None):
        pwhplot.plot_boundaries(self.geometry_settings_plotting, self.jitted_hessian, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])

    def plot_true_sol(self, save=True, log=False, step=None):
        pwhplot.plot_true_sol(self.geometry_settings_plotting, self.jitted_hessian, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])
        
        
    def do_every(self, epoch: int | None = None, loss_term_fun: Callable | None = None, **kwargs):
        
        max_epochs = self.train_settings.iterations
        plot_every = self.result_plots.plot_every
        log_every = self.logging.log_every
        do_log = self.logging.do_logging
        sample_every = self.train_settings.resampling["resample_steps"]
        do_resample = self.train_settings.resampling["do_resampling"]
        checkpoint_every = self.train_settings.checkpoint_every

        if do_log and epoch % log_every == log_every-1:
            loss_terms = loss_term_fun(**kwargs)
            if epoch // log_every == 0:
                self.all_losses = jnp.zeros((0, loss_terms.shape[0]))
            self.all_losses = self.log_scalars(loss_terms, self.loss_names, all_losses=self.all_losses, log=False)

        if plot_every and epoch % plot_every == plot_every-1:
            self.plot_results(save=False, log=True, step=epoch)

        if do_resample:
            if (epoch % sample_every == (sample_every-1)):
                if epoch < (max_epochs-1):
                    self.resample(self.resample_eval)

        if epoch % checkpoint_every == checkpoint_every-1:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_losses"):
                with open(self.dir.log_dir.joinpath('all_losses.npy'), "wb") as f:
                    jnp.save(f, self.all_losses)
        
        return