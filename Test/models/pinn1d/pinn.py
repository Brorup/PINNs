from time import perf_counter
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from datahandlers.generators import generate_interval_points
from models.derivatives import gradient, hessian
from models.loss import mse, maxabse, L2rel
from models.networks import netmap
from models.pinn import PINN
from setup.parsers import parse_arguments
from utils.plotting import save_fig

from matplotlib import rc
rc("text", usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')


class PINN1D(PINN):
    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self.type = kwargs.get("type","").lower()
        self._register_static_loss_arg("update_key")
        self.init_model(settings["model"]["pinn"]["network"])
        self._set_loss(loss_term_fun_name="loss_terms")
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")
        self.network = self.net[0]
        self.opt_state = self.optimizer.init(self.params)
        return

    def forward(self, params, input: jax.Array) -> jax.Array:
        return self.network.apply(params["net0"], input)
    
    def grad1(self, params, input: jax.Array) -> jax.Array:
        return gradient(self.forward)(params, input)

    def grad2(self, params, input: jax.Array) -> jax.Array:
        return hessian(self.forward)(params, input)
    
    def grad3(self, params, input: jax.Array) -> jax.Array:
        return gradient(hessian(self.forward))(params, input)
    
    def grad4(self, params, input: jax.Array) -> jax.Array:
        return hessian(hessian(self.forward))(params, input)

    def loss_terms(self,
                   params,
                   inputs: dict[str, jax.Array],
                   true_val: dict[str, jax.Array],
                   update_key: int | None = None,
                   loss_fn: Callable | None = None,
                   **kwargs
                   ) -> jax.Array:
        
        if update_key == 0:
            loss = self.loss0(params, inputs["coll"], true_val=true_val["0"], loss_fn=loss_fn)
            return jnp.array((loss,))

        if update_key == 1:
            loss1 = self.loss1(params, inputs["coll"], true_val=true_val["1"], loss_fn=loss_fn)
            bcloss0 = self.loss0(params, inputs["bc"], true_val=true_val["bc0"], loss_fn=loss_fn)
            return jnp.array((loss1, bcloss0))
        
        if update_key == 2:
            loss2 = self.loss2(params, inputs["coll"], true_val=true_val["2"], loss_fn=loss_fn)
            bcloss1 = self.loss1(params, inputs["bc"], true_val=true_val["bc1"], loss_fn=loss_fn)
            bcloss0 = self.loss0(params, inputs["bc"], true_val=true_val["bc0"], loss_fn=loss_fn)
            return jnp.array((loss2, bcloss1, bcloss0))
        
        if update_key == 3:
            loss3 = self.loss3(params, inputs["coll"], true_val=true_val["3"], loss_fn=loss_fn)
            bcloss2 = self.loss2(params, inputs["bc"], true_val=true_val["bc2"], loss_fn=loss_fn)
            bcloss1 = self.loss1(params, inputs["bc"], true_val=true_val["bc1"], loss_fn=loss_fn)
            bcloss0 = self.loss0(params, inputs["bc"], true_val=true_val["bc0"], loss_fn=loss_fn)
            return jnp.array((loss3, bcloss2, bcloss1, bcloss0))
        
        if update_key == 4:
            loss4 = self.loss4(params, inputs["coll"], true_val=true_val["4"], loss_fn=loss_fn)
            bcloss3 = self.loss3(params, inputs["bc"], true_val=true_val["bc3"], loss_fn=loss_fn)
            bcloss2 = self.loss2(params, inputs["bc"], true_val=true_val["bc2"], loss_fn=loss_fn)
            bcloss1 = self.loss1(params, inputs["bc"], true_val=true_val["bc1"], loss_fn=loss_fn)
            bcloss0 = self.loss0(params, inputs["bc"], true_val=true_val["bc0"], loss_fn=loss_fn)
            return jnp.array((loss4, bcloss3, bcloss2, bcloss1, bcloss0))
        
        if update_key == 5:
            loss4 = self.loss4(params, inputs["coll"], true_val=true_val["4"], loss_fn=loss_fn)
            bcloss2 = self.loss2(params, inputs["bc"], true_val=true_val["bc2"], loss_fn=loss_fn)
            return jnp.array((loss4, bcloss2))

        return 
    
    def loss0(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out0 = netmap(self.forward)(params, input)
        if loss_fn is not None:
            return loss_fn(out0, true_val)
        
        return mse(out0, true_val)
    
    def loss1(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out1 = netmap(self.grad1)(params, input)
        if loss_fn is not None:
            return loss_fn(out1, true_val)
        
        return mse(out1, true_val)
    
    def loss2(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out2 = netmap(self.grad2)(params, input)
        if loss_fn is not None:
            return loss_fn(out2, true_val)
        
        return mse(out2, u_true=true_val)
    
    def loss3(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out3 = netmap(self.grad3)(params, input)
        if loss_fn is not None:
            return loss_fn(out3, true_val)
        
        return mse(out3, true_val)
    
    def loss4(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out4 = netmap(self.grad4)(params, input)
        
        if loss_fn is not None:
            return loss_fn(out4, true_val)
        
        return mse(out4, true_val)
    
        
    def get_true_vals(self, x, true_vals = None, prefix = None):
        if prefix is None:                
            prefix = ""
        
        if true_vals is None:
            true_vals = {}
        
        if self.type == "sin":
            func = lambda x: jnp.sin(x)
        elif self.type == "expsin":
            func = lambda x: jnp.exp(jnp.sin(x))
        elif self.type == "sinxx":
            func = lambda x: jnp.sin(x)*x
        else: 
            func = lambda x: jnp.cos(x)*jnp.sin(x)
        
        true_vals[prefix + "0"] = func(x.ravel())
        true_vals[prefix + "1"] = jax.vmap(jax.grad(func))(x.ravel())
        true_vals[prefix + "2"] = jax.vmap(jax.grad(jax.grad(func)))(x.ravel())
        true_vals[prefix + "3"] = jax.vmap(jax.grad(jax.grad(jax.grad(func))))(x.ravel())
        true_vals[prefix + "4"] = jax.vmap(jax.grad(jax.grad(jax.grad(jax.grad(func)))))(x.ravel())
        
        return true_vals
    
    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_key, eval_key = jax.random.split(self._key, 3)
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        # Sampling points in domain and on boundaries
        self.train_points = {}
        self.eval_points = {}

        self.train_points["coll"] = generate_interval_points(train_key, xlim, train_sampling["coll"], sobol=True)
        self.eval_points["coll"] = generate_interval_points(eval_key, xlim, eval_sampling["coll"], sobol=True)
        self.train_points["bc"] = jnp.array(xlim).reshape(-1, 1)
        self.eval_points["bc"] = jnp.array(xlim).reshape(-1, 1)

        # Get corresponding function values
        self.train_true_val = self.get_true_vals(self.train_points["coll"])
        self.eval_true_val = self.get_true_vals(self.eval_points["coll"])

        self.train_true_val = self.get_true_vals(self.train_points["bc"], self.train_true_val, prefix = "bc")
        self.eval_true_val = self.get_true_vals(self.eval_points["bc"], self.eval_true_val, prefix = "bc")

        return
    
    def train(self, update_key: int | None = None, early_stop_tol = 0):
        if not self.do_train:
            print("Model is not set to train")
            return
        
        max_epochs = self.train_settings.iterations
        log_every = self.logging.log_every
                
        jitted_loss = jax.jit(self.loss_terms, static_argnames=("update_key", "loss_fn"))
        
        # Create arrays for losses for function values and 1st-4th order gradients
        loss_shape = jitted_loss.eval_shape(self.params, self.train_points, self.train_true_val, update_key).shape[0]
        self.loss_log_train = np.zeros((max_epochs // log_every + 1, loss_shape))
        self.loss_log_eval = np.zeros((max_epochs // log_every + 1, loss_shape))
        self.loss_log_epochs = np.arange(0, max_epochs+log_every, log_every)
        # Loss counter
        l = 0

        
        # Start time
        t0 = perf_counter()
        for epoch in range(max_epochs):
            
            self.get_weights(epoch, 
                             loss_term_fun=jitted_loss, 
                             params=self.params, 
                             inputs=self.train_points, 
                             true_val=self.train_true_val, 
                             update_key=update_key)
            
            # Update step
            self.params, self.opt_state, total_loss, loss_terms = self.update(opt_state=self.opt_state,
                                                                                             params=self.params,
                                                                                             inputs=self.train_points,
                                                                                             weights=self.weights,
                                                                                             true_val=self.train_true_val,
                                                                                             update_key=update_key,
                                                                                             start_time=t0,
                                                                                             epoch=epoch,
                                                                                             learning_rate=self.schedule(epoch)
                                                                                             )
            
            
            if (epoch % log_every == 0):
                self.loss_log_train[l] = jitted_loss(self.params, self.train_points, true_val=self.train_true_val, update_key=update_key)
                self.loss_log_eval[l] = jitted_loss(self.params, self.eval_points, true_val=self.eval_true_val, update_key=update_key, loss_fn=maxabse)
                l += 1
                self.eval()
                if update_key == 5:
                    if self.eval_result["maxabse"]["2"] < early_stop_tol:
                        print(f"Early stopping at {epoch} epochs due to the max abs error being below {early_stop_tol}\n")
                        return epoch
                else: 
                    if self.eval_result["maxabse"]["0"] < early_stop_tol:
                        print(f"Early stopping at {epoch} epochs due to the max abs error being below {early_stop_tol}\n")
                        return epoch
                    
        
        # Log latest model loss
        self.loss_log_train[-1] = jitted_loss(self.params, self.train_points, true_val=self.train_true_val, update_key=update_key)
        self.loss_log_eval[-1] = jitted_loss(self.params, self.eval_points, true_val=self.eval_true_val, update_key=update_key, loss_fn=maxabse)
        
        return -1

    def plot_results(self, update_key=None):
        self.plot_losses(update_key)
        if update_key == 0:
            self.plot_derivatives()
        elif update_key in [1, 2, 3, 4]:
            self.plot_u()
            self.plot_derivatives()
        elif update_key == 5:
            self.plot_u2()
            self.plot_derivatives()

    def plot_losses(self, update_key=None):
        if update_key == 5:
            offset = 0
        else:
            offset = 2
        
        fig = plt.figure()
        plt.semilogy(self.loss_log_epochs, self.loss_log_train)
        plt.legend(["Coll"] + ["BC" + str(i) for i in range(self.loss_log_train.shape[1] - offset, -1, -1)])
        save_fig(self.dir.figure_dir, "train.pdf", format="pdf", fig=fig)
        
        fig = plt.figure()
        plt.semilogy(self.loss_log_epochs, self.loss_log_eval)
        plt.legend(["Coll"] + ["BC" + str(i) for i in range(self.loss_log_train.shape[1] - offset, -1, -1)])
        save_fig(self.dir.figure_dir, "eval.pdf", format="pdf", fig=fig)

        return
    
    def plot_u2(self):
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        xx = jnp.linspace(xlim[0], xlim[1], 501)
        true_vals = self.get_true_vals(xx)
        
        fig = plt.figure()
        plt.plot(xx, netmap(self.grad2)(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, true_vals["2"].ravel(), color='orange')
        plt.legend(["PINN", "True"])
        save_fig(self.dir.figure_dir, "u_diff2.pdf", format="pdf", fig=fig)

        fig = plt.figure()
        plt.semilogy(xx, jnp.abs(netmap(self.grad2)(self.params, xx.reshape(-1, 1)).ravel() - true_vals["2"].ravel()))
        save_fig(self.dir.figure_dir, "u_diff2_error.pdf", format="pdf", fig=fig)
        
        return
        
    def plot_u(self):
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        xx = jnp.linspace(xlim[0], xlim[1], 501)
        true_vals = self.get_true_vals(xx)

        fig = plt.figure()
        plt.plot(xx, netmap(self.forward)(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, true_vals["0"].ravel(), color='orange')
        plt.legend(["PINN", "True"])
        save_fig(self.dir.figure_dir, "u.pdf", format="pdf", fig=fig)

        fig = plt.figure()
        plt.semilogy(xx, jnp.abs(netmap(self.forward)(self.params, xx.reshape(-1, 1)).ravel() - true_vals["0"].ravel()))
        save_fig(self.dir.figure_dir, "u_error.pdf", format="pdf", fig=fig)
        
        return
        
    def plot_derivatives(self):
        rc("text", usetex=True)
        rc('text.latex', preamble=r'\usepackage{amsmath}')
        
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        xx = jnp.linspace(xlim[0], xlim[1], 501)
        true_vals = self.get_true_vals(xx)
        
        plot_data0 = [netmap(func)(self.params, xx.reshape(-1,1)).ravel() for func in [self.forward, self.grad1, self.grad2, self.grad3, self.grad4]]
        plot_data1 = [true_vals[str(i)].ravel() for i in range(5)]
        plot_data2 = [jnp.abs(plot_data0[i] - plot_data1[i]) for i in range(5)]
        
        # fig = plt.figure()
        # for i in range(5):
        #     plt.plot(xx, plot_data0[i])
        # plt.legend([r"$\hat u_{" + "x"*i + "}$" for i in range(5)])
        # plt.xlabel("x")
        # save_fig(self.dir.figure_dir, "diff.pdf", format="pdf", fig=fig)
        
        # fig = plt.figure()
        # for i in range(5):
        #     plt.plot(xx, plot_data1[i])
        # plt.legend([r"$u_{" + "x"*i + "}$" for i in range(5)])
        # plt.xlabel("x")
        # save_fig(self.dir.figure_dir, "diff_true.pdf", format="pdf", fig=fig)

        # fig = plt.figure()
        # for i in range(5):
        #     plt.semilogy(xx, plot_data2[i])
        # plt.legend([r"$|\hat u_{" + "x"*i + "} - u_{" + "x"*i + "}|$" for i in range(5)])
        # plt.xlabel("x")
        # save_fig(self.dir.figure_dir, "diff_error.pdf", format="pdf", fig=fig)


        fig, ax = plt.subplots(5, 3, figsize=(30, 50))
        for i in range(5):
            ax[i, 0].plot(xx, plot_data0[i])
            if i == 0:
                ax[i, 0].legend([r"$\hat u$"], fontsize=25, loc="lower left")
            else:
                ax[i, 0].legend([r"$\partial_{" + r"x"*i + r"}\hat u$"], fontsize=25, loc="lower left")
            ax[i, 0].set_ylim(min([min(data - (min(data)+max(data))/2)*2 + (min(data)+max(data))/2 for data in plot_data1]), max([max(data - (min(data)+max(data))/2)*2 + (min(data)+max(data))/2 for data in plot_data1]))
            ax[i, 0].set_xlim(self.geometry_settings["domain"]["interval"]["xlim"])
            ax[i, 0].set_box_aspect(1)
        
        for i in range(5):
            ax[i, 1].plot(xx, plot_data1[i])
            if i == 0:
                ax[i, 1].legend([r"$u$"], fontsize=25, loc="lower left")
            else:
                ax[i, 1].legend([r"$\partial_{" + r"x"*i + r"}u$"], fontsize=25, loc="lower left")
            ax[i, 1].set_ylim(min([min(data - (min(data)+max(data))/2)*2 + (min(data)+max(data))/2 for data in plot_data1]), max([max(data - (min(data)+max(data))/2)*2 + (min(data)+max(data))/2 for data in plot_data1]))
            ax[i, 1].set_xlim(self.geometry_settings["domain"]["interval"]["xlim"])
            ax[i, 1].set_box_aspect(1)
        
        for i in range(5):
            ax[i, 2].semilogy(xx, plot_data2[i])
            if i == 0:
                ax[i, 2].legend([r"$|\hat u - u|$"], fontsize=25, loc="lower left")
            else:
                ax[i, 2].semilogy(xx, plot_data2[i-1], '--')
                if i == 1:
                    ax[i, 2].legend([r"$|\partial_{" + r"x"*i + r"}\hat u - \partial_{" + r"x"*i + r"}u|$", r"$|\hat u - u|$"], fontsize=25, loc="lower left")
                else:
                    ax[i, 2].legend([r"$|\partial_{" + r"x"*i + r"}\hat u - \partial_{" + r"x"*i + r"}u|$", r"$|\partial_{" + r"x"*(i-1) + r"}\hat u - \partial_{" + r"x"*(i-1) + r"}u|$"], fontsize=25, loc="lower left")
            ax[i, 2].set_ylim(min([min(data) for data in plot_data2])/10, max([max(data) for data in plot_data2])*10)
            ax[i, 2].set_xlim(self.geometry_settings["domain"]["interval"]["xlim"])
            ax[i, 2].set_box_aspect(1)
        
        titles = ["Prediction", "True solution", "Absolute error"]
        for c in range(3):
            ax[0, c].set_title(titles[c], fontsize=50)
            ax[4, c].set_xlabel(r"$x$", fontsize=40)
            for r in range(5):
                ax[r, c].tick_params(labelsize=25)
        
        save_fig(self.dir.figure_dir, "diff_all.pdf", format="pdf", fig=fig)
        
        return

    def eval(self):
        
        u = jnp.squeeze(netmap(self.forward)(self.params, self.eval_points["coll"]))
        u1 = jnp.squeeze(netmap(self.grad1)(self.params, self.eval_points["coll"]))
        u2 = jnp.squeeze(netmap(self.grad2)(self.params, self.eval_points["coll"]))
        u3 = jnp.squeeze(netmap(self.grad3)(self.params, self.eval_points["coll"]))
        u4 = jnp.squeeze(netmap(self.grad4)(self.params, self.eval_points["coll"]))
        
        u_true = self.eval_true_val["0"]
        u1_true = self.eval_true_val["1"]
        u2_true = self.eval_true_val["2"]
        u3_true = self.eval_true_val["3"]
        u4_true = self.eval_true_val["4"]
        
        for metric_fun, fun_name in zip([mse, maxabse, L2rel],["mse", "maxabse", "L2rel"]):

            err = metric_fun(u, u_true)
            err1 = metric_fun(u1, u1_true)
            err2 = metric_fun(u2, u2_true)
            err3 = metric_fun(u3, u3_true)
            err4 = metric_fun(u4, u4_true)
            
            attr_name = "eval_result"

            if hasattr(self, attr_name):
                if isinstance(self.eval_result, dict):
                    self.eval_result.update({fun_name: {"0": err, 
                                                    "1": err1, 
                                                    "2": err2, 
                                                    "3": err3, 
                                                    "4": err4}})
                else:
                    raise TypeError(f"Attribute '{attr_name}' is not a dictionary. "
                                    f"Evaluation error cannot be added.")
            else:
                self.eval_result = {fun_name: {}}
                self.eval_result[fun_name] = {"0": err, 
                                            "1": err1, 
                                            "2": err2, 
                                            "3": err3, 
                                            "4": err4}
                
        return err