from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from matplotlib import rc

from setup.settings import (
    ModelNotInitializedError,
    log_settings,
    DefaultSettings,
    SoftAdaptSettings,
    WeightedSettings,
    UnweightedSettings,
    GradNormSettings,
    AdaptiveWeightSchemeSettings
)
from setup.parsers import (
    parse_verbosity_settings,
    parse_logging_settings,
    parse_plotting_settings,
    parse_directory_settings,
    parse_run_settings,
    parse_loss_settings
)
from .optim import get_update
from .loss import softadapt, gradnorm, running_average
from utils.plotting import save_fig
from utils.checkpoint import write_model, load_model


class Model(metaclass=ABCMeta):
    """
    Base class that specific models such as PINNs and DeepONets inherit from.

    The main functionality on this level is to parse the settings specified in the JSON file.
    Additionally, abstract methods are defined.
    """
    params: optax.Params
    train_points: dict
    train_true_val: dict | None
    eval_points: dict
    eval_true_val: dict | None

    def __init__(self, settings: dict, *args, **kwargs):
        """
        Initialize the model by calling the settings parsers.
        """
        
        self._parse_settings(settings)
        return

    def _parse_settings(self, settings: dict):
        """
        Parse various settings.
        """
        
        # Parse verbosity, seed and ID
        self._verbose = parse_verbosity_settings(settings.get("verbosity"))
        self._seed = settings.get("seed")
        if self._seed is None:
            if self._verbose.init:
                print(f"No seed specified in settings. Seed is set to {DefaultSettings.SEED}.")
            self._seed = DefaultSettings.SEED
        self._id = settings.get("id")
        if self._id is None:
            if self._verbose.init:
                print(f"No ID specified in settings. ID is set to '{DefaultSettings.ID}'.")
            self._id = DefaultSettings.ID
        
        self._static_loss_args = ()
        
        # Set a key for use in other methods
        # Important: Remember to return a new _key as well, e.g.:
        #   self._key, key_for_some_task = jax.random.split(self._key)
        self._key = jax.random.PRNGKey(self._seed)

        # Parse more settings
        self._parse_directory_settings(settings["io"], self._id)
        self._parse_run_settings(settings["run"])
        self._prase_loss_settings(settings["run"]["train"].get("loss_fn"))
        self._parse_plotting_settings(settings["plotting"])
        self._parse_logging_settings(settings["logging"])
        
        if self.logging.do_logging:
            log_settings(settings, self.dir.log_dir, tensorboard=True, text_file=False, empty_dir=False)
        
        if self.train_settings.checkpoint_every is not None:
            self.write_model(init=True)

        if settings.get("description"):
            with open(self.dir.log_dir / "description.txt", "a+") as file:
                file.writelines([settings["description"], "\n\n"])

        return

    def _parse_directory_settings(self, dir_settings: dict, id: str) -> None:
        """
        Parse settings related to file directories.
        """

        self.dir = parse_directory_settings(dir_settings, id)
        return

    def _parse_run_settings(self, run_settings: dict) -> None:
        """
        Parse settings related to the type of run.
        """

        self.train_settings, self.do_train = parse_run_settings(run_settings, run_type="train")
        self.eval_settings, self.do_eval = parse_run_settings(run_settings, run_type="eval")
        return
    
    def _prase_loss_settings(self, loss_settings: str | None = None):
        if loss_settings is None:
            loss_settings = "mse"
        self.loss_fn = parse_loss_settings(loss_settings)
        return
    
    def _parse_plotting_settings(self, plot_settings: dict) -> None:
        """
        Parse settings related to plotting.
        """
        
        self.sample_plots = parse_plotting_settings(plot_settings.get("sampling"))
        self.result_plots = parse_plotting_settings(plot_settings.get("results"))
        plot_settings.pop("sampling")
        plot_settings.pop("results")
        self.plot_settings = plot_settings
        return
    
    def _parse_logging_settings(self, log_settings: dict) -> None:
        self.logging = parse_logging_settings(log_settings)
        return

    def _gen_key(self, n: int = 1):
        if n < 1:
            n = 1
        if n == 1:
            self._key, key = jax.random.split(self._key, 2)
            return key
        self._key, *keys = jax.random.split(self._key, n+1)
        return keys

    @abstractmethod
    def init_model(self) -> None:
        """
        This method should be implemented to initialize the model, e.g. the params.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        The forward pass through a model.
        """
        pass

    
    def load_model(self, step: int | None = None, dir: str | None = None):
        """
        Load model state.
        """
        if dir is None:
            dir = self.dir.model_dir
        
        if step is not None:
            self.params = load_model(step, dir)
        
        try: 
            self.train_settings.iterations
        except:
            step = 0
        else:
            step = self.train_settings.iterations
        finally:
            self.params, self.checkpoint_epoch = load_model(step, dir)
        return
        
    def write_model(self, step: int | None = None, dir: str | None = None, init: bool = False):
        """
        Save model state. Call to a more general function write_model() ...
        """
        if dir is None:
            dir = self.dir.model_dir
        
        if init:
            write_model(None, None, dir, init=init)
            return
        
        if step is not None:
            write_model(self.params, step, dir)
            return
            
        try: 
            self.train_settings.iterations
        except:
            step = 0
        else:
            step = self.train_settings.iterations
        finally:
            write_model(self.params, step, dir)
        return

    def _loss_terms(self) -> jax.Array:
        """
        Method for calculating loss terms. Loss terms may be defined
        in separate methods or calculated here.
        
        Must be overwritten by inheriting classes.
        """
        pass

    def _total_loss(self, *args, weights: jax.Array | None = None, **kwargs):
        """
        This function sums up the loss terms return in the loss_terms
        method and passes through the auxillary output.
        """
        loss_terms = self._loss_terms(*args, **kwargs)
        if weights is None:
            loss = jnp.sum(loss_terms)
        else:
            loss = jnp.dot(weights, loss_terms)
        return loss, loss_terms
    
    def _register_static_loss_arg(self, arg: str):
        if arg not in self._static_loss_args:
            self._static_loss_args = self._static_loss_args + (arg,)
        return
    
    def _unregister_static_loss_arg(self, arg: str):
        self._static_loss_args = tuple(a for a in self._static_loss_args if a != arg)
        return
    
    def _set_loss(self,
                  loss_term_fun_name: str) -> None:
        """
        (Stateful) method for setting loss function.
        """
        
        try:
            fun = getattr(self, loss_term_fun_name)
        except AttributeError as attr_err:
            raise RuntimeError(f"No loss term method with name '{loss_term_fun_name}'.") from attr_err

        self._loss_terms = fun
        return
    
    def init_weights(self,
                    loss_term_fun: Callable[..., jax.Array],
                    *args,
                    **kwargs
                    ) -> None:
        
        if not self._initialized():
            raise ModelNotInitializedError("The model has not been initialized properly.")
        
        loss_shape = loss_term_fun.eval_shape(*args, **kwargs).shape[0]
        
        # if not isinstance(self.train_settings.update_settings, AdaptiveWeightSchemeSettings):
        if self.train_settings.update_scheme == "weighted":
            
            curr_weights = self.train_settings.update_kwargs["weights"]
            if loss_shape > len(curr_weights):
                self.weights = jnp.concatenate([jnp.array(curr_weights), jnp.ones((loss_shape - len(curr_weights),))])
            else:
                self.weights = jnp.array(curr_weights[:loss_shape])
        else:
            self.weights = jnp.ones((loss_shape))
        
        if self.train_settings.update_settings.normalized:
            self.weights = self.weights / jnp.sum(self.weights)
        
        if self.train_settings.update_scheme == "softadapt":
            # Initialize previous losses
            self._prevlosses = loss_term_fun(*args, **kwargs).reshape(1, -1)
        return
            
    

    def get_weights(self,
                    epoch: int,
                    loss_term_fun: Callable[..., jax.Array],
                    *args,
                    **kwargs
                    ) -> None:
        """
        Method for initializing array of weights.
        If weights are None, the method sets weights
        to an array determined by the loss type.
        """
        
        if not self._initialized():
            raise ModelNotInitializedError("The model has not been initialized properly.")
        
        if not hasattr(self, "weights"):
            self.init_weights(loss_term_fun, *args, **kwargs)
            # raise AttributeError(f"Model {self.__class__} has no loss term weights.")
            return

        # Weights are not changed from initial state if the scheme is not adaptive
        if not isinstance(self.train_settings.update_settings, AdaptiveWeightSchemeSettings):
            return

        # Skip specified number of steps before updating weights
        if (epoch % self.train_settings.update_settings.update_every):
            return
        
        if self.train_settings.update_scheme == "softadapt":
            # Compute loss
            curr_loss = loss_term_fun(*args, **kwargs)

            # Not enough previous losses registered - use default weights, but register losses for next time
            if self._prevlosses.shape[0] <= self.train_settings.update_settings.order:
                self._prevlosses = jnp.concatenate((self._prevlosses, curr_loss.reshape(1, -1)), axis=0)
                return
            
            # Push oldest losses out
            self._prevlosses = jnp.roll(self._prevlosses, -1, axis=0).at[-1, :].set(curr_loss)

            # Calculate weights
            new_weights = softadapt(self.train_settings.update_settings, self._prevlosses, fdm_coeff=None)

        else: # grad_norm
            # Compute loss
            curr_loss = loss_term_fun(*args, **kwargs)

            # Prevent gradient tracking through the weights
            detached_params = jax.lax.stop_gradient(args[0])
            detached_grads = jax.jacrev(loss_term_fun)(detached_params, *args[1:], **kwargs)
            # detached_grads = jax.jacrev(loss_term_fun)(*args, **kwargs)
            new_weights = gradnorm(self.train_settings.update_settings, detached_grads, self.weights.shape[0], loss_weights=curr_loss)
        
        if self.train_settings.update_settings.running_average is not None:
            self.weights = running_average(new_weights, self.weights,
                                           alpha=self.train_settings.update_settings.running_average,
                                           normalized=self.train_settings.update_settings.normalized)
            if self.train_settings.update_settings.normalized:
                self.weights = self.weights / jnp.sum(self.weights)
        else:
            self.weights = new_weights

        return

    def _set_update(self,
                    loss_fun_name: str = "_total_loss",
                    optimizer_name: str = "optimizer"
                    ) -> None:
        """
        Method for setting update function.

        Currently supported:
            The usual unweighted update
            The SoftAdapt update

        Args:
            loss_fun_name:   Name of function computing the total loss.
            optimizer_name:  Name of the optimizer used.
        """

        loss_fun = getattr(self, loss_fun_name)
        optimizer = getattr(self, optimizer_name)
        
        self.update = get_update(loss_fun,
                                 optimizer,
                                 self.train_settings.jitted_update,
                                 verbose=self._verbose.training,
                                 verbose_kwargs={"print_every": self.logging.print_every},
                                 static_argnames=self._static_loss_args)
        return
    
    def _initialized(self) -> bool:
        has_params = hasattr(self, "params")
        has_train_points = hasattr(self, "train_points")
        has_train_points_trunk = hasattr(self, "train_points_trunk")
        has_eval_points = hasattr(self, "eval_points")
        has_eval_points_trunk = hasattr(self, "eval_points_trunk")
        return has_params and (has_train_points or (has_train_points_trunk)) #and (has_eval_points or (has_eval_points_branch and has_eval_points_trunk))

 
    @abstractmethod
    def train(self):
        """
        Method for training a model. Typically a loop structure with
        an update procedure in the loop. The update should be defined
        in a separate method so it can be JIT'ed easily.
        
        Must be overwritten by inheriting classes.
        """
        pass

    @abstractmethod
    def eval(self):
        """
        Method for evaluating a model.
        
        Must be overwritten by inheriting classes.
        """
        pass

    def __str__(self):
        """
        The string representation of the model.

        Prints out the methods/attributes and the documentation if any.
        """
        
        s = f"\n\nModel '{self.__class__.__name__}' with the following methods:\n\n\n\n"
        
        for m in dir(self):
            attr = getattr(self, m)
            if m.startswith("_") or not inspect.ismethod(attr):
                continue
            s += m
            s += ":\n"
            docstr = attr.__doc__
            s += docstr if docstr is not None else "\n\tNo documentation.\n"
            s += "\n\n"
        
        s += "\n\n\n\n\n... and the following attributes:\n\n\n\n"

        for m in dir(self):
            attr = getattr(self, m)
            if m.startswith("_") or inspect.ismethod(attr):
                continue
            s += m
            s += "\n\n"

        return s
    
    
    def plot_loss(
    self,
    loss_arr: jax.Array,
    loss_map: dict,
    *,
    fig_dir,
    name,
    epoch_step = None,
    extension="png",
    figsize = (35, 30)
    ) -> None:
        """
        Plots losses from array in different subplots according to the specified dict.
        """
        
        # num_plots = len(loss_map.keys())
        # fig, ax = plt.subplots(int(np.ceil(np.sqrt(num_plots))), int(np.ceil(np.sqrt(num_plots))), figsize=figsize)
        # plot_split = list(loss_map.keys())

        # if epoch_step is not None:
        #     epochs = epoch_step*np.arange(loss_arr.shape[0])
        #     for i in range(num_plots):
        #         ax[np.unravel_index(i, ax.shape)].semilogy(epochs, loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
        #         ax[np.unravel_index(i, ax.shape)].tick_params(axis='x')
        #         ax[np.unravel_index(i, ax.shape)].tick_params(axis='y')
        #         # ax[i].fill_between(epochs[epochs % 10000 >= 5000], 0, facecolor='gray', alpha=.5)
        # else:
        #     for i in range(num_plots):
        #         ax[np.unravel_index(i, ax.shape)].semilogy(loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
        #         ax[np.unravel_index(i, ax.shape)].tick_params(axis='x')
        #         ax[np.unravel_index(i, ax.shape)].tick_params(axis='y')
            
        # save_fig(fig_dir, name, extension, fig=fig)
        # plt.clf()
        
        # num_plots = len(loss_map.keys())
        
        # if epoch_step is not None:
        #     epochs = epoch_step*np.arange(loss_arr.shape[0])
        # else:
        #     epochs = np.arange(loss_arr.shape[0])
        
        # plt.semilogy(epochs, loss_arr, linewidth=2)
        # plt.tick_params(axis='x')
        # plt.tick_params(axis='y')
            
        # save_fig(fig_dir, "all_" + name, extension, fig=plt.gcf())
        
        rc("text", usetex=True)
        rc('text.latex', preamble=r'\usepackage{amsmath}')
        
        fig = plt.figure(figsize=(18,10))
        plt.grid(True)
        if epoch_step is not None:
            ll = epoch_step*np.arange(1, loss_arr.shape[0] + 1)
        else:
            ll = np.arange(1, loss_arr.shape[0] + 1)
        plt.semilogy(ll, loss_arr)
        # if hasattr (self, "loss_names"):
        #     plt.legend(self.loss_names, fontsize=25, framealpha=1.0)
        # else:
        plt.legend(list(loss_map.keys()), fontsize=25, framealpha=1.0, loc="upper right")
        # plt.ylim(8e-7, 1e1)
        ax = fig.gca()
        ax.set_xticks(np.linspace(0, max(ll), 11, dtype=np.int32))
        ax.set_xticklabels(np.linspace(0, max(ll), 11, dtype=np.int32), fontsize=25)
        ax.set_xlabel(r"\textbf{Epochs}", fontsize=30)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_ylabel(r"\textbf{MSE}", fontsize=30, rotation=0, ha="right", labelpad=5)
        fig.savefig(fig_dir / (name + ".pdf"), bbox_inches="tight")
        
        return
        
        
    