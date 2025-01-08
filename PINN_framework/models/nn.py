from typing import override
import sys

import jax
import jax.numpy as jnp
import optax

from .base import Model
from .networks import setup_network
from models.loss import L2rel, mse, maxabse, ms, sq, sqe

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
        match metric.lower():
            case "all":
                do_all_metrics = True
            case "l2-rel":
                metric_fun = jax.jit(L2rel)
                metric_description = "L2 relative error"
            case "l2rel":
                metric_fun = jax.jit(L2rel)
                metric_description = "L2 relative error"
            case "mse":
                metric_fun = jax.jit(mse)
                metric_description = "Mean squared error"
            case "maxabse":
                metric_fun = jax.jit(maxabse)
                metric_description = "Max abs error"
            case _:
                print(f"Unknown metric: '{metric}'. Default ('L2-rel') is used for evaluation.")
                metric_fun = jax.jit(L2rel)
        
        if do_all_metrics:
            metric_funs = [jax.jit(L2rel), jax.jit(mse), jax.jit(maxabse)]
            metric_descriptions = ["L2 relative error", "Mean squared error", "Max abs error"]
        else:
            metric_funs = [metric_fun]
            metric_descriptions = [metric_description]
        
        for (metric_fun, metric_description) in zip(metric_funs, metric_descriptions):
            err = metric_fun(u, u_true)

            attr_name = "eval_result"

            if hasattr(self, attr_name):
                if isinstance(self.eval_result, dict):
                    self.eval_result[metric] = err
                else:
                    raise TypeError(f"Attribute '{attr_name}' is not a dictionary. "
                                    f"Evaluation error cannot be added.")
            else:
                self.eval_result = {metric: err}
            
            if verbose:
                print(f"{metric_description} of model: {err:.2g}")
        
        if verbose:
            print("\n###############################################################\n\n")
            sys.stdout.flush()

        return err
        


    # def log_scalars(self,
    #                 scalars,
    #                 scalar_names: str | None = None,
    #                 tag: str | None = None,
    #                 step: int | None = None,
    #                 log: bool = False,
    #                 all_losses: jnp.ndarray | None = None):
    #     if log:
    #         writer = SummaryWriter(log_dir=self.dir.log_dir)
    #         writer.add_scalars(tag,
    #                         {name: np.array(loss) for name, loss in zip(scalar_names, scalars)},
    #                         global_step=step)
                    
    #         writer.close()
        
    #     return jnp.concatenate([all_losses, scalars.reshape(-1, scalars.shape[0])])
    
    # def plot_training_points(self, save=True, log=False, step=None):
        
    #     rc("text", usetex=True)
    #     rc('text.latex', preamble=r'\usepackage{amsmath}')
        
    #     plt.figure()
    #     _ = jtu.tree_map_with_path(lambda x, y: plt.scatter(np.array(y)[:,0], np.array(y)[:,1], **self.sample_plots.kwargs[x[0].key]), OrderedDict(self.train_points))
        
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1))
    #     plt.xlabel("$x$", fontsize=15)
    #     plt.ylabel("$y$", fontsize=15)
    #     plt.gca().set_box_aspect(1)
        
    #     if save:
    #         save_fig(self.dir.figure_dir, "training_points", "png", plt.gcf())
        
    #     plt.close()