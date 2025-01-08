from collections.abc import Callable
from typing import override
from functools import partial
from pathlib import Path
import sys
import time
import datetime
import json

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import f90nml
import numpy as np
import pprint
from sklearn.preprocessing import normalize

from datahandlers.generators import (
    generate_rectangle,
    generate_collocation_points,
    resample,
    resample_idx,
    JaxDataset
)
from models import NN
from models.networks import netmap
from models.cts import plotting as ctsplot
from models.loss import L2rel, mse, maxabse, ms, sq, sqe
from utils.utils import timer
from utils.plotting import plot_loss_history
from setup.parsers import load_json

class CTSNN(NN):
    """
    NN class.

    The methods in this class are common for all experiments made.
    They include:
    
        self.predict():
            The method use for external calls of the models.
        
        self.plot_results():
            The method used for plotting the prediction.

        self.load_data():
            The method used for loading data for training 
    
    ```
                    Model   ________________________
                                                    |
                      |                             |
                      V                             V

                     NN             << Other models, e.g. DeepONet >>

                      |
                      V

                    CTSNN

                      |
                      V
                
        << Specific implementations >>
    
    ```
    """

    net: nn.Module
    optimizer: optax.GradientTransformation

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)

        self.init_model(settings["model"]["nn"]["network"])
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")

        self.data_dir = self.dir.data_dir / settings["run"]["data_dir"]
        self.train_data_dir = self.data_dir / "train"
        self.eval_data_dir = self.data_dir / "test"

        self.truncate_spectra_to = settings["run"]["truncate_spectra_to"]
        
        self.spectra_scaling = settings["run"]["beam_overlap"] * settings["run"]["input_power"] * 1e3

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
    
    def load_data(self):
        """
        Method used for loading training and test data from data folder.
        """

        if self._verbose.data:
            print("Loading data:\n")

        # Load train and test data from data folder
        if Path.is_file(self.data_dir / "param_ranges.json"):
            param_ranges = load_json(self.data_dir / "param_ranges.json")
            num_params = len(param_ranges.keys()) - 1
            self.param_names = []
            if self._verbose.data:
                print("Parameter ranges:")
            for (key, value) in param_ranges.items():
                if key != 'sampling':
                    if self._verbose.data:
                        print(f"\t{key}: \t[{value['min']}, {value['max']}]")
                        self.param_names.append(key)
                else:
                    if self._verbose.data:
                        print(f"\nSampling:")
                        print(f"\tTraining points generated using {value['train']['sampling']} sampling")
                        print(f"\tNumber of training points: {value['train']['points']}\n")
                        print(f"\tTest points generated using {value['test']['sampling']} sampling")
                        print(f"\tNumber of test points: {value['test']['points']}\n")

        else:
            num_params = 9
            self.param_names = ["FRQi", "theta", "phi", "Bmod", "Ne", "Te", "Ti", "Vd", "Ri"]

        cts_params_filename = self.train_data_dir / "scanem_list_var.bin"
        cts_params = np.fromfile(cts_params_filename, dtype=np.float64) 
        self.train_points = {"cts_params": cts_params.reshape((-1, num_params))}
        num_train_points = self.train_points["cts_params"].shape[0]

        nml_filename = self.train_data_dir / "scanem_list.par"
        nml = f90nml.read(nml_filename)["scannml"]
        num_freqs = nml["nfrqs"]

        spectra_file = self.train_data_dir / "scanem_list.dat"
        spectra = np.memmap(spectra_file, np.float64, mode='r', shape=(num_train_points,num_freqs))
        self.train_points["cts_spectra"] = spectra 

        cts_params_filename = self.eval_data_dir / "scanem_list_var.bin"
        cts_params = np.fromfile(cts_params_filename, dtype=np.float64) 
        self.eval_points = {"cts_params": cts_params.reshape((-1, num_params))}
        num_eval_points = self.eval_points["cts_params"].shape[0]

        nml_filename = self.eval_data_dir / "scanem_list.par"
        nml = f90nml.read(nml_filename)["scannml"]
        num_freqs = nml["nfrqs"]

        spectra_file = self.eval_data_dir / "scanem_list.dat"
        spectra = np.memmap(spectra_file, np.float64, mode='r', shape=(num_eval_points,num_freqs))
        self.eval_points["cts_spectra"] = spectra 

        self.truncate_spectra()

        self.normalize_params()

        self.scale_spectra()

        self.numpy_to_jax_data()

        self.train_true_val = self.train_points
        self.eval_true_val = self.eval_points

        self._key, dataset_key = jax.random.split(self._key, 2)

        self.full_batch_dataset = JaxDataset(key=dataset_key, xy=self.train_points["cts_params"], u=self.train_true_val["cts_spectra"], batch_size=self.train_settings.batch_size)
        
        self.train_points_batch = self.train_points.copy()
        self.train_true_val_batch = self.train_true_val.copy()

        if self._verbose.sampling:
            print("###############################################################\n\n")
            sys.stdout.flush()

        return

    def normalize_params(self):
        """
        Method to normalize and truncate (remove non-varying) parameters
        """

        # Load train and test data from data folder
        if Path.is_file(self.data_dir / "param_ranges.json"):
            self.param_ranges = []
            
            param_ranges = load_json(self.data_dir / "param_ranges.json")

            for index, (key, value) in enumerate(param_ranges.items()):
                if key != 'sampling':
                    self.param_ranges.append([value['min'], value['max']])
                    
                    if value['min'] == value['max']:
                        self.train_points["cts_params"][:, index] = self.train_points["cts_params"][:, index] / value['min']
                        self.eval_points["cts_params"][:, index] = self.eval_points["cts_params"][:, index] / value['min']
                    else:
                        self.train_points["cts_params"][:, index] = (self.train_points["cts_params"][:, index] - value['min']) / (value['max'] - value['min'])
                        self.eval_points["cts_params"][:, index] = (self.eval_points["cts_params"][:, index] - value['min']) / (value['max'] - value['min'])
            
            self.train_points["cts_params"] = self.train_points["cts_params"]
            self.eval_points["cts_params"] = self.eval_points["cts_params"]
        return
    
    def normalize_spectra(self, norm='max'):  
        """
        Method to normalize spectra according to some specified norm
        """
        self.train_points["cts_spectra"] = normalize(self.train_points["cts_spectra"], norm=norm)
        self.eval_points["cts_spectra"] = normalize(self.eval_points["cts_spectra"], norm=norm)
        return
    
    def scale_spectra(self, scaling = None):
        """
        Method to scale spectra according to some scaling. If no scaling is specified
        """
        if scaling is None:
            scaling = self.spectra_scaling

        if self._verbose.data:
            print(f"Scaling spectra with a factor of {scaling}")
        self.train_points["cts_spectra"] = self.train_points["cts_spectra"]*scaling
        self.eval_points["cts_spectra"] = self.eval_points["cts_spectra"]*scaling
        return

    def scale_output(self, spectra):
        """
        Method to scale the output back from having been normalized or scaled
        """
        if scaling is None:
            scaling = self.spectra_scaling

        return spectra / scaling
        
    def truncate_spectra(self):
        """
        method to truncate the spectra so the peaks at the edges are filtered out
        """
        if self.truncate_spectra_to != self.train_points["cts_spectra"].shape[1]:
            if self._verbose.data:
                print(f"Truncating data from {self.train_points["cts_spectra"].shape[1]} to {self.truncate_spectra_to} frequency bins...\n")
            num_truncate = self.train_points["cts_spectra"].shape[1] - self.truncate_spectra_to
            if num_truncate % 2 == 0:
                num_truncate_l = num_truncate // 2
                num_truncate_r = num_truncate // 2
            else:
                num_truncate_l = num_truncate // 2
                num_truncate_r = num_truncate // 2 + 1

            self.train_points["cts_spectra"] = np.delete(self.train_points["cts_spectra"], np.s_[-num_truncate_r:], 1)
            self.train_points["cts_spectra"] = np.delete(self.train_points["cts_spectra"], np.s_[:num_truncate_l], 1)

            self.eval_points["cts_spectra"] = np.delete(self.eval_points["cts_spectra"], np.s_[-num_truncate_r:], 1)
            self.eval_points["cts_spectra"] = np.delete(self.eval_points["cts_spectra"], np.s_[:num_truncate_l], 1)

        return

    def numpy_to_jax_data(self):
        """
        Method to convert from numpy arrays or memmaps to jax arrays
        """
        self.train_points["cts_spectra"] = jnp.asarray(self.train_points["cts_spectra"])
        self.eval_points["cts_spectra"] = jnp.asarray(self.eval_points["cts_spectra"])

        self.train_points["cts_params"] = jnp.asarray(self.train_points["cts_params"])
        self.eval_points["cts_params"] = jnp.asarray(self.eval_points["cts_params"])

        return
    
    def eval(self, metric: str  = "all", verbose = None, **kwargs):
        """
        Evaluates the error using the specified metric.
        """
        
        points = self.eval_points["cts_params"]
        u_true = self.eval_true_val["cts_spectra"]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
        err = self._eval(u, u_true, metric, verbose)
        
        return err

    def plot_results(self, save=True, log=False, step=None):
        ctsplot.plot_results(netmap(self.forward)(self.params, self.eval_points["cts_params"]), self.eval_true_val["cts_spectra"],
                             self.dir.figure_dir, save=save, dpi=self.plot_settings["dpi"])
        
    def plot_loss_history(self):
        with open(self.dir.log_dir.joinpath('all_train_losses.npy'), "rb") as f:
            train_loss_history = jnp.load(f)

        with open(self.dir.log_dir.joinpath('all_eval_losses.npy'), "rb") as f:
            eval_loss_history = jnp.load(f)

        plot_loss_history(train_loss_history, eval_loss_history=eval_loss_history, stepsize=self.logging.log_every, dir=self.dir.figure_dir, file_name="loss_history", format="png")

        return

    def do_every(self, epoch: int | None = None, loss_term_fun: Callable | None = None, last = False, **kwargs):
        
        log_every = self.logging.log_every
        do_log = self.logging.do_logging
        checkpoint_every = self.train_settings.checkpoint_every

        if do_log & (epoch % log_every == log_every-1):
            loss_terms = loss_term_fun(params=self.params, inputs=self.train_points, true_val=self.train_true_val, **kwargs)
            if epoch // log_every == 0:
                self.all_train_losses = []#jnp.zeros((0, loss_terms.shape[0]))
            self.all_train_losses.append(loss_terms)#jnp.append(self.all_train_losses, loss_terms)

            loss_terms = loss_term_fun(params=self.params, inputs=self.eval_points, true_val=self.eval_true_val, **kwargs)
            if epoch // log_every == 0:
                self.all_eval_losses = []#jnp.zeros((0, loss_terms.shape[0]))
            self.all_eval_losses.append(loss_terms)#jnp.append(self.all_eval_losses, loss_terms)

        if epoch % checkpoint_every == checkpoint_every-1:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_train_losses"):
                with open(self.dir.log_dir.joinpath('all_train_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_train_losses))
            
            if hasattr(self, "all_eval_losses"):
                with open(self.dir.log_dir.joinpath('all_eval_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_eval_losses))

        if last:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_train_losses"):
                with open(self.dir.log_dir.joinpath('all_train_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_train_losses))
            
            if hasattr(self, "all_eval_losses"):
                with open(self.dir.log_dir.joinpath('all_eval_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_eval_losses))
        
        return