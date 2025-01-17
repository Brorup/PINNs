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
from sklearn.model_selection import train_test_split

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
        
        self.truncate_spectra_to = settings["run"]["truncate_spectra_to"]
        self.notch_size = settings["run"]["notch_size"]

        self.init_model(self.alter_net_size(settings["model"]["nn"]["network"]))
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")

        self.data_dir = self.dir.data_dir / settings["run"]["data_dir"]
        self.train_data_dir = self.data_dir / "train"
        self.eval_data_dir = self.data_dir / "test"
        
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
        
    def alter_net_size(self, settings):
        """
        Method for automatically fixing the input (CTSinverse) or output (CTS) size of the networks taking truncation and notch into account
        """
        if settings[0]['specifications']['input_dim'] == -1:
            settings[0]['specifications']['input_dim'] = self.truncate_spectra_to - self.notch_size
    
        if settings[0]['specifications']['output_dim'] == -1:
            settings[0]['specifications']['output_dim'] = self.truncate_spectra_to - self.notch_size
    
        return settings
    
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
            self.varying_params = []
            if self._verbose.data:
                print("Parameter ranges:")
            for (key, value) in param_ranges.items():
                if key != 'sampling':
                    if self._verbose.data:
                        print(f"\t{key}: \t[{value['min']}, {value['max']}]")
                    self.param_names.append(key)
                    if value['min'] == value["max"]:
                        self.varying_params.append(False)
                    else:
                        self.varying_params.append(True)
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

        # Load train and validation data
        # Load varying params
        cts_params_filename = self.train_data_dir / "scanem_list_var.bin"
        cts_params = np.fromfile(cts_params_filename, dtype=np.float64).reshape((-1, num_params))
        num_data_points = cts_params.shape[0]

        # Load static params from namelist file
        nml_filename = self.train_data_dir / "scanem_list.par"
        nml = f90nml.read(nml_filename)["scannml"]
        num_freqs = nml["nfrqs"]

        # Load spectra
        spectra_file = self.train_data_dir / "scanem_list.dat"
        spectra = np.memmap(spectra_file, np.float64, mode='r', shape=(num_data_points,num_freqs))

        # Split train and validation data
        self._key, train_test_key = jax.random.split(self._key, 2)
        if self.train_settings.train_validation_split == 1.0:
            train_cts_params = cts_params
            train_spectra = spectra
            validation_cts_params = cts_params # These are not used, but are just generated to not generate errors later on
            validation_spectra = spectra # These are not used, but are just generated to not generate errors later on
        else:
            train_cts_params, validation_cts_params, train_spectra, validation_spectra = train_test_split(cts_params, spectra, train_size=self.train_settings.train_validation_split, random_state=np.random.RandomState(train_test_key))
        self.train_points = {"cts_params": train_cts_params, "cts_spectra": train_spectra}
        self.validation_points = {"cts_params": validation_cts_params, "cts_spectra": validation_spectra}


        # Load test data
        # Load varying params
        cts_params_filename = self.eval_data_dir / "scanem_list_var.bin"
        cts_params = np.fromfile(cts_params_filename, dtype=np.float64).reshape((-1, num_params))
        num_eval_points = cts_params.shape[0]

        # Load static params from namelist file
        nml_filename = self.eval_data_dir / "scanem_list.par"
        nml = f90nml.read(nml_filename)["scannml"]
        num_freqs = nml["nfrqs"]

        # Load spectra
        spectra_file = self.eval_data_dir / "scanem_list.dat"
        spectra = np.memmap(spectra_file, np.float64, mode='r', shape=(num_eval_points,num_freqs))

        self.eval_points = {"cts_params": cts_params, "cts_spectra": spectra}

        self.truncate_spectra()

        self.notch_spectra()

        self.normalize_params()

        self.scale_spectra()

        self.numpy_to_jax_data()

        self.train_true_val = self.train_points
        self.validation_true_val = self.validation_points
        self.eval_true_val = self.eval_points

        self._key, dataset_key = jax.random.split(self._key, 2)

        self.full_batch_dataset = JaxDataset(key=dataset_key, xy=self.train_points["cts_params"], u=self.train_true_val["cts_spectra"], batch_size=self.train_settings.batch_size)
        
        self.train_points_batch = self.train_points.copy()
        self.train_true_val_batch = self.train_true_val.copy()

        if self._verbose.sampling:
            print("###############################################################\n\n")
            sys.stdout.flush()

        return
    
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

            self.validation_points["cts_spectra"] = np.delete(self.validation_points["cts_spectra"], np.s_[-num_truncate_r:], 1)
            self.validation_points["cts_spectra"] = np.delete(self.validation_points["cts_spectra"], np.s_[:num_truncate_l], 1)
            
            self.eval_points["cts_spectra"] = np.delete(self.eval_points["cts_spectra"], np.s_[-num_truncate_r:], 1)
            self.eval_points["cts_spectra"] = np.delete(self.eval_points["cts_spectra"], np.s_[:num_truncate_l], 1)

        return
    
    def notch_spectra(self):
        """
        Truncate spectra according to notch filter
        """
        if self._verbose.data:
            print(f"Truncating data from {self.train_points["cts_spectra"].shape[1]} to {self.train_points["cts_spectra"].shape[1] - self.notch_size} frequency bins, by applying notch filter...\n")
        num_truncate = self.notch_size
        truncate_from = int(self.train_points["cts_spectra"].shape[1]/2 - num_truncate/2)
        truncate_to = int(self.train_points["cts_spectra"].shape[1]/2 + num_truncate/2)

        self.train_points["cts_spectra"] = np.delete(self.train_points["cts_spectra"], np.s_[truncate_from:truncate_to], 1)

        self.validation_points["cts_spectra"] = np.delete(self.validation_points["cts_spectra"], np.s_[truncate_from:truncate_to], 1)
        
        self.eval_points["cts_spectra"] = np.delete(self.eval_points["cts_spectra"], np.s_[truncate_from:truncate_to], 1)

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
                        if value['min'] == 0:
                            self.train_points["cts_params"][:, index] = self.train_points["cts_params"][:, index] + 1
                            self.validation_points["cts_params"][:, index] = self.validation_points["cts_params"][:, index] + 1
                            self.eval_points["cts_params"][:, index] = self.eval_points["cts_params"][:, index] + 1
                        else:
                            self.train_points["cts_params"][:, index] = self.train_points["cts_params"][:, index] / value['min']
                            self.validation_points["cts_params"][:, index] = self.validation_points["cts_params"][:, index] / value['min']
                            self.eval_points["cts_params"][:, index] = self.eval_points["cts_params"][:, index] / value['min']
                    else:
                        self.train_points["cts_params"][:, index] = (self.train_points["cts_params"][:, index] - value['min']) / (value['max'] - value['min'])
                        self.validation_points["cts_params"][:, index] = (self.validation_points["cts_params"][:, index] - value['min']) / (value['max'] - value['min'])
                        self.eval_points["cts_params"][:, index] = (self.eval_points["cts_params"][:, index] - value['min']) / (value['max'] - value['min'])

        return
    
    def normalize_spectra(self, norm='max'):  
        """
        Method to normalize spectra according to some specified norm
        """
        self.train_points["cts_spectra"] = normalize(self.train_points["cts_spectra"], norm=norm)
        self.validation_points["cts_spectra"] = normalize(self.validation_points["cts_spectra"], norm=norm)
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
        self.validation_points["cts_spectra"] = self.validation_points["cts_spectra"]*scaling
        self.eval_points["cts_spectra"] = self.eval_points["cts_spectra"]*scaling
        return

    def scale_output(self, spectra):
        """
        Method to scale the output back from having been normalized or scaled
        """
        if scaling is None:
            scaling = self.spectra_scaling

        return spectra / scaling
    

    def numpy_to_jax_data(self):
        """
        Method to convert from numpy arrays or memmaps to jax arrays
        """
        self.train_points["cts_spectra"] = jnp.asarray(self.train_points["cts_spectra"])
        self.validation_points["cts_spectra"] = jnp.asarray(self.validation_points["cts_spectra"])
        self.eval_points["cts_spectra"] = jnp.asarray(self.eval_points["cts_spectra"])

        self.train_points["cts_params"] = jnp.asarray(self.train_points["cts_params"])
        self.validation_points["cts_params"] = jnp.asarray(self.validation_points["cts_params"])
        self.eval_points["cts_params"] = jnp.asarray(self.eval_points["cts_params"])

        return
    
    @partial(jax.jit, static_argnums=(0,))
    def early_stopping_jittable(self, points, u_true, params):
        return mse(netmap(self.forward)(params, points), u_true)
    
    def early_stopping(self):
        """
        Stops training early if validation loss stops decreasing
        """
        if self.early_stop_vars.do_check:
            points = self.validation_points["cts_params"]
            u_true = self.validation_true_val["cts_spectra"]
            err = self.early_stopping_jittable(points, u_true, self.params)
        else:
            err = None

        stop = self._early_stopping(err)

        return stop
    
    def eval(self, metric: str  = "all", verbose = None, **kwargs):
        """
        Evaluates the error using the specified metric.
        """
        
        points = self.eval_points["cts_params"]
        u_true = self.eval_true_val["cts_spectra"]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
        err = self._eval(u, u_true, metric, verbose)
        
        return err
    
    def validation(self, metric: Callable | str  = "all", verbose = None, **kwargs):
        """
        Evaluates the error using the specified metric.
        """
        
        points = self.validation_points["cts_params"]
        u_true = self.validation_true_val["cts_spectra"]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
        err = self._eval(u, u_true, metric, verbose)
        
        return err

    def plot_results(self, save=True, log=False, step=None):
        ctsplot.plot_results(netmap(self.forward)(self.params, self.train_points["cts_params"]), self.train_true_val["cts_spectra"],
                             self.dir.figure_dir, type="train", save=save, dpi=self.plot_settings["dpi"])
        ctsplot.plot_results(netmap(self.forward)(self.params, self.eval_points["cts_params"]), self.eval_true_val["cts_spectra"],
                             self.dir.figure_dir, type="eval", save=save, dpi=self.plot_settings["dpi"])
        
    def plot_loss_history(self):
        with open(self.dir.log_dir.joinpath('all_train_losses.npy'), "rb") as f:
            train_loss_history = jnp.load(f)

        with open(self.dir.log_dir.joinpath('all_validation_losses.npy'), "rb") as f:
            validation_loss_history = jnp.load(f)

        plot_loss_history(train_loss_history, validation_loss_history=validation_loss_history, stepsize=self.logging.log_every, dir=self.dir.figure_dir, file_name="loss_history", format="png")

        return

    def do_every(self, epoch: int | None = None, loss_term_fun: Callable | None = None, last = False, **kwargs):
        
        log_every = self.logging.log_every
        do_log = self.logging.do_logging
        checkpoint_every = self.train_settings.checkpoint_every

        if do_log & (epoch % log_every == log_every-1):
            loss_terms = loss_term_fun(params=self.params, inputs=self.train_points, true_val=self.train_true_val, **kwargs)
            if epoch // log_every == 0:
                self.all_train_losses = []
            self.all_train_losses.append(loss_terms)

            loss_terms = loss_term_fun(params=self.params, inputs=self.validation_points, true_val=self.validation_true_val, **kwargs)
            if epoch // log_every == 0:
                self.all_validation_losses = []
            self.all_validation_losses.append(loss_terms)

        if epoch % checkpoint_every == checkpoint_every-1:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_train_losses"):
                with open(self.dir.log_dir.joinpath('all_train_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_train_losses))
            
            if hasattr(self, "all_validation_losses"):
                with open(self.dir.log_dir.joinpath('all_validation_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_validation_losses))

        if last:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_train_losses"):
                with open(self.dir.log_dir.joinpath('all_train_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_train_losses))
            
            if hasattr(self, "all_validation_losses"):
                with open(self.dir.log_dir.joinpath('all_validation_losses.npy'), "wb") as f:
                    jnp.save(f, jnp.array(self.all_validation_losses))
        
        return