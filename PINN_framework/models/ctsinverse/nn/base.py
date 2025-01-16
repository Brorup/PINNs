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
from models.cts import CTSNN
from models.networks import netmap
from models.ctsinverse import plotting as ctsplot
from models.loss import L2rel, mse, maxabse, ms, sq, sqe
from utils.utils import timer
from setup.parsers import load_json

class CTSINN(CTSNN):
    """
    NN class. Inherits from CTSNN but implements things that differ between the forward and inverse problem

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

                    CTSINN
                
        << Specific implementations >>
    
    ```
    """

    net: nn.Module
    optimizer: optax.GradientTransformation

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)

        return

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
    
    def loss_data_individual(self, params, input: jax.Array, true_val: jax.Array | None = None):
        
        # Compute model output
        out = netmap(self.forward)(params, input)

        # Compute loss
        if self.loss_fn is None:
            loss_fn = mse
        else:
            loss_fn = self.loss_fn

        if true_val is None:
            return jax.vmap(loss_fn, in_axes=(1, ))(out)
        return jax.vmap(loss_fn, in_axes=(1, 1))(out, true_val)

    def scale_output(self, params):
        """
        Method to scale the output back from having been normalized or scaled
        """
        for index, param_range in enumerate(self.param_ranges):
            if param_range[0] == param_range[1]:
                params[:, index] = params[:, index] * param_range[0]
            else:
                params[:, index] = params[:, index] * (param_range[1] - param_range[0]) + param_range[0]

        return params

    def early_stopping(self):
        """
        Stops training early if validation loss stops decreasing
        """
        if self.early_stop_vars.do_check: 
            points = self.validation_points["cts_spectra"]
            u_true = self.validation_true_val["cts_params"]
            err = self.early_stopping_jittable(points, u_true, self.params)
        else:
            err = None

        stop = self._early_stopping(err)

        return stop

    def eval(self, metric: str  = "all", verbose = None, **kwargs):
        """
        Evaluates the error using the specified metric.
        """
        
        points = self.eval_points["cts_spectra"]
        u_true = self.eval_true_val["cts_params"]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
        err = self._eval(u, u_true, metric, verbose)
        
        return err
    
    def validation(self, metric: str  = "all", verbose = None, **kwargs):
        """
        Evaluates the error using the specified metric.
        """
        
        points = self.validation_points["cts_spectra"]
        u_true = self.validation_true_val["cts_params"]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
        err = self._eval(u, u_true, metric, verbose)
        
        return err

    def plot_results(self, save=True):
        ctsplot.plot_results(netmap(self.forward)(self.params, self.train_points["cts_spectra"]), self.train_true_val["cts_params"], self.param_names,
                             self.dir.figure_dir, name="train_data", varying_params=self.varying_params, save=save, dpi=self.plot_settings["dpi"])
        ctsplot.plot_results(netmap(self.forward)(self.params, self.eval_points["cts_spectra"]), self.eval_true_val["cts_params"], self.param_names,
                             self.dir.figure_dir, name="test_data", varying_params=self.varying_params, save=save, dpi=self.plot_settings["dpi"])
        
        