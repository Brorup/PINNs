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

    def scale_output(self):
        pass

    def eval(self, metric: str  = "all", verbose = None, **kwargs):
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

        points = self.eval_points["cts_spectra"]
        u_true = self.eval_true_val["cts_params"]
        
        u = netmap(self.forward)(self.params, points).squeeze()
        
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

    def plot_results(self, save=True):
        ctsplot.plot_results(netmap(self.forward)(self.params, self.eval_points["cts_spectra"]), self.eval_true_val["cts_params"], self.param_names,
                             self.dir.figure_dir, save=save, dpi=self.plot_settings["dpi"])
        
        