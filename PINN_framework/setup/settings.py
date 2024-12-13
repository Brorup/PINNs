import os
from dataclasses import dataclass, field
from collections.abc import Callable
import pathlib

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from utils.utils import WaveletActivation


class SettingsInterpretationError(Exception):
    pass


class SettingsNotSupportedError(Exception):
    pass


class ModelNotInitializedError(RuntimeError):
    pass


class Settings:
    pass


class DefaultSettings(Settings):
    """
    Contains default values for various settings
    """

    # General defaults for a model
    SEED: int = 0
    ID: str = "generic_id"

    # Related to loss algorithms 
    SOFTADAPT_ORDER: int = 1
    SAVE_LAST_LOSSES: int = 5
    SOFTMAX_TOLERANCE: float = 1e-8

    # Related to verbosity
    PRINT_EVERY: int = 1000
    



@dataclass
class SupportedCustomInitializers:
    """
    Besides these, all functions from the
    flax.linen.initializers module are supported.
    """
    pass


@dataclass
class SupportedCustomOptimizerSchedules:
    """
    Besides these, all functions from the
    optax.schedules module are supported.
    """
    pass


@dataclass
class SupportedActivations:
    tanh: Callable = nn.tanh
    sigmoid: Callable = nn.sigmoid
    silu: Callable = nn.silu
    swish: Callable = nn.silu
    sin: Callable = jax.jit(jnp.sin)
    cos: Callable = jax.jit(jnp.cos)
    wavelet: Callable = WaveletActivation(1.0)


@dataclass
class SupportedOptimizers:
    adam: Callable = optax.adam
    adamw: Callable = optax.adamw
    set_to_zero: Callable = optax.set_to_zero


@dataclass
class SupportedEquations:
    """
    Class for supported equations. Not in use yet.
    """
    pass


@dataclass
class SupportedSamplingDistributions:
    uniform: Callable = jax.random.uniform


@dataclass
class VerbositySettings(Settings):
    init: bool = True
    training: bool = True
    evaluation: bool = True
    sampling: bool = True
    data: bool = True


@dataclass
class LoggingSettings(Settings):
    do_logging: bool = True
    log_every: int | None = None
    print_every: int | None = None


@dataclass
class DirectorySettings(Settings):
    base_dir: pathlib.Path
    figure_dir: pathlib.Path | None = None
    model_dir: pathlib.Path | None = None
    image_dir: pathlib.Path | None = None
    log_dir: pathlib.Path | None = None
    data_dir: pathlib.Path | None = None
    settings_path: pathlib.Path | None = None


@dataclass
class WeightSchemeSettings:
    normalized: bool = False


@dataclass
class AdaptiveWeightSchemeSettings(WeightSchemeSettings):
    running_average: float | None = None
    update_every: int = 50
    loss_weighted: bool = False


@dataclass(kw_only=True)
class WeightedSettings(WeightSchemeSettings):
    weights: jax.Array


@dataclass(kw_only=True)
class UnweightedSettings(WeightSchemeSettings):
    pass


@dataclass(kw_only=True)
class GradNormSettings(AdaptiveWeightSchemeSettings):
    pass


@dataclass(kw_only=True)
class SoftAdaptSettings(AdaptiveWeightSchemeSettings):
    order: int = 1
    beta: float = 0.1
    normalized_rates: bool = False
    delta_time: float | None = None
    shift_by_max_val: bool = True


@dataclass(kw_only=True)
class TrainingSettings(Settings):
    sampling: dict | None = None
    iterations: int = 1000
    optimizer: Callable = SupportedOptimizers.adam
    update_scheme: str = "unweighted"
    update_kwargs: dict | None = None
    update_settings: WeightSchemeSettings = field(default_factory=lambda: UnweightedSettings())
    learning_rate: float = 1e-3
    batch_size: int | None = None
    decay_rate: float | None = None
    decay_steps: int | None = None
    transfer_learning: bool = False
    checkpoint_every: int | None = None
    resampling: dict | None = None
    jitted_update: bool = True


@dataclass
class EvaluationSettings(Settings):
    sampling: dict | None = None
    error_metric: str = "L2-rel"
    transfer_learning: bool = False


@dataclass
class PlottingSettings(Settings):
    do_plots: bool = False
    plot_every: int | None = None
    overwrite: bool = False
    file_extension: str = "png"
    kwargs: dict | None = None


@dataclass
class MLPSettings(Settings):
    name: str = "MLP"
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: Callable | list[Callable] = SupportedActivations.tanh
    initialization: Callable | list[Callable] = nn.initializers.glorot_normal
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False


@dataclass
class ResNetBlockSettings(Settings):
    name: str = "ResNetBlock"
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: Callable | list[Callable] = SupportedActivations.tanh
    initialization: Callable | list[Callable] = nn.initializers.glorot_normal
    pre_act: Callable | None = None
    post_act: Callable | None = SupportedActivations.tanh
    shortcut_init: Callable | None = None


# def log_settings(settings_dict: dict,
#                  log_dir: pathlib.Path,
#                  *,
#                  tensorboard: bool = False,
#                  text_file: bool = False,
#                  empty_dir: bool = False
#                  ) -> None:
#     """
#     Logs JSON file of settings in Tensorboard and/or a text file.
#     """

#     if text_file:
#         #TODO
#         raise NotImplementedError("Logging to a text file is not supported yet.")

#     # Function for JSON formatting
#     def pretty_json(hp):
#         json_hp = json.dumps(hp, indent=2)
#         return "".join("\t" + line for line in json_hp.splitlines(True))
    
#     if empty_dir:
#         os.system(f"rm -rf {log_dir}/*")
#         #TODO uncomment three below lines
#     # writer = SummaryWriter(log_dir=log_dir)
#     # writer.add_text("settings.json", pretty_json(settings_dict))
#     # writer.close()
#     return


def settings2dict(settings: Settings) -> dict:
    return settings.__dict__