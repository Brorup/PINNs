from dataclasses import dataclass
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from setup.parsers import (
    parse_MLP_settings,
    parse_ResNetBlock_settings
)

from utils.transforms import xy2r, xy2theta, xy2rtheta


class FactDense(nn.Module):
    # Source: https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/archs.py
    features: int
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.zeros
    reparam: None | dict = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


class MLP(nn.Module):
    """
    A classic multilayer perceptron, also known as
    a feed-forward neural network (FNN).
    
    Args:
        name: ................. The name of the network.
        input_dim: ............ The input dimension of the network.
        output_dim: ........... The output dimension of the network.
        hidden_dims: .......... A sequence containing the number of
                                hidden neurons in each layer.
        activation: ........... A sequence containing the activation
                                functions for each layer. Must be
                                the same length as hidden_dims.
        initialization: ....... A sequence of the initializations for
                                the weight matrices. Length must
                                be len(hidden_dims) + 1.
    """
    
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False

    @nn.compact
    def __call__(self, input, transform = None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input
        
        if self.nondim:
            x = x / self.nondim
        
        if self.polar:
            r = jnp.linalg.norm(x, axis=-1)
            theta = jnp.arctan2(x[1], x[0])
            x = jnp.concatenate((x, jnp.array([r, jnp.cos(2*theta), jnp.sin(2*theta)])))
            
        if self.only_polar:
            r = jnp.linalg.norm(x, axis=-1)
            theta = jnp.arctan2(x[1], x[0])
            x = jnp.array([r, jnp.cos(2*theta), jnp.sin(2*theta)])

        if self.embed:
            x = FourierEmbedding(self.embed["embed_scale"], self.embed["embed_dim"])(x)
        
        for i, feats in enumerate(self.hidden_dims):
            x = FactDense(features=feats,
                          kernel_init=self.initialization[i](),
                          name=f"{self.name}_linear{i}",
                          reparam=self.reparam)(x)
            x = self.activation[i](x)
        
        x = FactDense(features=self.output_dim,
                      kernel_init=self.initialization[-1](),
                      name=f"{self.name}_linear_output",
                      reparam=self.reparam)(x)
        
        if self.activate_last_layer:
            x = self.activation[i](x)
        
        return x

    def __str__(self):
        s = f"\n"
        s += f"name:             {self.name}\n"
        s += f"input_dim:        {self.input_dim}\n"
        s += f"output_dim:       {self.output_dim}\n"
        s += f"hidden_dims:      {self.hidden_dims}\n"
        s += f"activation:       {[f.__name__ for f in self.activation]}\n"
        s += f"initialization:   {[f.__name__ for f in self.initialization]}\n"
        s += f"\n"
        return s


class PoissonAnsatzMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False

    @nn.compact
    def __call__(self, input, transform = None):
        y = input
        x = MLP(name=self.name+"_phi", 
                input_dim=self.input_dim, 
                output_dim=self.output_dim, 
                hidden_dims=self.hidden_dims, 
                activation=self.activation,
                initialization=self.initialization,
                embed=self.embed,
                reparam=self.reparam,
                nondim=self.nondim,
                polar=self.polar
                )(input, transform=transform)
        ansatz = (y[0]-jnp.pi)*y[0]*(y[1]-jnp.pi)*y[1]
        return x*ansatz/((jnp.pi**2)*0.25)**2


class BiharmonicAnsatzMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False

    @nn.compact
    def __call__(self, input, transform = None):
        y = input
        x = MLP(name=self.name+"_nn", 
                input_dim=self.input_dim, 
                output_dim=self.output_dim, 
                hidden_dims=self.hidden_dims, 
                activation=self.activation,
                initialization=self.initialization,
                embed=self.embed,
                reparam=self.reparam,
                nondim=self.nondim,
                polar=self.polar,
                only_polar=self.only_polar
                )(input, transform=transform)
        res = MLP(name=self.name+"_res",
                  input_dim=self.input_dim,
                  output_dim=self.output_dim,
                  hidden_dims=[16 for _ in self.hidden_dims],
                  activation=self.activation,
                  initialization=self.initialization,
                  embed=self.embed,
                  reparam=self.reparam,
                  nondim=self.nondim,
                  polar=self.polar,
                  only_polar=self.only_polar
                  )(input, transform=transform)

        # Constrain network to 0 on boundary of [-10, 10] x [-10, 10]
        C_xx = (0.1*y[0]-1.)**3 * (0.1*y[0]+1.)**3
        C_yy = (0.1*y[1]-1.)**3 * (0.1*y[1]+1.)**3

        # Add function such that dyy = 10, and add residual net
        return x*C_xx*C_yy + 5 * y[1]**2 + res


class DoubleMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False
        
    @nn.compact
    def __call__(self, input, transform = None):
        phi = MLP(name=self.name+"_phi", 
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        
        psi = MLP(name=self.name+"_psi",
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        return jnp.array([phi, psi])
    
    
class DoubleAnsatzMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False
        
    @nn.compact
    def __call__(self, input, transform = None):
        phi = BiharmonicAnsatzMLP(name=self.name+"_phi", 
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        
        psi = MLP(name=self.name+"_psi",
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        return jnp.array([phi, psi])


class ModifiedMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False

    @nn.compact
    def __call__(self, input, transform = None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input
        
        if self.nondim:
            x = x / self.nondim
        
        if self.polar:
            r = jnp.linalg.norm(x, axis=-1)
            theta = jnp.arctan2(x[1], x[0])
            x = jnp.concatenate((x, jnp.array([r, jnp.cos(2*theta), jnp.sin(2*theta)])))
            
        if self.only_polar:
            r = jnp.linalg.norm(x, axis=-1)
            theta = jnp.arctan2(x[1], x[0])
            x = jnp.array([r, jnp.cos(2*theta), jnp.sin(2*theta)])

        if self.embed:
            x = FourierEmbedding(self.embed["embed_scale"], self.embed["embed_dim"])(x)
        
        u = FactDense(features=self.hidden_dims[0],
                      kernel_init=self.initialization[0](),
                      name=f"{self.name}_u",
                      reparam={"type": "weight_fact", "mean": 0.5, "stddev": 0.1})(x)
        v = FactDense(features=self.hidden_dims[0],
                      kernel_init=self.initialization[0](),
                      name=f"{self.name}_v",
                         reparam={"type": "weight_fact", "mean": 0.5, "stddev": 0.1})(x)
        
        u = self.activation[0](u)
        v = self.activation[0](v)

        for i, feats in enumerate(self.hidden_dims):
            x = FactDense(features=feats,
                          kernel_init=self.initialization[i](),
                          name=f"{self.name}_linear{i}",
                          reparam=self.reparam)(x)
            x = self.activation[i](x)
            x = x*u  + (1-x)*v
        
        x = FactDense(features=self.output_dim,
                      kernel_init=self.initialization[-1](),
                      name=f"{self.name}_linear_output",
                      reparam=self.reparam)(x)
        
        return x


class DoubleModifiedMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False
        
    @nn.compact
    def __call__(self, input, transform = None):
        phi = ModifiedMLP(name=self.name+"_phi", 
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        
        psi = MLP(name=self.name+"_psi",
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        return jnp.array([phi, psi])


class BiharmonicModifiedAnsatzMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False

    @nn.compact
    def __call__(self, input, transform = None):
        y = input
        x = ModifiedMLP(name=self.name+"_nn", 
                input_dim=self.input_dim, 
                output_dim=self.output_dim, 
                hidden_dims=self.hidden_dims, 
                activation=self.activation,
                initialization=self.initialization,
                embed=self.embed,
                reparam=self.reparam,
                nondim=self.nondim,
                polar=self.polar,
                only_polar=self.only_polar
                )(input, transform=transform)
        res = MLP(name=self.name+"_res",
                  input_dim=self.input_dim,
                  output_dim=self.output_dim,
                  hidden_dims=[16 for _ in self.hidden_dims],
                  activation=self.activation,
                  initialization=self.initialization,
                  embed=self.embed,
                  reparam=self.reparam,
                  nondim=self.nondim,
                  polar=self.polar,
                  only_polar=self.only_polar
                  )(input, transform=transform)

        # Constrain network to 0 on boundary of [-10, 10] x [-10, 10]
        C_xx = (0.1*y[0]-1.)**3 * (0.1*y[0]+1.)**3
        C_yy = (0.1*y[1]-1.)**3 * (0.1*y[1]+1.)**3

        # Add function such that dyy = 10, and add residual net
        return x*C_xx*C_yy + 5 * y[1]**2 + res


class DoubleModifiedAnsatzMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False
        
    @nn.compact
    def __call__(self, input, transform = None):
        phi = BiharmonicModifiedAnsatzMLP(name=self.name+"_phi", 
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        
        psi = ModifiedMLP(name=self.name+"_psi",
                           input_dim=self.input_dim, 
                           output_dim=self.output_dim, 
                           hidden_dims=self.hidden_dims, 
                           activation=self.activation,
                           initialization=self.initialization,
                           embed=self.embed,
                           reparam=self.reparam,
                           nondim=self.nondim,
                           polar=self.polar,
                           only_polar=self.only_polar
                           )(input, transform=transform)
        return jnp.array([phi, psi])


class ResNetBlock(nn.Module):
    """
    A ResNet block. This block can be combined with other models.

    This module consists of two paths: A linear one that is just
    an MLP, and a shortcut that adds the input to the MLP output.
    ```
            o---- shortcut ----o
            |                  |
        x --|                  + ---> y
            |                  |
            o------ MLP  ------o
    ```
    The shortcut is an identity mapping if `dim(x) == dim(y)`.
    Else, it is a linear mapping.

    Args:
        name: ................. The name of the network.
        input_dim: ............ The input dimension of the network.
        output_dim: ........... The output dimension of the network.
        hidden_dims: .......... A list containing the number of
                                hidden neurons in each layer.
        activation: ........... A list containing the activation
                                functions for each layer. Must be
                                the same length as hidden_dims.
        initialization: ....... A list of the initializations for
                                the weight matrices. Length must
                                be len(hidden_dims) + 1.
        pre_act: .............. Activation function to be applied
                                to the MLP input before passing it
                                through the MLP.
        post_act: ............. Activation function to be applied
                                to the output, i.e. the sum of the
                                MLP output and the shortcut output.
        shortcut_init: ........ Initialization for the shortcut if
                                dim(x) != dim(y). If not specified,
                                the first init function in the MLP
                                will be used.
    """
    
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    pre_act: Callable | None = None
    post_act: Callable | None = None
    shortcut_init: Callable | None = None
    reparam: dict | None = None

    def setup(self):
        if self.input_dim != self.output_dim:
            if self.shortcut_init is None:
                self.shortcut_init = self.initialization[0]
        else:
            self.shortcut_init = None
        return

    @nn.compact
    def __call__(self, input):
        
        # MLP input
        x = input
    
        # Shortcut input
        y = input

        # If dimensions match, use identity mapping:
        # Else use linear map between input and output
        if self.input_dim != self.output_dim:
            y = nn.Dense(features=self.output_dim, kernel_init=self.shortcut_init(), name=f"{self.name}_linear_shortcut")(y)

        if self.pre_act is not None:
            x = self.pre_act(x)

        # for i, feats in enumerate(self.hidden_dims):
        #     x = nn.Dense(features=feats, kernel_init=self.initialization[i](), name=f"{self.name}_linear{i}")(x)
        #     x = self.activation[i](x)
        
        # x = nn.Dense(features=self.output_dim, kernel_init=self.initialization[-1](), name=f"{self.name}_linear_output")(x)
        x = MLP(name=self.name+"_mlp",
                input_dim=self.input_dim, 
                output_dim=self.output_dim, 
                hidden_dims=self.hidden_dims, 
                activation=self.activation,
                initialization=self.initialization,
                embed=None,
                reparam=self.reparam,
                nondim=None,
                polar=None)(x)
        
        # Add the two flows together
        out = x+y
        if self.post_act is not None:
            out = self.post_act(out)
        return out

    def __str__(self):
        s = f"\n"
        s += f"name:             {self.name}\n"
        s += f"input_dim:        {self.input_dim}\n"
        s += f"output_dim:       {self.output_dim}\n"
        s += f"hidden_dims:      {self.hidden_dims}\n"
        s += f"activation:       {[f.__name__ for f in self.activation]}\n"
        s += f"initialization:   {[f.__name__ for f in self.initialization]}\n"
        s += f"pre_activation:   {self.pre_act.__name__ if self.pre_act is not None else str(None)}\n"
        s += f"post_activation:  {self.post_act.__name__ if self.post_act is not None else str(None)}\n"
        if self.shortcut_init is not None:
            print("IF", self.shortcut_init)
        
        s_init = self.shortcut_init.__name__ if self.shortcut_init is not None else self.shortcut_init
        s += f"shortcut_init:    {s_init}\n"
        s += f"\n"
        return s


class ResNet(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False

    @nn.compact
    def __call__(self, input, transform = None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input
        
        if self.nondim:
            x = x / self.nondim
        
        if self.polar:
            r = jnp.linalg.norm(x, axis=-1)
            theta = jnp.arctan2(x[1], x[0])
            x = jnp.concatenate((x, jnp.array([r, jnp.cos(2*theta)])))
        
        if self.embed:
            x = FourierEmbedding(self.embed["embed_scale"], self.embed["embed_dim"])(x)
        


        raise NotImplementedError("ResNet implementation not finished.")


class AiryNet(nn.Module):
    input_dim: int = 2
    param_scale: float = 1.0
    
    @nn.compact
    def __call__(self, xy: jax.Array):
        
        rt = xy2rtheta(xy)

        cos_2t = jnp.cos(2*rt[1])
        sin_2t = jnp.sin(2*rt[1])

        r1 = rt[0]
        r2 = jnp.square(r1)
        r4 = jnp.square(r2)
        rinv2 = jnp.reciprocal(r2)
        log_r = jnp.log(r1)
        r2log_r = jnp.multiply(log_r, r2)

        # p = self.param(
        #     "coeff", nn.initializers.normal(self.param_scale), (12,)
        # )
        
        # u_r = jnp.dot(p[:-1], jnp.array([r1, r2, r3, r4, rinv1, rinv2, rinv3, rinv4, log_r, rlog_r, r2log_r]))
        # u_t = jnp.add(cos_2t, p[-1])
        # return jnp.multiply(u_r, u_t)
        p = self.param(
            "coeff", nn.initializers.normal(self.param_scale), (8,)
        )
        phibasis = jnp.array([r2log_r, r2, log_r, 1, r2*cos_2t, r4*cos_2t, rinv2*cos_2t, cos_2t])
        return jnp.dot(p, phibasis)


class FourierEmbedding(nn.Module):
    # Source: https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/archs.py
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", nn.initializers.normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        # y = jnp.cos(jnp.dot(x, kernel))
        return y


class DualMLP(nn.Module):
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    embed: dict | None = None
    reparam: dict | None = None
    nondim: float | None = None
    polar: bool = False
    only_polar: bool = False
    activate_last_layer: bool = False
    
    @nn.compact
    def __call__(self, xy):
        rt = xy2rtheta(xy)
        cos_2t = jnp.cos(2*rt[1])
        sin_2t = jnp.sin(2*rt[1])
        rcossin2t = jnp.array([rt[0], cos_2t, sin_2t])
        sym = MLP(name="sym_mlp", input_dim=1, output_dim=1, hidden_dims=self.hidden_dims,
                  activation=self.activation, initialization=self.initialization)(rt[0].reshape((1,)))
        nonsym = MLP(name="nonsym_mlp", input_dim=3, output_dim=1, hidden_dims=self.hidden_dims,
                     activation=self.activation, initialization=self.initialization)(rcossin2t)
        return jnp.add(sym, nonsym)


def netmap(model: Callable[..., jax.Array], **kwargs) -> Callable[..., jax.Array]:
    """
    Applies the jax.vmap function with in_axes=(None, 0).
    """
    return jax.vmap(model, in_axes=(None, 0), **kwargs)


def deeponetmap(model: Callable[..., jax.Array], **kwargs) -> Callable[..., jax.Array]:
    """
    Applies the jax.vmap function with in_axes=(None, None, 0).
    """
    return jax.vmap(model, in_axes=(None, None, 0), **kwargs)


def setup_network(network_settings: dict[str, str | dict]) -> MLP | ResNetBlock:
    """
    Given a dict of network settings, returns a network instance.
    """
    
    arch = network_settings["architecture"].lower()
    
    match arch:
        
        case "mlp":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return MLP(**parsed_settings)

        case "modifiedmlp":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return ModifiedMLP(**parsed_settings)
        
        case "resnet":
            parsed_settings = parse_ResNetBlock_settings(network_settings["specifications"])
            return ResNetBlock(**parsed_settings)
        
        case "airynet":
            return AiryNet(name="airy")
        
        case "dualmlp":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return DualMLP(**parsed_settings)
        
        case "doublemlp":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return DoubleMLP(**parsed_settings)

        case "doubleansatz":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return DoubleAnsatzMLP(**parsed_settings)

        case "doublemodified":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return DoubleModifiedMLP(**parsed_settings)

        case "ansatz":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return BiharmonicAnsatzMLP(**parsed_settings)
        
        case "modifiedansatz":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return BiharmonicModifiedAnsatzMLP(**parsed_settings)

        case "doublemodifiedansatz":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return DoubleModifiedAnsatzMLP(**parsed_settings)
        
        case _:
            raise ValueError(f"Invalid network architecture: '{arch}'.")
        

def _weight_fact(init_fn, mean, stddev):
    # Source: https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/archs.py
    def init(key, shape):
        key1, key2 = jax.random.split(key)
        w = init_fn(key1, shape)
        g = mean + nn.initializers.normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init