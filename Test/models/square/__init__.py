"""
This module contains relevant code for the specific problem of
determining the stress throughout a plate with a hole in it
subject to boundary stress.
"""

from .analytic import (
    cart_stress_true
)
from .loss import (
    loss_rect
)
# from .pinn_old import SQHPINN
from .pinn import SquarePINN
from .plotting import (
    plot_loss,
    plot_stress
)