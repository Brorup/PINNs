"""
This module contains relevant code for the specific problem of
determining the stress throughout a plate with a hole in it
subject to boundary stress.
"""

from .analytic import (
    cart_stress_true,
    polar_stress_true
)
from .loss import (
    loss_rect,
    loss_circ_rr_rt,
    loss_dirichlet
)
from .pinn import PlateWithHolePINN
from .plotting import (
    plot_loss,
    plot_stress,
    plot_polar_stress
)