"""
heat_pinn.py
------------
Physics loss computation, training data generation, and validation
for the 1D Heat Equation PINN.

Governing PDE:
    u_t = alpha * u_xx,    x ∈ [-1, 1],  t ∈ [0, 1]

Conditions:
    IC:  u(x, 0) = -sin(π * x)
    BC:  u(-1, t) = u(1, t) = 0

Analytical (Exact) Solution via Separation of Variables:
    u(x, t) = -sin(π * x) * exp(-alpha * π² * t)

This exact solution is used for rigorous quantitative validation (Relative L² Error).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# Type aliases for cleaner signatures
DataTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
DataPair    = Tuple[torch.Tensor, torch.Tensor]

PI = torch.tensor(np.pi, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Solution
# ─────────────────────────────────────────────────────────────────────────────

def analytical_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """Compute the exact (analytical) solution of the 1D heat equation.

    Derived via separation of variables for the specific IC u(x,0) = -sin(πx)
    and homogeneous Dirichlet BCs u(±1, t) = 0.

    The exact solution is:
        u(x, t) = -sin(π * x) * exp(-alpha * π² * t)

    Verification:
        ✓  Satisfies PDE:  u_t = α * u_xx  (by direct substitution)
        ✓  IC:             u(x, 0) = -sin(πx)
        ✓  BC:             u(±1, t) = -sin(±π) * exp(…) = 0

    Args:
        x     (Tensor): Spatial coordinates, shape (N, 1).
        t     (Tensor): Temporal coordinates, shape (N, 1).
        alpha (float):  Thermal diffusivity. Default: 0.01.

    Returns:
        Tensor: Exact temperature field u(x, t), shape (N, 1).
    """
    return -torch.sin(PI * x) * torch.exp(
        torch.tensor(-alpha * np.pi**2, dtype=torch.float32) * t
    )


# ─────────────────────────────────────────────────────────────────────────────
# Physics (PDE) Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_physics_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """Compute the MSE PDE residual loss: (u_t - alpha * u_xx)².

    Uses PyTorch automatic differentiation to compute exact partial derivatives
    of the network output with respect to its inputs (no numerical FD needed).

    Args:
        model (nn.Module): PINN model mapping (x, t) → u(x, t).
        x     (Tensor):    Spatial collocation points, shape (N, 1).
        t     (Tensor):    Temporal collocation points, shape (N, 1).
        alpha (float):     Thermal diffusivity. Default: 0.01.

    Returns:
        Tensor: Scalar mean-squared PDE residual loss.
    """
    # Detach and re-enable gradients w.r.t. inputs
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(x, t)

    # ∂u/∂t  — first-order temporal derivative
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    # ∂u/∂x  — first-order spatial derivative
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    # ∂²u/∂x²  — second-order spatial derivative
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    # PDE residual: u_t − α · u_xx = 0
    residual = u_t - alpha * u_xx
    return torch.mean(residual ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Training Data Generator
# ─────────────────────────────────────────────────────────────────────────────

def get_training_data(
    n_ic:  int = 100,
    n_bc:  int = 100,
    n_pde: int = 5_000,
) -> Tuple[DataTriplet, DataTriplet, DataPair]:
    """Generate training collocation points for IC, BC, and the PDE domain.

    Sampling strategy:
      - IC points: uniform grid on x ∈ [-1, 1] at t = 0
      - BC points: random uniform t ∈ [0, 1] at x = ±1
      - PDE points: uniform random in the full domain [−1,1] × [0,1]

    Note on n_pde default (5000): reduced from 10000 for efficient CPU training
    on mid-range hardware (e.g., AMD Ryzen 5 5500U) while maintaining accuracy.

    Args:
        n_ic  (int): IC sample count. Default: 100.
        n_bc  (int): BC sample count (split equally between x=−1 and x=+1). Default: 100.
        n_pde (int): PDE interior collocation count. Default: 5000.

    Returns:
        Tuple:
            - IC:  (x_ic,  t_ic,  u_ic)   — points where u(x, 0) = −sin(πx)
            - BC:  (x_bc,  t_bc,  u_bc)   — points where u(±1, t) = 0
            - PDE: (x_pde, t_pde)          — interior collocation points
    """
    # ── Initial Condition ──────────────────────────────────────────────────
    # t = 0,  x ∈ [−1, 1],  u(x, 0) = −sin(π x)
    x_ic = torch.linspace(-1, 1, n_ic).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(PI * x_ic)

    # ── Boundary Conditions ────────────────────────────────────────────────
    # x = −1 or +1,  t ∈ [0, 1],  u = 0
    n_half = n_bc // 2
    t_left  = torch.rand(n_half, 1)
    t_right = torch.rand(n_half, 1)

    x_bc = torch.cat([-torch.ones(n_half, 1), torch.ones(n_half, 1)], dim=0)
    t_bc = torch.cat([t_left, t_right], dim=0)
    u_bc = torch.zeros_like(x_bc)

    # ── PDE Collocation Points ─────────────────────────────────────────────
    # Uniformly random over [−1, 1] × [0, 1]
    x_pde = torch.rand(n_pde, 1) * 2.0 - 1.0
    t_pde = torch.rand(n_pde, 1)

    return (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde)


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_relative_l2_error(
    model: nn.Module,
    alpha: float = 0.01,
    n_test: int = 200,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute the Relative L² Error against the analytical solution.

    This is the de-facto standard validation metric in PINN literature
    (Raissi et al., 2019; Cuomo et al., 2022). Reports how close the
    neural network solution is to the exact physics solution.

    Formula:
        Relative L² Error = ‖û − u‖₂ / ‖u‖₂

    A well-trained PINN typically achieves < 1% relative L² error on
    this problem. Values < 0.5% are considered excellent.

    Args:
        model  (nn.Module): Trained PINN model.
        alpha  (float):     Thermal diffusivity. Default: 0.01.
        n_test (int):       Grid resolution per axis (n_test × n_test points).
        device:             Computation device.

    Returns:
        float: Relative L² error (as a decimal, e.g., 0.005 = 0.5%).
    """
    x_arr = np.linspace(-1, 1, n_test)
    t_arr = np.linspace(0,  1, n_test)
    X, T  = np.meshgrid(x_arr, t_arr)

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    model.eval()
    with torch.no_grad():
        u_pred  = model(x_flat, t_flat).cpu()
        u_exact = analytical_solution(x_flat.cpu(), t_flat.cpu(), alpha=alpha)

    numerator   = torch.norm(u_pred - u_exact)
    denominator = torch.norm(u_exact)
    return (numerator / denominator).item()
