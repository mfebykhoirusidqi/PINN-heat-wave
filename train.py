"""
train.py
--------
Main training script for the PINN Heat Equation solver.

Implements the standard hybrid optimization strategy from PINN literature:
    Phase 1 — Adam optimizer   : fast convergence from random initialization
    Phase 2 — L-BFGS optimizer : high-precision fine-tuning (second-order method)

Defaults are tuned for efficient CPU training on mid-range hardware
(e.g., AMD Ryzen 5 5500U, 8 GB RAM) — typical training time: 5–10 minutes.

Usage:
    python train.py                                          # default settings
    python train.py --adam_epochs 1000 --lbfgs_iters 300   # more training
    python train.py --help                                   # all options
"""

import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt

from model import PINN
from heat_pinn import (
    compute_physics_loss,
    get_training_data,
    analytical_solution,
    compute_relative_l2_error,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "PINN solver for the 1D Heat Equation.\n"
            "Uses hybrid Adam → L-BFGS training (standard in PINN literature).\n"
            "Defaults optimized for CPU training on mid-range hardware."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Reproducibility ──────────────────────────────────────────────────────
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for full reproducibility")

    # ── Model architecture ───────────────────────────────────────────────────
    parser.add_argument("--hidden_dim",   type=int,   default=32,
                        help="Neurons per hidden layer")
    parser.add_argument("--num_hidden",   type=int,   default=3,
                        help="Number of hidden layers")

    # ── Phase 1: Adam optimizer ──────────────────────────────────────────────
    parser.add_argument("--adam_epochs",  type=int,   default=500,
                        help="Adam training epochs (Phase 1)")
    parser.add_argument("--lr",           type=float, default=1e-3,
                        help="Adam learning rate")

    # ── Phase 2: L-BFGS optimizer ────────────────────────────────────────────
    parser.add_argument("--lbfgs_iters",  type=int,   default=200,
                        help="L-BFGS outer iterations (Phase 2). "
                             "Each performs up to --lbfgs_max_iter line searches.")
    parser.add_argument("--lbfgs_max_iter", type=int, default=20,
                        help="L-BFGS max line-search iterations per step")

    # ── Physics ──────────────────────────────────────────────────────────────
    parser.add_argument("--alpha",        type=float, default=0.01,
                        help="Thermal diffusivity constant (α)")

    # ── Loss weights ─────────────────────────────────────────────────────────
    parser.add_argument("--w_ic",         type=float, default=10.0,
                        help="Weight for Initial Condition loss")
    parser.add_argument("--w_bc",         type=float, default=10.0,
                        help="Weight for Boundary Condition loss")
    parser.add_argument("--w_pde",        type=float, default=1.0,
                        help="Weight for PDE residual loss")

    # ── Collocation points ───────────────────────────────────────────────────
    parser.add_argument("--n_ic",         type=int,   default=100,
                        help="Number of IC collocation points")
    parser.add_argument("--n_bc",         type=int,   default=100,
                        help="Number of BC collocation points")
    parser.add_argument("--n_pde",        type=int,   default=5000,
                        help="Number of PDE interior collocation points "
                             "(5000 default optimized for CPU)")

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir",   type=str,   default=".",
                        help="Directory for saved plots and model")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Loss Helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
    alpha, w_ic, w_bc, w_pde,
):
    """Compute weighted composite loss: w_ic·L_IC + w_bc·L_BC + w_pde·L_PDE."""
    loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
    loss_bc  = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
    loss_pde = compute_physics_loss(model, x_pde, t_pde, alpha=alpha)
    total    = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde
    return total, loss_ic, loss_bc, loss_pde


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Adam Training
# ─────────────────────────────────────────────────────────────────────────────

def train_adam(
    model, data, epochs, lr, alpha, w_ic, w_bc, w_pde, device,
) -> list:
    """Train with Adam optimizer (Phase 1: fast convergence).

    Args:
        model:   PINN model.
        data:    Tuple of (IC, BC, PDE) tensors.
        epochs:  Number of Adam steps.
        lr:      Learning rate.
        alpha:   Thermal diffusivity.
        w_ic/bc/pde: Loss weights.
        device:  Torch device.

    Returns:
        list: Loss history per epoch [total, ic, bc, pde].
    """
    (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde) = data

    optimizer  = optim.Adam(model.parameters(), lr=lr)
    history    = []
    log_every  = max(1, epochs // 10)
    start      = time.time()

    print(f"\n{'─'*60}")
    print(f"  Phase 1 — Adam Optimizer  ({epochs} epochs, lr={lr})")
    print(f"{'─'*60}")

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, l_ic, l_bc, l_pde = compute_total_loss(
            model, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
            alpha, w_ic, w_bc, w_pde,
        )
        loss.backward()
        optimizer.step()

        history.append([loss.item(), l_ic.item(), l_bc.item(), l_pde.item()])

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  [{epoch:>{len(str(epochs))}}/{epochs}]  "
                f"Loss={loss.item():.3e}  "
                f"IC={l_ic.item():.3e}  BC={l_bc.item():.3e}  "
                f"PDE={l_pde.item():.3e}  "
                f"[{time.time()-start:.1f}s]"
            )

    print(f"  ✓ Adam done in {time.time()-start:.1f}s\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — L-BFGS Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────

def train_lbfgs(
    model, data, n_iters, max_iter_per_step,
    alpha, w_ic, w_bc, w_pde, device,
) -> list:
    """Fine-tune with L-BFGS optimizer (Phase 2: high-precision convergence).

    L-BFGS approximates the Hessian to navigate ill-conditioned loss landscapes
    common in PDE residuals. This is the standard second-order optimizer for
    PINNs (Raissi et al., 2019; Wang et al., 2021).

    Args:
        model:               PINN model.
        data:                Tuple of (IC, BC, PDE) tensors.
        n_iters:             Number of outer L-BFGS calls.
        max_iter_per_step:   Max line-search iterations per call.
        alpha:               Thermal diffusivity.
        w_ic/bc/pde:         Loss weights.
        device:              Torch device.

    Returns:
        list: Loss history per iteration [total, ic, bc, pde].
    """
    (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde) = data

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter_per_step,
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )

    history   = []
    log_every = max(1, n_iters // 10)
    start     = time.time()
    step_log  = [0.0, 0.0, 0.0, 0.0]  # shared state for closure logging

    print(f"{'─'*60}")
    print(f"  Phase 2 — L-BFGS Optimizer  ({n_iters} iterations, "
          f"max_iter={max_iter_per_step})")
    print(f"{'─'*60}")

    model.train()
    for it in range(1, n_iters + 1):

        def closure():
            optimizer.zero_grad()
            loss, l_ic, l_bc, l_pde = compute_total_loss(
                model,
                x_ic, t_ic, u_ic,
                x_bc, t_bc, u_bc,
                x_pde, t_pde,
                alpha, w_ic, w_bc, w_pde,
            )
            loss.backward()
            step_log[0] = loss.item()
            step_log[1] = l_ic.item()
            step_log[2] = l_bc.item()
            step_log[3] = l_pde.item()
            return loss

        optimizer.step(closure)
        history.append(step_log.copy())

        if it % log_every == 0 or it == 1:
            print(
                f"  [{it:>{len(str(n_iters))}}/{n_iters}]  "
                f"Loss={step_log[0]:.3e}  "
                f"IC={step_log[1]:.3e}  BC={step_log[2]:.3e}  "
                f"PDE={step_log[3]:.3e}  "
                f"[{time.time()-start:.1f}s]"
            )

    print(f"  ✓ L-BFGS done in {time.time()-start:.1f}s\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, alpha, device) -> None:
    """Compute and print quantitative validation metrics.

    Computes the Relative L² Error against the analytical exact solution.
    This is the standard benchmark metric in PINN literature.

    A well-trained PINN typically achieves:
        - Good:      Rel. L² Error < 5%
        - Very good: Rel. L² Error < 1%
        - Excellent: Rel. L² Error < 0.5%
    """
    rel_l2 = compute_relative_l2_error(model, alpha=alpha, n_test=200, device=device)
    pct    = rel_l2 * 100

    print(f"{'═'*60}")
    print(f"  📐 Validation vs Analytical Solution")
    print(f"{'═'*60}")
    print(f"  Exact solution:      u(x,t) = -sin(πx) · exp(-α π² t)")
    print(f"  Relative L² Error:   {pct:.4f}%  ({rel_l2:.6f})")

    if pct < 0.5:
        grade = "🟢 Excellent  (< 0.5%)"
    elif pct < 1.0:
        grade = "🟡 Very good  (< 1.0%)"
    elif pct < 5.0:
        grade = "🟠 Good       (< 5.0%)"
    else:
        grade = "🔴 Poor       (≥ 5.0%) — consider more training"

    print(f"  Quality grade:       {grade}")
    print(f"{'═'*60}\n")
    return rel_l2


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(
    adam_history:  list,
    lbfgs_history: list,
    output_path:   str = "loss.png",
) -> None:
    """Plot training loss history for both Adam and L-BFGS phases.

    Args:
        adam_history:  List of [total, ic, bc, pde] per Adam epoch.
        lbfgs_history: List of [total, ic, bc, pde] per L-BFGS iteration.
        output_path:   Save path for the figure.
    """
    adam_arr  = np.array(adam_history)
    lbfgs_arr = np.array(lbfgs_history) if lbfgs_history else None
    n_adam    = len(adam_arr)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = {"total": "#2C3E50", "ic": "#E74C3C", "bc": "#2980B9", "pde": "#27AE60"}

    # Adam phase
    ax.semilogy(adam_arr[:, 0], color=colors["total"], lw=2,   label="Total (Adam)")
    ax.semilogy(adam_arr[:, 1], color=colors["ic"],    lw=1.2, label="IC Loss",  ls="--")
    ax.semilogy(adam_arr[:, 2], color=colors["bc"],    lw=1.2, label="BC Loss",  ls=":")
    ax.semilogy(adam_arr[:, 3], color=colors["pde"],   lw=1.2, label="PDE Loss", ls="-.")

    # L-BFGS phase
    if lbfgs_arr is not None and len(lbfgs_arr) > 0:
        x_shift = np.arange(n_adam, n_adam + len(lbfgs_arr))
        ax.semilogy(x_shift, lbfgs_arr[:, 0], color=colors["total"], lw=2,
                    ls="-", label="Total (L-BFGS)")
        # Vertical separator
        ax.axvline(n_adam, color="gray", lw=1.5, ls="--", alpha=0.7, label="Adam → L-BFGS")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title("PINN Training Loss — Hybrid Adam + L-BFGS", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Loss plot  → {output_path}")


def plot_comparison(
    model:       PINN,
    alpha:       float = 0.01,
    output_path: str   = "solution.png",
) -> None:
    """Generate a 3-panel comparison figure (standard academic format).

    Panels:
        1. PINN prediction  u_pred(x, t)
        2. Analytical exact u_exact(x, t)
        3. Absolute error   |u_pred − u_exact|

    This three-panel layout is the required format for academic publications
    on PDE solvers (including PINN papers).

    Args:
        model:       Trained PINN model.
        alpha:       Thermal diffusivity.
        output_path: Save path for the figure.
    """
    x_arr = np.linspace(-1, 1, 200)
    t_arr = np.linspace(0,  1, 200)
    X, T  = np.meshgrid(x_arr, t_arr)

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        u_pred  = model(x_flat, t_flat).numpy().reshape(200, 200)
        u_exact = analytical_solution(x_flat, t_flat, alpha=alpha).numpy().reshape(200, 200)

    abs_error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "PINN vs Analytical Solution — 1D Heat Equation",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Shared colormap limits for pred & exact
    vmin, vmax = u_exact.min(), u_exact.max()

    titles = ["PINN Prediction  $\\hat{u}(x,t)$",
              "Analytical Exact  $u(x,t)$",
              "Absolute Error  $|\\hat{u} - u|$"]
    data   = [u_pred, u_exact, abs_error]
    cmaps  = ["RdBu_r", "RdBu_r", "inferno"]
    vlims  = [(vmin, vmax), (vmin, vmax), (0, abs_error.max())]

    for ax, dat, title, cmap, (vlo, vhi) in zip(axes, data, titles, cmaps, vlims):
        cf = ax.pcolormesh(T, X, dat, cmap=cmap, shading="auto", vmin=vlo, vmax=vhi)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Time  $t$", fontsize=11)
        ax.set_ylabel("Position  $x$", fontsize=11)
        ax.set_title(title, fontsize=11, pad=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼️  Comparison → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device (CPU for AMD Ryzen — PyTorch ROCm not available on Windows) ─
    device = torch.device("cpu")

    # ── Banner ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  🔥 Physics-Informed Neural Network — Heat Equation")
    print(f"{'═'*60}")
    print(f"  Device    : {device}  (AMD Ryzen optimized defaults)")
    print(f"  Seed      : {args.seed}")
    print(f"  α (alpha) : {args.alpha}   (thermal diffusivity)")
    print(f"  Weights   : IC={args.w_ic}  BC={args.w_bc}  PDE={args.w_pde}")
    print(f"  n_pde     : {args.n_pde:,} collocation points")
    print(f"{'═'*60}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = PINN(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_hidden_layers=args.num_hidden,
    ).to(device)
    print(f"\n  Model: {model}\n")

    # ── Data ───────────────────────────────────────────────────────────────
    data = get_training_data(n_ic=args.n_ic, n_bc=args.n_bc, n_pde=args.n_pde)
    # Move all tensors to device
    data = (
        tuple(t.to(device) for t in data[0]),
        tuple(t.to(device) for t in data[1]),
        tuple(t.to(device) for t in data[2]),
    )

    total_start = time.time()

    # ── Phase 1: Adam ──────────────────────────────────────────────────────
    adam_hist = train_adam(
        model, data,
        epochs=args.adam_epochs,
        lr=args.lr,
        alpha=args.alpha,
        w_ic=args.w_ic, w_bc=args.w_bc, w_pde=args.w_pde,
        device=device,
    )

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────
    lbfgs_hist = train_lbfgs(
        model, data,
        n_iters=args.lbfgs_iters,
        max_iter_per_step=args.lbfgs_max_iter,
        alpha=args.alpha,
        w_ic=args.w_ic, w_bc=args.w_bc, w_pde=args.w_pde,
        device=device,
    )

    total_time = time.time() - total_start
    print(f"  ⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)\n")

    # ── Evaluation ─────────────────────────────────────────────────────────
    rel_l2 = evaluate(model, alpha=args.alpha, device=device)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("  Saving outputs...")
    plot_loss(
        adam_hist, lbfgs_hist,
        output_path=os.path.join(args.output_dir, "loss.png"),
    )
    plot_comparison(
        model,
        alpha=args.alpha,
        output_path=os.path.join(args.output_dir, "solution.png"),
    )

    # ── Save Model ─────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "pinn_heat.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim":        args.hidden_dim,
            "num_hidden_layers": args.num_hidden,
            "alpha":             args.alpha,
            "seed":              args.seed,
            "rel_l2_error":      rel_l2,
        },
    }, model_path)
    print(f"  💾 Model saved → {model_path}")
    print(f"\n  Done! ✅\n")
