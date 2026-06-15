<div align="center">

# 🔥 Physics-Informed Neural Networks
### Solving the 1D Heat Equation with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

*A modern deep learning approach to solving partial differential equations by embedding physical laws directly into the neural network training process.*

</div>

---

## 📖 Overview

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **1D Heat Equation** — a classic partial differential equation (PDE) describing how temperature diffuses over time in a one-dimensional domain.

Unlike traditional numerical solvers (FEM, FDM), PINNs leverage **automatic differentiation** to enforce physical constraints during training, requiring no spatial discretization mesh.

### The Heat Equation

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [-1, 1], \quad t \in [0, 1]$$

The model optimizes a **weighted** composite loss:

$$\mathcal{L}_{total} = w_{IC}\,\mathcal{L}_{IC} + w_{BC}\,\mathcal{L}_{BC} + w_{PDE}\,\mathcal{L}_{PDE}$$

| Term | Formula | Default Weight | Purpose |
|------|---------|----------------|---------|
| $\mathcal{L}_{IC}$ | $\frac{1}{N}\sum(\hat{u}(x,0) - u_0)^2$ | $w_{IC}=10$ | Enforces initial condition |
| $\mathcal{L}_{BC}$ | $\frac{1}{N}\sum(\hat{u}(\pm1,t))^2$ | $w_{BC}=10$ | Enforces boundary conditions |
| $\mathcal{L}_{PDE}$ | $\frac{1}{N}\sum(u_t - \alpha u_{xx})^2$ | $w_{PDE}=1$ | Enforces the heat equation |

> **Why weighted loss?** Without balancing, PDE residual loss typically dominates IC/BC
> by orders of magnitude, causing training pathologies. Weights $w_{IC}=w_{BC}=10$
> follow the recommendation of Wang et al. (2021).

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning** | Multi-layer perceptron with Tanh activations |
| ⚡ **Auto Differentiation** | PyTorch autograd for exact PDE derivative computation |
| 📐 **Physics Constraints** | PDE residual embedded directly in the loss function |
| 📊 **Visualization** | Automatic heatmap and loss curve generation |
| 💾 **Model Persistence** | Save & load trained model weights |
| 🖥️ **CLI Support** | Configurable training via command-line arguments |

---

## 🗂️ Project Structure

```
PINN/
│
├── 📄 model.py           # Neural network architecture (MLP with Tanh)
├── 📄 heat_pinn.py       # Physics loss & training data generation
├── 📄 train.py           # Main training script with CLI support
├── 📄 requirements.txt   # Pinned Python dependencies
│
├── 🖼️ loss.png           # Training loss history (generated after training)
├── 🖼️ solution.png       # 2D temperature heatmap (generated after training)
├── 🤖 pinn_heat.pth      # Saved model weights (generated after training)
│
├── 📄 .gitignore         # Git ignore rules
├── 📄 LICENSE            # MIT License
└── 📄 README.md          # This file
```

---

## 🏗️ Architecture

```
Input: [x, t]
    │
    ▼  Xavier (Glorot) normal initialization
┌─────────────────────────────────────────┐
│  Linear(2 → hidden_dim) + Tanh          │  ← Input layer
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Linear(hidden_dim → hidden_dim) + Tanh │  ← Hidden layers (num_hidden_layers)
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Linear(hidden_dim → 1)                 │  ← Output layer (no activation)
└─────────────────────────────────────────┘
    │
    ▼
Output: u(x, t)  ← Predicted temperature

Total trainable parameters: Variable based on config
```

> **Why Tanh?** Tanh is smooth and infinitely differentiable, enabling accurate
> computation of ∂²u/∂x² via autograd. ReLU has zero second derivative almost
> everywhere and is **not suitable** for second-order PDEs like the heat equation.

---

## ✅ Analytical Verification

For this specific problem, the **exact closed-form solution** is known via separation of variables:

$$u_{\text{exact}}(x, t) = -\sin(\pi x) \cdot e^{-\alpha \pi^2 t}$$

Verification:
- ✓ **PDE:** $u_t = -\sin(\pi x)(-\alpha\pi^2)e^{-\alpha\pi^2 t} = \alpha\pi^2\sin(\pi x)e^{-\alpha\pi^2 t} = \alpha u_{xx}$
- ✓ **IC:** $u(x, 0) = -\sin(\pi x)\cdot 1 = -\sin(\pi x)$
- ✓ **BC:** $u(\pm 1, t) = -\sin(\pm\pi)\cdot e^{\ldots} = 0$

After training, the model automatically reports the **Relative L² Error**:

$$\text{Relative } L^2 = \frac{\|\hat{u} - u_{\text{exact}}\|_2}{\|u_{\text{exact}}\|_2}$$

Expected results with default settings:
| Training Config | Rel. L² Error | Training Time (Ryzen 5 5500U) |
|-----------------|---------------|-------------------------------|
| Adam 500ep only | ~2–5% | ~2 min |
| Adam 500 + L-BFGS 200 | ~0.3–1% | ~5–8 min |
| Adam 1000 + L-BFGS 500 | ~0.1–0.5% | ~15–20 min |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.7+
- pip

### 2. Clone & Install

```bash
git clone https://github.com/mfebykhoirusidqi/PINN.git
cd PINN

# Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model

> **Windows users:** Run with UTF-8 encoding for correct output display:
> ```powershell
> $env:PYTHONIOENCODING="utf-8"; python train.py
> ```

**Basic training (defaults optimized for CPU laptop):**
```bash
# Linux / macOS
python train.py

# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"; python train.py
```

**Higher accuracy (longer training):**
```bash
python train.py --adam_epochs 1000 --lbfgs_iters 500 --hidden_dim 64
```

**All available options:**
```
python train.py --help

Reproducibility:
  --seed             INT    Random seed (default: 42)

Model architecture:
  --hidden_dim       INT    Neurons per hidden layer (default: 32)
  --num_hidden_layers INT   Number of hidden layers (default: 3)

Phase 1 — Adam:
  --adam_epochs      INT    Adam epochs (default: 500)
  --lr               FLOAT  Adam learning rate (default: 0.001)

Phase 2 — L-BFGS:
  --lbfgs_iters      INT    L-BFGS outer iterations (default: 200)

Physics:
  --alpha            FLOAT  Thermal diffusivity α (default: 0.01)

Loss weights:
  --w_ic             FLOAT  IC loss weight (default: 10.0)
  --w_bc             FLOAT  BC loss weight (default: 10.0)
  --w_pde            FLOAT  PDE loss weight (default: 1.0)

Collocation points:
  --n_ic             INT    IC points (default: 100)
  --n_bc             INT    BC points (default: 100)
  --n_pde            INT    PDE points (default: 5000, CPU-optimized)
```

---

## 📊 Results

After training, the following files are generated:

### Training Loss History (`loss.png`)

![Loss History](loss.png)

> Shows loss curves for both Adam (Phase 1) and L-BFGS (Phase 2). The vertical
> dashed line marks the optimizer transition. L-BFGS rapidly refines the solution
> to high precision.

### Three-Panel Comparison (`solution.png`)

![PINN Solution](solution.png)

> **Left:** PINN prediction $\hat{u}(x,t)$  
> **Center:** Analytical exact solution $u(x,t) = -\sin(\pi x)e^{-\alpha\pi^2 t}$  
> **Right:** Absolute pointwise error $|\hat{u} - u|$ — shows where the model is least accurate.

### Console Output Example
```
  📐 Validation vs Analytical Solution
  Exact solution:      u(x,t) = -sin(πx) · exp(-α π² t)
  Relative L² Error:   0.4231%  (0.004231)
  Quality grade:       🟢 Excellent  (< 0.5%)
```

---

## 🌐 Interactive Interfaces

To make this project stand out in an academic portfolio, it includes two interactive interfaces. Ensure you have installed the full requirements (`pip install -r requirements.txt`) and have trained a model (`pinn_heat.pth`) before running these.

### 1. Web Dashboard (Streamlit)
A beautiful, interactive web application that lets you use sliders to explore the temperature profile at specific times, view global error metrics, and see the full 2D spatiotemporal evolution.

```bash
# Run the Streamlit app
streamlit run app.py
```
*This will open a new tab in your default web browser.*

### 2. Jupyter Notebook (`demo.ipynb`)
The standard format for academic and data science exploration. The notebook walks step-by-step through loading the model, making predictions on a grid, and rendering the 3-panel comparison plot.

```bash
# Start Jupyter Lab or Notebook
jupyter lab
# Then open demo.ipynb
```

---

## 🔬 How It Works

```
1. Sample training points
       │
       ├─── x_ic, t_ic → u(x,0) = -sin(πx)      [Initial Condition]
       ├─── x_bc, t_bc → u(±1,t) = 0              [Boundary Conditions]
       └─── x_pde, t_pde → residual points         [PDE Domain]
       │
2. Forward pass: neural network predicts u(x,t)
       │
3. Compute derivatives via autograd:
       │    u_t  = ∂u/∂t
       │    u_x  = ∂u/∂x
       │    u_xx = ∂²u/∂x²
       │
4. Evaluate weighted composite loss:
       │    L = w_ic*L_IC + w_bc*L_BC + w_pde*L_PDE
       │
5. Backpropagate and update weights
       │
6. Repeat until convergence ✓
```

---

## 🔄 PINN vs Traditional Methods

| Aspect | FDM / FEM | PINN |
|--------|-----------|------|
| Mesh requirement | ✅ Required | ❌ Mesh-free |
| Data requirements | ✅ Dense grid | ✅ Sparse samples |
| Handles noisy data | ❌ Difficult | ✅ Natural |
| Inverse problems | ❌ Complex | ✅ Straightforward |
| Physics consistency | ✅ Exact (by construction) | ✅ Enforced via loss |
| Generalization | ❌ Per-mesh | ✅ Continuous solution |

---

## 🐍 Python API

### Load Trained Model and Run Inference
```python
from model import PINN
from heat_pinn import analytical_solution, compute_relative_l2_error
import torch

# Load checkpoint
checkpoint = torch.load('pinn_heat.pth')
cfg = checkpoint['config']

# Reconstruct model
model = PINN(
    input_dim=2,
    hidden_dim=cfg['hidden_dim'],
    output_dim=1,
    num_hidden_layers=cfg['num_hidden_layers'],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict at arbitrary points
x = torch.tensor([[0.0], [0.5], [-0.5]], dtype=torch.float32)
t = torch.tensor([[0.5], [0.5], [ 0.5]], dtype=torch.float32)

with torch.no_grad():
    u_pred  = model(x, t)
    u_exact = analytical_solution(x, t, alpha=cfg['alpha'])

print(f"Prediction:  {u_pred.T}")
print(f"Exact:       {u_exact.T}")

# Recompute validation error
rel_l2 = compute_relative_l2_error(model, alpha=cfg['alpha'])
print(f"Rel. L² Error: {rel_l2*100:.4f}%")
```

---

## ⚙️ Hyperparameter Guide

| Parameter | Default | Notes |
|-----------|---------|-------|
| `adam_epochs` | 500 | Phase 1 fast convergence |
| `lbfgs_iters` | 200 | Phase 2 high-precision fine-tuning |
| `lr` | 0.001 | Adam learning rate |
| `hidden_dim` | 32 | More neurons → more expressive |
| `num_hidden` | 3 | More layers → captures complexity |
| `n_pde` | 5000 | CPU-optimized default; increase for GPU |
| `w_ic` / `w_bc` | 10.0 | Higher → stricter IC/BC enforcement |
| `alpha` | 0.01 | Thermal diffusivity of the material |

---

## 📦 Dependencies

```
torch>=1.9.0         # Deep learning + autograd
numpy>=1.19.0        # Numerical computation
matplotlib>=3.3.0    # Plotting and visualization
scipy>=1.5.0         # Scientific computing utilities
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧩 Extending This Project

Want to solve a **different PDE**? Just modify `heat_pinn.py`:

- **Wave Equation:** Change PDE residual to `u_tt - c²·u_xx = 0`
- **Burgers' Equation:** `u_t + u·u_x - ν·u_xx = 0`
- **2D Poisson:** `u_xx + u_yy = f(x,y)`

You only need to update:
1. `compute_physics_loss()` — for the new PDE
2. `get_training_data()` — for new IC/BC

---

## 📚 References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

2. **Cuomo, S. et al. (2022)**. Scientific machine learning through physics-informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92(3), 88. [doi:10.1007/s10915-022-01939-z](https://doi.org/10.1007/s10915-022-01939-z)

3. **Wang, S., Teng, Y., & Perdikaris, P. (2021)**. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081. *(Loss weighting motivation)*

4. [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
5. [Heat Equation — Wikipedia](https://en.wikipedia.org/wiki/Heat_equation)

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ and Physics

⭐ **If this helped you, consider giving it a star!** ⭐

</div>
