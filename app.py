import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from model import PINN
from heat_pinn import analytical_solution, compute_relative_l2_error

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PINN: 1D Heat Equation",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        background-color: #e8f4f8;
        border-color: #bee5eb;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached Data & Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path="pinn_heat.pth"):
    """Load the trained PINN model and configuration."""
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    cfg = checkpoint.get('config', {
        'hidden_dim': 32,
        'num_hidden_layers': 3,
        'alpha': 0.01,
        'seed': 42
    })
    
    model = PINN(
        input_dim=2,
        hidden_dim=cfg['hidden_dim'],
        output_dim=1,
        num_hidden_layers=cfg.get('num_hidden_layers', 3) # Fallback for older checkpoints
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions for Plotting
# ─────────────────────────────────────────────────────────────────────────────
def generate_data(t_val, nx=100):
    """Generate spatial data at a specific time t."""
    x = np.linspace(-1, 1, nx)
    t = np.ones_like(x) * t_val
    
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    
    return x, x_tensor, t_tensor

def plot_temperature_profile(x, u_pred, u_exact, t_val):
    """Plot the 1D temperature profile at a specific time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(x, u_exact, 'k--', linewidth=2.5, label='Analytical (Exact)', alpha=0.7)
    ax.plot(x, u_pred, 'r-', linewidth=2, label='PINN Prediction')
    
    # Fill error area
    ax.fill_between(x, u_pred, u_exact, color='red', alpha=0.2, label='Absolute Error')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.1, 0.1])
    ax.set_xlabel('Position (x)', fontsize=12)
    ax.set_ylabel('Temperature (u)', fontsize=12)
    ax.set_title(f'Temperature Profile at Time $t = {t_val:.2f}$', fontsize=14, fontweight='bold')
    
    ax.legend(loc='lower center', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_spatiotemporal_heatmap(model, alpha, nx=100, nt=100):
    """Plot the full 2D spatiotemporal heatmap comparison."""
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
    t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).numpy().reshape(nt, nx)
        u_exact = analytical_solution(x_tensor, t_tensor, alpha=alpha).numpy().reshape(nt, nx)
    
    error = np.abs(u_pred - u_exact)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = u_exact.min(), u_exact.max()
    
    # Prediction
    cf0 = axes[0].pcolormesh(T, X, u_pred, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('PINN Prediction $\\hat{u}(x,t)$', fontsize=14)
    fig.colorbar(cf0, ax=axes[0])
    
    # Exact
    cf1 = axes[1].pcolormesh(T, X, u_exact, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Analytical Exact $u(x,t)$', fontsize=14)
    fig.colorbar(cf1, ax=axes[1])
    
    # Error
    cf2 = axes[2].pcolormesh(T, X, error, cmap='inferno', shading='auto', vmin=0, vmax=error.max())
    axes[2].set_title('Absolute Error $|\\hat{u} - u|$', fontsize=14)
    fig.colorbar(cf2, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('Time (t)', fontsize=12)
        ax.set_ylabel('Position (x)', fontsize=12)
        
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Main UI Construction
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🔥 Physics-Informed Neural Networks (PINNs)")
    st.markdown("### Solving the 1D Heat Equation with Deep Learning")
    
    st.markdown("""
    This interactive dashboard demonstrates a PINN trained to solve the 1D Heat Equation:
    $$ \\frac{\\partial u}{\\partial t} = \\alpha \\frac{\\partial^2 u}{\\partial x^2} $$
    
    The neural network learns the solution $u(x,t)$ by minimizing the physical PDE residual along with initial and boundary conditions, **without requiring any simulation data or spatial grids.**
    """)
    
    # Load Model
    model, cfg = load_model()
    
    if model is None:
        st.error("⚠️ Model checkpoint (`pinn_heat.pth`) not found. Please train the model first by running `python train.py`.")
        return
    
    alpha = cfg['alpha']
    
    # Sidebar Controls
    with st.sidebar:
        st.header("🎛️ Interactive Controls")
        st.markdown("Explore the temperature distribution at different points in time.")
        
        t_slider = st.slider(
            "Time (t)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.01,
            help="Drag to see how the temperature diffuses over time."
        )
        
        st.divider()
        st.header("📊 Model Information")
        st.info(f"""
        **Architecture:**
        - Input: $(x, t)$
        - Hidden: {cfg['hidden_dim']} neurons $\\times$ {cfg['num_hidden_layers']} layers
        - Activation: $\\text{{Tanh}}$
        - Params: {model.count_parameters():,}
        
        **Physics:**
        - Diffusivity $\\alpha$: {alpha}
        - IC: $u(x,0) = -\\sin(\\pi x)$
        - BC: $u(\\pm 1, t) = 0$
        """)
        
        st.markdown("---")
        st.markdown("Built for S2 Physics Portfolio Demonstration.")

    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⏱️ 1D Temperature Profile (Time Slice)")
        
        # Generate and plot 1D slice
        x, x_tensor, t_tensor = generate_data(t_slider)
        
        with torch.no_grad():
            u_pred = model(x_tensor, t_tensor).numpy().flatten()
            u_exact = analytical_solution(x_tensor, t_tensor, alpha=alpha).numpy().flatten()
            
        fig_1d = plot_temperature_profile(x, u_pred, u_exact, t_slider)
        st.pyplot(fig_1d)
        
        # Current Error Metric
        mae = np.mean(np.abs(u_pred - u_exact))
        st.metric("Mean Absolute Error (at current $t$)", f"{mae:.6f}")

    with col2:
        st.subheader("📏 Global Validation Metrics")
        
        # Calculate global relative L2 error
        rel_l2 = compute_relative_l2_error(model, alpha=alpha)
        
        st.markdown("""
        The academic standard for evaluating PINN accuracy is the **Relative $L^2$ Error** computed over the entire spatiotemporal domain $[x, t]$.
        """)
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Global Relative L² Error", f"{rel_l2 * 100:.4f} %")
        with col_metric2:
            if rel_l2 < 0.005:
                st.success("🟢 Excellent Accuracy")
            elif rel_l2 < 0.01:
                st.info("🟡 Very Good Accuracy")
            else:
                st.warning("🟠 Good Accuracy")
                
        st.markdown("""
        > **Note:** A relative $L^2$ error of $<1\\%$ indicates that the neural network has successfully discovered the underlying physical laws mathematically, matching the analytical separation-of-variables solution almost perfectly.
        """)
        
    st.divider()
    
    # 2D Heatmap Section
    st.subheader("🗺️ 2D Spatiotemporal Evolution")
    st.markdown("Comparing the full neural network prediction against the exact mathematical solution over the entire domain.")
    
    with st.spinner("Generating high-resolution heatmaps..."):
        fig_2d = plot_spatiotemporal_heatmap(model, alpha)
        st.pyplot(fig_2d)

if __name__ == "__main__":
    main()
