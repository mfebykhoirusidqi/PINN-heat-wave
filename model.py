"""
model.py
--------
Physics-Informed Neural Network (PINN) architecture.

The model is a fully-connected multi-layer perceptron (MLP) with Tanh
activations — the standard choice for PINNs because Tanh is smooth and
infinitely differentiable, enabling accurate computation of high-order
partial derivatives via automatic differentiation.

Note on activation choice:
    ReLU and its variants are NOT used here because their second derivatives
    are zero almost everywhere, making the PDE residual loss untrainable
    for equations involving u_xx.

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks. J. Comput. Phys., 378, 686–707.
"""

import torch
import torch.nn as nn
from typing import Optional


class PINN(nn.Module):
    """Multi-layer perceptron for Physics-Informed Neural Networks.

    Maps spatial-temporal coordinates (x, t) to a scalar field u(x, t)
    representing the PDE solution. All hidden layers use Tanh activation.

    Architecture (default num_hidden_layers=3):
        Input(2) → [Linear(hidden) → Tanh] × num_hidden_layers → Linear(1)

    Weight Initialization:
        Xavier (Glorot) normal initialization is applied to all Linear layers.
        This is optimal for Tanh networks and promotes faster convergence
        compared to PyTorch's default Kaiming uniform init.

    Args:
        input_dim        (int): Input dimension. Default: 2 (x and t).
        hidden_dim       (int): Neurons per hidden layer. Default: 32.
        output_dim       (int): Output dimension. Default: 1 (scalar u).
        num_hidden_layers(int): Number of hidden layers. Default: 3.

    Example:
        >>> model = PINN(input_dim=2, hidden_dim=32, output_dim=1, num_hidden_layers=3)
        >>> x = torch.rand(100, 1)
        >>> t = torch.rand(100, 1)
        >>> u = model(x, t)  # shape: (100, 1)
        >>> print(model)
        PINN(input=2, hidden=32×3, output=1, params=3,201)
    """

    def __init__(
        self,
        input_dim:         int = 2,
        hidden_dim:        int = 32,
        output_dim:        int = 1,
        num_hidden_layers: int = 3,
    ) -> None:
        super(PINN, self).__init__()

        # Store config
        self.input_dim         = input_dim
        self.hidden_dim        = hidden_dim
        self.output_dim        = output_dim
        self.num_hidden_layers = num_hidden_layers

        # Build network: Input → [Hidden × Tanh] × N → Output
        layers = []
        layers += [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier (Glorot) normal init — optimal for Tanh activations
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier normal initialization to all Linear layers.

        Xavier initialization sets weight variance to 2 / (fan_in + fan_out),
        preserving the variance of activations across layers. Combined with
        Tanh, this prevents vanishing/exploding gradients during early training.
        Biases are initialized to zero.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the solution u(x, t).

        Args:
            x (Tensor): Spatial coordinate, shape (N, 1).
            t (Tensor): Temporal coordinate, shape (N, 1).

        Returns:
            Tensor: Predicted PDE solution u(x, t), shape (N, 1).
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"PINN("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_dim}×{self.num_hidden_layers}, "
            f"output={self.output_dim}, "
            f"params={self.count_parameters():,})"
        )
