import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Type, Union


class BrainConnectivityMLP(nn.Module):
    """
    MLP whose layer width equals the number of nodes in a brain functional
    connectivity (FC) network.  Each linear layer is NxN, so the FC matrix can
    be used directly as a weight initializer.

    Architecture (example with n_hidden_layers=2):
        Input(N) -> Linear(N,N) -> Act -> Linear(N,N) -> Act -> Linear(N,N) -> [Out Act]

    Args:
        fc_matrix:        Square NxN functional-connectivity matrix (numpy or tensor).
        n_hidden_layers:  Number of hidden layers (>=0). Total linear layers = n_hidden_layers + 1.
        use_fc_init:      If True, every linear layer's weight is initialised from fc_matrix
                          and biases are zeroed.  If False, PyTorch default (Kaiming uniform)
                          is kept.
        activation:       Hidden-layer activation class (default: ReLU).
        output_activation: Optional activation class applied after the last layer.
        keep_ratio:       Fraction of off-diagonal connections to keep (0, 1].
                          Connections are ranked by absolute value; the weakest
                          (1 - keep_ratio) fraction are zeroed before the matrix
                          is used as a weight initialiser.  None means no thresholding.
        n_frozen_layers:  Number of the first N linear layers whose parameters are frozen
                          (requires_grad=False) during training.
        frozen_fc_init:   If True (default), frozen layers are initialised from the FC matrix.
                          If False, frozen layers keep their Kaiming random initialisation.
                          Has no effect when n_frozen_layers=0.
    """

    def __init__(
        self,
        fc_matrix: Union[np.ndarray, torch.Tensor],
        n_hidden_layers: int = 2,
        use_fc_init: bool = True,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
        keep_ratio: Optional[float] = None,
        n_frozen_layers: int = 0,
        frozen_fc_init: bool = True,
    ) -> None:
        super().__init__()

        fc_tensor = self._to_tensor(fc_matrix)
        n = fc_tensor.shape[0]
        if fc_tensor.shape != (n, n):
            raise ValueError(f"fc_matrix must be square NxN, got {fc_tensor.shape}")
        if n_hidden_layers < 0:
            raise ValueError("n_hidden_layers must be >= 0")
        if keep_ratio is not None and not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be in (0, 1]")
        n_total_layers = n_hidden_layers + 1
        if not (0 <= n_frozen_layers <= n_total_layers):
            raise ValueError(
                f"n_frozen_layers must be between 0 and {n_total_layers}, got {n_frozen_layers}"
            )

        self.n_nodes = n
        self.n_hidden_layers = n_hidden_layers
        self.use_fc_init = use_fc_init
        self.keep_ratio = keep_ratio
        self.n_frozen_layers = n_frozen_layers
        self.frozen_fc_init = frozen_fc_init

        if keep_ratio is not None:
            fc_tensor = self._threshold_fc(fc_tensor, keep_ratio)

        # Store the (possibly thresholded) FC matrix as a non-trainable buffer.
        self.register_buffer("fc_matrix", fc_tensor)

        self.network = self._build_network(fc_tensor, n, n_hidden_layers,
                                           use_fc_init, activation, output_activation,
                                           n_frozen_layers, frozen_fc_init)

        # Freeze the first n_frozen_layers linear layers.
        for layer in self.linear_layers()[:n_frozen_layers]:
            layer.requires_grad_(False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(m: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(m, np.ndarray):
            return torch.from_numpy(m).float()
        return m.float()

    @staticmethod
    def _threshold_fc(fc: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """
        Zero out the weakest off-diagonal connections, keeping keep_ratio
        fraction by absolute value.  Diagonal entries are always preserved.
        """
        n = fc.shape[0]
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=fc.device)
        abs_vals = fc[off_diag_mask].abs()
        total = abs_vals.numel()

        cutoff = torch.quantile(abs_vals, 1.0 - keep_ratio)
        fc_thresh = fc.clone()
        fc_thresh[off_diag_mask & (fc.abs() < cutoff)] = 0.0

        kept = int((fc_thresh[off_diag_mask] != 0).sum().item())
        print(
            f"FC thresholding: kept {kept:,} / {total:,} off-diagonal connections "
            f"({100 * kept / total:.1f}%)"
        )
        return fc_thresh

    @staticmethod
    def _build_network(
        fc_tensor: torch.Tensor,
        n: int,
        n_hidden_layers: int,
        use_fc_init: bool,
        activation: Type[nn.Module],
        output_activation: Optional[Type[nn.Module]],
        n_frozen_layers: int = 0,
        frozen_fc_init: bool = True,
    ) -> nn.Sequential:
        n_linear = n_hidden_layers + 1  # hidden layers + output layer
        layers: list[nn.Module] = []

        for i in range(n_linear):
            linear = nn.Linear(n, n)

            if use_fc_init or (i < n_frozen_layers and frozen_fc_init):
                with torch.no_grad():
                    linear.weight.copy_(fc_tensor)
                    nn.init.zeros_(linear.bias)

            layers.append(linear)

            is_last = i == n_linear - 1
            if not is_last:
                layers.append(activation())
            elif output_activation is not None:
                layers.append(output_activation())

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., N).
        Returns:
            Tensor of shape (..., N).
        """
        return self.network(x)

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def linear_layers(self) -> list[nn.Linear]:
        """Return all Linear sub-modules in order."""
        return [m for m in self.network if isinstance(m, nn.Linear)]

    def get_weight(self, layer_idx: int) -> torch.Tensor:
        """Weight matrix of the i-th linear layer (0-indexed)."""
        return self.linear_layers()[layer_idx].weight.data

    def weight_fc_correlation(self) -> list[float]:
        """
        Pearson correlation between each layer's current weight matrix and the
        original FC matrix.  Useful for tracking how much training drifts from
        the brain-derived initialisation.
        """
        fc_flat = self.fc_matrix.flatten()
        correlations = []
        for layer in self.linear_layers():
            w_flat = layer.weight.data.flatten()
            corr = torch.corrcoef(torch.stack([fc_flat, w_flat]))[0, 1].item()
            correlations.append(corr)
        return correlations

    def __repr__(self) -> str:
        init_mode  = "fc_matrix" if self.use_fc_init else "random (Kaiming)"
        thresh_str = f", keep_ratio={self.keep_ratio}" if self.keep_ratio is not None else ""
        if self.n_frozen_layers > 0:
            frozen_init = "fc" if self.frozen_fc_init else "random"
            frozen_str  = f", frozen_layers={self.n_frozen_layers}({frozen_init}_init)"
        else:
            frozen_str  = ""
        return (
            f"BrainConnectivityMLP("
            f"n_nodes={self.n_nodes}, "
            f"n_hidden_layers={self.n_hidden_layers}, "
            f"init={init_mode}"
            f"{thresh_str}"
            f"{frozen_str})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 10  # brain nodes

    # Simulate a symmetric FC matrix (correlations in [-1, 1])
    raw = rng.standard_normal((N, N))
    fc = (raw + raw.T) / 2
    np.fill_diagonal(fc, 1.0)

    print("=== FC-initialised MLP ===")
    model_fc = BrainConnectivityMLP(fc, n_hidden_layers=2, use_fc_init=True)
    print(model_fc)
    print("Layer 0 weight matches FC:", torch.allclose(
        model_fc.get_weight(0), torch.from_numpy(fc).float()))

    print("\n=== Randomly-initialised MLP ===")
    model_rand = BrainConnectivityMLP(fc, n_hidden_layers=2, use_fc_init=False)
    print(model_rand)

    print("\n=== FC-initialised MLP with top-20% connections ===")
    model_thresh = BrainConnectivityMLP(fc, n_hidden_layers=2, use_fc_init=True, keep_ratio=0.2)
    print(model_thresh)
    nonzero = (model_thresh.get_weight(0) != 0).sum().item()
    print(f"Non-zero weights in layer 0: {nonzero} / {N*N}")

    x = torch.randn(5, N)
    out = model_fc(x)
    print(f"\nForward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}")

    print("\nWeight–FC correlations per layer:", model_fc.weight_fc_correlation())
