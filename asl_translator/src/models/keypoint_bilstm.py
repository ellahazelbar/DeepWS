import torch
import torch.nn as nn
from typing import Optional, Tuple

class KeypointBiLSTM(nn.Module):
    """
    BiLSTM classifier for Mediapipe/landmark feature sequences.

    Accepts input shaped (B, T, D) or (B, D, T). If given (B, D, T), it will
    automatically transpose to (B, T, D) using a simple heuristic:
      - The smaller of the two last dimensions is treated as feature_dim (D),
        and the larger as time_len (T). This fits common cases like (B, 1629, 131)
        or (B, 131, 1629) without code changes.

    Args:
        num_classes: number of output classes.
        embed_dim:   size of per-time-step embedding before LSTM.
        hidden_size: LSTM hidden size.
        num_layers:  number of LSTM layers.
        bidirectional: use BiLSTM if True.
        dropout_p:   dropout rate after LSTM and before classifier.
        norm_input:  if True, apply LayerNorm to raw D-dim inputs per time step.
        use_mean_pool: if True, mean-pool LSTM outputs over time; otherwise use last time step.
    """
    def __init__(
        self,
        num_classes: int,
        D : int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
        norm_input: bool = True,
        use_mean_pool: bool = True,
    ) -> None:
        super().__init__()
        self.use_mean_pool = use_mean_pool

        # We don't know D at construction; build lazy modules on first forward.
        self._norm_input = norm_input

        # Placeholders that will be initialized lazily
        self.input_norm: Optional[nn.LayerNorm] = None
        self.input_proj: Optional[nn.Linear] = None
        self.temporal: Optional[nn.LSTM] = None
        self.dropout = nn.Dropout(dropout_p)
        self.classifier: Optional[nn.Linear] = None
        self._bidirectional = bidirectional
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._embed_dim = embed_dim
        self._num_classes = num_classes
        self.build(D)

    def build(self, D: int) -> None:
        if self._norm_input:
            self.input_norm = nn.LayerNorm(D)
        self.input_proj = nn.Linear(D, self._embed_dim)
        self.temporal = nn.LSTM(
            input_size=self._embed_dim,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            batch_first=True,
            bidirectional=self._bidirectional,
            dropout=0.0 if self._num_layers < 2 else 0.2,
        )
        lstm_out = self._hidden_size * (2 if self._bidirectional else 1)
        self.classifier = nn.Linear(lstm_out, self._num_classes)

    def _to_B_T_D(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, D, T); convert to (B, T, D)
        assert x.dim() == 3, f"Expected 3D input (B,D,T), got {x.shape}"
        B, A, B2 = x.shape
        # x is (B, D, T) -> transpose to (B, T, D)
        x = x.transpose(1, 2).contiguous()
        # else assume already (B, T, D)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, T, D) or (B, D, T).
        Returns:
            logits: (B, num_classes)
        """
        x = self._to_B_T_D(x)
        B, T, D = x.shape

        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.input_proj(x)  # (B, T, embed_dim)

        # LSTM encoding
        seq, _ = self.temporal(x)  # (B, T, H')
        if self.use_mean_pool:
            pooled = seq.mean(dim=1)
        else:
            pooled = seq[:, -1]

        logits = self.classifier(self.dropout(pooled))  # (B, num_classes)
        return logits