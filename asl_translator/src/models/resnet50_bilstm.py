# models/resnet50_bilstm.py
"""Bigger-capacity vision‑LSTM baseline for ASL recognition.

Backbone: torchvision.models.resnet50 (pre‑trained)
Temporal encoder: 2‑layer Bi‑LSTM
Classification head: dropout → FC → logits

Input expected: Tensor[B, T, C, H, W] with T fixed by the dataloader.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tv


class ResNet50BiLSTM(nn.Module):
    """Frame‑wise ResNet‑50 + sequence Bi‑LSTM classifier."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout_p: float = 0.5,
        bidirectional: bool = True,
        freeze_backbone_until: str | None = "layer2",
    ) -> None:
        super().__init__()

        # ‑‑‑ CNN backbone ‑‑‑
        backbone: tv.ResNet = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # keep up to conv5_x
        self.backbone_out_channels: int = 2048

        # Optionally freeze early stages to speed convergence
        if freeze_backbone_until is not None:
            freeze = True
            for name, param in backbone.named_parameters():
                if name.startswith(freeze_backbone_until):
                    freeze = False
                param.requires_grad = not freeze

        # ‑‑‑ Temporal encoder ‑‑‑
        self.lstm = nn.LSTM(
            input_size=self.backbone_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)

        # ‑‑‑ Classification head ‑‑‑
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_out, num_classes),
        )

    @torch.no_grad()
    def _extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN on a batch of frames. Shape: (B*T, C, H, W) -> (B*T, C')"""
        feats = self.backbone(x)  # (B*T, C', H', W')
        feats = torch.mean(feats, dim=[-2, -1])  # global avg‑pool
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """x: (B, T, C, H, W)"""
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)  # merge time into batch
        feats = self._extract_feat(x)  # (B*T, C')
        feats = feats.view(b, t, -1)  # (B, T, C')

        seq_out, _ = self.lstm(feats)  # (B, T, H)
        last = seq_out[:, -1]  # (B, H)  use last step (after bidirectional concat)
        logits = self.head(last)  # (B, num_classes)
        return logits


# For quick sanity check
if __name__ == "__main__":
    model = ResNet50BiLSTM(num_classes=30)
    dummy = torch.randn(2, 16, 3, 224, 224)
    out = model(dummy)
    print(out.shape)  # (2, 30)
