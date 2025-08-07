import torch
import torch.nn as nn
import torchvision.models as tv


class AttentionResNet50(nn.Module):
    """ResNet‑50 with attention mechanism classifier."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout_p: float = 0.5,
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

        # ‑‑‑ Attention mechanism ‑‑‑
        self.attention = nn.Sequential(
            nn.Linear(self.backbone_out_channels, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )

        # ‑‑‑ Classification head ‑‑‑
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.backbone_out_channels, num_classes),
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

        # Apply attention
        attn_weights = self.attention(feats)  # (B, T, 1)
        attn_applied = torch.sum(feats * attn_weights, dim=1)  # (B, C')

        logits = self.head(attn_applied)  # (B, num_classes)
        return logits


# For quick sanity check
if __name__ == "__main__":
    model = AttentionResNet50(num_classes=30)
    dummy = torch.randn(2, 16, 3, 224, 224)
    out = model(dummy)
    print(out.shape)  # (2, 30)