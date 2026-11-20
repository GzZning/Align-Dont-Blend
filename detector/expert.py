from __future__ import annotations

import torch
from torch import nn

from .third_party.dda.lora import apply_lora_to_linear_layers
from .third_party.dinov2.models.vision_transformer import vit_large


class DDAExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = vit_large(
            img_size=518, patch_size=14, init_values=1.0, block_chunks=0,
            num_register_tokens=0, interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        self.base_model.fc = nn.Linear(1024, 1)
        apply_lora_to_linear_layers(
            self.base_model.model, rank=8, alpha=1.0,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            trainable_orig=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.base_model.model.forward_features(images)["x_norm_clstoken"]
        return self.base_model.fc(features)
