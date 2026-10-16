from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from detector.core import read_inputs


class PixelArtifactExpert:
    def __init__(self, source: Path, backbone_repo: Path, checkpoint: Path, device: str):
        import torch
        from torchvision import transforms

        source = source.resolve()
        backbone_repo = backbone_repo.resolve()
        checkpoint = checkpoint.resolve()
        if not (source / "models" / "dinov2_models_lora.py").is_file():
            raise FileNotFoundError(f"Expert source not found: {source}")
        if not (backbone_repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Backbone source not found: {backbone_repo}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        sys.path.insert(0, str(source))
        try:
            from models.dinov2_models_lora import DINOv2ModelWithLoRA
        finally:
            sys.path.remove(str(source))

        original_hub_load = torch.hub.load

        def local_backbone(repo: str, name: str, *args, **kwargs):
            if repo != "facebookresearch/dinov2" or name != "dinov2_vitl14":
                raise ValueError(f"Unexpected backbone request: {repo}/{name}")
            kwargs.pop("pretrained", None)
            return original_hub_load(
                str(backbone_repo), name, *args,
                source="local", pretrained=False, **kwargs,
            )

        torch.hub.load = local_backbone
        try:
            model = DINOv2ModelWithLoRA(
                name="dinov2_vitl14", lora_rank=8, lora_alpha=1,
                lora_targets=None,
            )
        finally:
            torch.hub.load = original_hub_load

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.device = device
        self.torch = torch
        self.transform = transforms.Compose([
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def probabilities(self, images: list[str], batch_size: int) -> list[float]:
        from PIL import Image

        scores: list[float] = []
        for start in range(0, len(images), batch_size):
            batch = []
            for path in images[start:start + batch_size]:
                with Image.open(path) as image:
                    batch.append(self.transform(image.convert("RGB")))
            tensor = self.torch.stack(batch).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(tensor).flatten()
                probabilities = logits.sigmoid().cpu().tolist()
            if len(probabilities) != len(batch):
                raise RuntimeError("Expert returned the wrong batch size")
            scores.extend(probabilities)
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--backbone-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    records = read_inputs(args.input.resolve())
    expert = PixelArtifactExpert(
        args.expert_source, args.backbone_repo, args.checkpoint, args.device,
    )
    probabilities = expert.probabilities(
        [row["image"] for row in records], args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row, probability in zip(records, probabilities, strict=True):
            stream.write(json.dumps({
                "image": row["image"], "fake_probability": probability,
            }) + "\n")


if __name__ == "__main__":
    main()
