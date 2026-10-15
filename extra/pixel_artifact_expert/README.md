# Expert interface

AlignGemini accepts predictions from any pixel-artifact expert. For each input image, write one JSONL record with its image path and either `fake` (a boolean) or `fake_probability` (a number in `[0, 1]`, where larger means more likely synthetic). With probability inputs, the default decision threshold is `0.5`; boolean inputs are used directly. See the main [inference instructions](../../README.md#inference).

## DDA example

The optional [DDA exporter](examples/dda.py) converts an independently obtained DDA checkpoint into this format. Obtain the [DDA inference source](https://github.com/roy-ch/Dual-Data-Alignment/tree/main/Inference), a compatible DINOv2 backbone, and an expert checkpoint separately; no expert weights are bundled here.

```bash
python extra/pixel_artifact_expert/examples/dda.py \
  --input images.txt --output expert.jsonl \
  --expert-source /path/to/Dual-Data-Alignment/Inference \
  --backbone-repo /path/to/dinov2 \
  --checkpoint /path/to/expert_checkpoint.pth
```

The exporter writes `image` and `fake_probability` for each image. Other experts only need to produce the same JSONL interface; they do not need to use this script.

To run both branches from one command, pass the exporter as `--expert-command` to `predict.py`; omit `--input` and `--output` inside the quoted command because `predict.py` supplies them:

```bash
python predict.py --input images.txt --output predictions.jsonl \
  --vlm-adapter weights/vlm_adapter \
  --vlm-base-model /path/to/Qwen2.5-VL-7B-Instruct \
  --expert-command 'python extra/pixel_artifact_expert/examples/dda.py --expert-source /path/to/Dual-Data-Alignment/Inference --backbone-repo /path/to/dinov2 --checkpoint /path/to/expert_checkpoint.pth'
```
