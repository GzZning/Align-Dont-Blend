# Third-party sources

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, Apache-2.0.
- [roy-ch/Dual-Data-Alignment](https://github.com/roy-ch/Dual-Data-Alignment/tree/08d6b414a33826403b9096cfdb7758d5eae12dbb), commit `08d6b414a33826403b9096cfdb7758d5eae12dbb`, Apache-2.0.

DDA: LoRA layers from `Inference/models/lora.py`.
DINOv2: vision transformer and supporting layers; training head omitted.
Modified DINOv2 imports to use this package; omitted the training-head export.
License texts are retained in each source directory.
