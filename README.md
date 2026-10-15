# AlignGemini

### Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable AI-Generated Image Detection

[Project Page](https://github.com/GzZning/Align-Don-t-Blend) | [arXiv](https://arxiv.org/abs/2512.06746)

This repository is the official implementation of "Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable AI-Generated Image Detection".

![AlignGemini pipeline](assets/images/pipeline.jpg)

## Training

### VLM branch

The semantic preference dataset ([Hugging Face](https://huggingface.co/datasets/Gaffeyzz/AlignGemini-Semantic-Preference-10K), [ModelScope](https://www.modelscope.cn/datasets/GaffeyGao/AlignGemini-Semantic-Preference-10K)) contains 5,000 Unsplash real images, 5,000 semantically implausible Echo-4o images, and their DPO preference records. We recommend ms-swift 3.10.x. The provided script is configured for one training epoch:

```bash
VLM_MODEL=/path/to/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=/path/to/output \
bash train_vlm.sh
```

## Inference

Run on an image folder:

```bash
python predict.py --input /path/to/images --output predictions.jsonl \
  --vlm-adapter weights/vlm_adapter \
  --vlm-base-model /path/to/Qwen2.5-VL-7B-Instruct
```

`predictions.jsonl` contains each image path, final decision, expert decision, VLM decision, and VLM response. The expert is optional; without it, the VLM makes the final decision. See the [expert example](extra/pixel_artifact_expert/README.md#dda-example) to add one.

## New Challenging Benchmark

AIGI-Now is available on [ModelScope](https://www.modelscope.cn/datasets/GaffeyGao/AIGI-Now) and [Hugging Face](https://huggingface.co/datasets/Gaffeyzz/AIGI-Now).

![AIGI-Now benchmark](assets/images/aigi-now2.jpg)

## Citing

If you find this work useful, please cite the following works.

```bibtex
@inproceedings{chen2026aligngemini,
  title={Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable AI-Generated Image Detection},
  author={Chen, Ruoxin and Gao, Jiahui and Lin, Kaiqing and Zhang, Keyue and Zhao, Yandan and Guan, Isabel and Yao, Taiping and Ding, Shouhong},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}

@inproceedings{chen2025dual,
  title={Dual Data Alignment Makes {AI}-Generated Image Detector Easier Generalizable},
  author={Ruoxin Chen and Junwei Xi and Zhiyuan Yan and Ke-Yue Zhang and Shuang Wu and Jingyi Xie and Xu Chen and Lei Xu and Isabel Guan and Taiping Yao and Shouhong Ding},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=C39ShJwtD5}
}
```
