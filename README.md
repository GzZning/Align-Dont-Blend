
<div align="center">

<h1>Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable AI-Generated Image Detection</h1>

<p>
  <a href="https://arxiv.org/abs/2512.06746"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Gaffeyzz/AIGI-Now"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow" alt="Hugging Face"></a>
  <a href="https://www.modelscope.cn/datasets/gaffeykao/AIGI-Now"><img src="https://img.shields.io/badge/ModelScope-Dataset-blue" alt="ModelScope"></a>
</p>

</div>

This is the official repository for **AlignGemini**, a two-branch detector for generalizable AI-generated image detection. AlignGemini pairs a vision-language model trained with pure semantic supervision with a vision expert trained with pure pixel-artifact supervision. This orthogonal design improves average balanced accuracy by **9.5 points** across five in-the-wild benchmarks.

We also introduce **AIGI-Now**, a diagnostic benchmark covering nine contemporary image generators with separate semantic and pixel-artifact evaluation subsets.

## Release Status

- Paper: available
- AIGI-Now test dataset: available
- Training code and inference scripts: coming soon

## Dataset

| Platform | Link |
| --- | --- |
| Hugging Face | [Gaffeyzz/AIGI-Now](https://huggingface.co/datasets/Gaffeyzz/AIGI-Now) |
| ModelScope | [gaffeykao/AIGI-Now](https://www.modelscope.cn/datasets/gaffeykao/AIGI-Now) |

## Citation

```bibtex
@article{chen2025aligngemini,
  title={Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable AI-Generated Image Detection},
  author={Chen, Ruoxin and Gao, Jiahui and Lin, Kaiqing and Zhang, Keyue and Zhao, Yandan and Guan, Isabel and Yao, Taiping and Ding, Shouhong},
  journal={arXiv preprint arXiv:2512.06746},
  year={2025}
}
```
