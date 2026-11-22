from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from detector.core import (
    VLM_SYSTEM_PROMPT, VLM_USER_PROMPT, fuse, parse_vlm_verdict,
    read_expert_results, read_inputs, read_vlm_results,
)

DEFAULT_MAX_PIXELS = 1210000


def run_vlm(
    records: list[dict[str, str]], adapter: Path, swift_bin: str, workdir: Path,
    tensor_parallel_size: int, base_model: Path, max_pixels: int = DEFAULT_MAX_PIXELS,
    infer_backend: str = "vllm", torch_dtype: str | None = None,
    attn_impl: str | None = None, device_map: str | None = None,
) -> Path:
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"VLM adapter_config.json not found in {adapter}")
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"VLM base model config.json not found in {base_model}")
    dataset = workdir / "vlm_input.jsonl"
    result = workdir / "vlm_result.jsonl"
    with dataset.open("w", encoding="utf-8") as stream:
        for row in records:
            sample = {
                "messages": [
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": VLM_USER_PROMPT},
                ],
                "images": [row["image"]],
            }
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    command = [
        swift_bin, "infer", "--adapters", str(adapter),
        "--infer_backend", infer_backend,
        "--max_model_len", "8192",
        "--max_pixels", str(max_pixels),
        "--write_batch_size", "500",
        "--temperature", "0",
        "--max_new_tokens", "512",
        "--repetition_penalty", "1.0",
        "--top_p", "1.0",
        "--val_dataset", str(dataset),
        "--result_path", str(result),
        "--seed", "42",
        "--model", str(base_model),
        "--load_args", "false",
    ]
    if infer_backend == "vllm":
        command.extend([
            "--vllm_tensor_parallel_size", str(tensor_parallel_size),
            "--vllm_max_model_len", "8192",
            "--vllm_gpu_memory_utilization", "0.85",
        ])
    if torch_dtype is not None:
        command.extend(["--torch_dtype", torch_dtype])
    if attn_impl is not None:
        command.extend(["--attn_impl", attn_impl])
    if device_map is not None:
        command.extend(["--device_map", device_map])
    subprocess.run(
        command,
        check=True,
        env={**os.environ, "MAX_PIXELS": str(max_pixels)},
    )
    if not result.is_file():
        raise RuntimeError("VLM inference completed without writing result JSONL")
    return result


def run_expert(command: str, manifest: Path, workdir: Path) -> Path:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("--expert-command must not be empty")
    result = workdir / "expert_result.jsonl"
    subprocess.run(argv + ["--input", str(manifest), "--output", str(result)], check=True)
    if not result.is_file():
        raise RuntimeError("Expert command completed without writing result JSONL")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Image directory, path list (.txt), or single-image JSONL")
    parser.add_argument("--output", type=Path, required=True)
    expert = parser.add_mutually_exclusive_group()
    expert.add_argument("--expert-results", type=Path,
                        help="Existing JSONL with image and either fake or fake_probability")
    expert.add_argument("--expert-command",
                        help="Expert command; --input and --output are appended automatically")
    parser.add_argument("--expert-threshold", type=float, default=0.5)
    vlm = parser.add_mutually_exclusive_group(required=True)
    vlm.add_argument("--vlm-adapter", type=Path)
    vlm.add_argument("--vlm-results", type=Path,
                     help="Existing VLM result JSONL; skip VLM inference")
    parser.add_argument("--vlm-base-model", type=Path,
                        help="Base model directory; required with --vlm-adapter")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--infer-backend", choices=("vllm", "pt"), default="vllm",
                        help="Use pt for PyTorch inference without vLLM")
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--attn-impl", choices=("eager", "sdpa", "flash_attn"))
    parser.add_argument("--device-map", help="PyTorch model placement, e.g. auto for multiple GPUs")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help="Maximum image pixels; default is 1210000")
    args = parser.parse_args()
    if args.tensor_parallel_size < 1:
        parser.error("Tensor parallel size must be positive")
    if args.infer_backend == "pt" and args.tensor_parallel_size != 1:
        parser.error("--tensor-parallel-size requires the vllm backend")
    if args.device_map is not None and args.infer_backend != "pt":
        parser.error("--device-map requires the pt backend")
    if args.max_pixels < 1:
        parser.error("--max-pixels must be positive")
    if args.vlm_adapter is not None and args.vlm_base_model is None:
        parser.error("--vlm-base-model is required with --vlm-adapter")
    if not 0.0 <= args.expert_threshold <= 1.0:
        parser.error("--expert-threshold must be in [0, 1]")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    records = read_inputs(args.input.resolve())
    with tempfile.TemporaryDirectory(prefix="detector-") as temp:
        result_path = args.vlm_results
        if result_path is None:
            result_path = run_vlm(
                records, args.vlm_adapter.resolve(), args.swift_bin,
                Path(temp), args.tensor_parallel_size,
                args.vlm_base_model.resolve(), args.max_pixels,
                args.infer_backend, args.torch_dtype, args.attn_impl, args.device_map,
            )
        responses = read_vlm_results(result_path.resolve(), records)
        expert_results = None
        if args.expert_results is not None or args.expert_command is not None:
            expert_path = args.expert_results
            if expert_path is None:
                expert_path = run_expert(args.expert_command, args.input.resolve(), Path(temp))
            expert_results = read_expert_results(
                expert_path.resolve(), records, args.expert_threshold,
            )
    verdicts: dict[str, bool | None] = {}
    for row in records:
        try:
            verdicts[row["image"]] = parse_vlm_verdict(responses[row["image"]])
        except ValueError:
            verdicts[row["image"]] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in records:
            vlm_fake = verdicts[row["image"]]
            expert_fake, _ = (
                expert_results[row["image"]] if expert_results is not None else (None, None)
            )
            output = {
                "image": row["image"],
                "vlm_fake": vlm_fake,
                "expert_fake": expert_fake,
                "fake": vlm_fake if expert_fake is None else fuse(vlm_fake, expert_fake),
                "vlm_response": responses[row["image"]],
            }
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
