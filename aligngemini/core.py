"""Input/output rules for the two-expert AlignGemini detector."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


VLM_SYSTEM_PROMPT = """Role: AI visual assistant.
You are an AI visual assistant that analyzes a single image to determine whether it is authentic or AI-generated.
Tasks
1. <rationale> section:
• Provide concise, evidence-based reasoning in fewer than 120 words.
• Focus strictly on semantic or logical reasoning.
2. <answer> section:
• If real: only output “This is an authentic image.”
• If AI-generated: only output “This is an AI-generated image.”
Output Structure (must match exactly):
<rationale></rationale>
<answer></answer>"""
VLM_USER_PROMPT = "<image>\nWhether the image is authentic or AI-generated?"

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_SENTENCE = re.compile(
    r"(?:^|\n)\s*This is an (authentic|AI-generated) image\.\s*(?:</answer>)?\s*$",
    re.IGNORECASE,
)
_VERDICTS = {
    "this is an authentic image.": False,
    "this is an ai-generated image.": True,
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_vlm_verdict(response: str) -> bool:
    """Return True for synthetic; accept an explicit final verdict only."""
    matches = _ANSWER.findall(response)
    if len(matches) == 1:
        answer = " ".join(matches[0].split()).casefold()
        if answer not in _VERDICTS:
            raise ValueError(f"Unrecognized VLM verdict: {matches[0]!r}")
        return _VERDICTS[answer]
    if len(matches) == 0:
        final = _FINAL_SENTENCE.search(response)
        if final:
            return final.group(1).casefold() == "ai-generated"
    raise ValueError(f"Expected one explicit final verdict, found {len(matches)} tags")


def _image_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def read_inputs(manifest: Path) -> list[dict[str, str]]:
    """Read images from a directory, path list, or single-image JSONL."""
    if manifest.is_dir():
        records = [
            {"id": str(index), "image": str(path.resolve())}
            for index, path in enumerate(sorted(
                (path for path in manifest.rglob("*")
                 if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
                key=lambda path: path.relative_to(manifest).as_posix(),
            ), 1)
        ]
        if not records:
            raise ValueError(f"No images found in {manifest}")
        return records
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            if manifest.suffix == ".jsonl":
                row = json.loads(raw)
                images = row.get("images")
                if images is not None:
                    if not isinstance(images, list) or len(images) != 1:
                        raise ValueError(f"Line {line_number}: expected exactly one image")
                    value = images[0]
                else:
                    value = row["image"]
                sample_id = str(row.get("id", line_number))
            else:
                value = raw.strip()
                sample_id = str(line_number)
            image = _image_path(value, manifest.parent)
            if sample_id in seen_ids or image in seen_images:
                raise ValueError(f"Line {line_number}: duplicate id or image")
            seen_ids.add(sample_id)
            seen_images.add(image)
            records.append({"id": sample_id, "image": image})
    if not records:
        raise ValueError("Input manifest is empty")
    return records


def read_vlm_results(path: Path, records: list[dict[str, str]]) -> dict[str, str]:
    """Match VLM JSONL responses to images, independent of row order."""
    expected = {row["image"] for row in records}
    responses: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            images = row.get("images")
            if images is not None:
                if not isinstance(images, list) or len(images) != 1:
                    raise ValueError(f"VLM result line {line_number}: expected one image")
                image_value = images[0]
            else:
                image_value = row.get("image")
            if not isinstance(image_value, str):
                raise ValueError(f"VLM result line {line_number}: missing image")
            image = str(Path(image_value).expanduser().resolve())
            if image not in expected or image in responses:
                raise ValueError(f"VLM result line {line_number}: unknown or duplicate image")
            response = row.get("response")
            if response is None:
                response_keys = [key for key in row if key.endswith("_response")]
                if len(response_keys) == 1:
                    response = row[response_keys[0]]
            if not isinstance(response, str):
                raise ValueError(f"VLM result line {line_number}: missing response")
            responses[image] = response
    if responses.keys() != expected:
        raise ValueError(f"Missing VLM results for {len(expected - responses.keys())} images")
    return responses


def read_expert_results(
    path: Path, records: list[dict[str, str]], threshold: float = 0.5,
) -> dict[str, tuple[bool, float | None]]:
    """Read one fake label or fake probability for every input image."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Expert threshold must be in [0, 1]")
    expected = {row["image"] for row in records}
    results: dict[str, tuple[bool, float | None]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            image = str(Path(row["image"]).expanduser().resolve())
            if image not in expected or image in results:
                raise ValueError(f"Expert result line {line_number}: unknown or duplicate image")
            has_label = "fake" in row
            has_probability = "fake_probability" in row
            if has_label == has_probability:
                raise ValueError(f"Expert result line {line_number}: provide fake or fake_probability")
            if has_label:
                if not isinstance(row["fake"], bool):
                    raise ValueError(f"Expert result line {line_number}: fake must be boolean")
                results[image] = (row["fake"], None)
            else:
                probability = row["fake_probability"]
                if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                        or not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
                    raise ValueError(f"Expert result line {line_number}: invalid fake_probability")
                results[image] = (probability >= threshold, float(probability))
    if results.keys() != expected:
        raise ValueError(f"Missing expert results for {len(expected - results.keys())} images")
    return results


def fuse(vlm_fake: bool, expert_fake: bool) -> bool:
    """An image is real only when both experts predict real."""
    return vlm_fake or expert_fake
