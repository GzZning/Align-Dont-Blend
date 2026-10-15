"""Map archived semantic-preference records to a local image root."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Directory containing train/unsplash and train/Echo-4o")
    parser.add_argument("--output", type=Path,
                        default=Path("data/semantic_preference_train.jsonl"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    counts: Counter[str] = Counter()
    rows = []
    for line_number, line in enumerate(args.source_manifest.open(encoding="utf-8"), 1):
        row = json.loads(line)
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"Line {line_number}: expected exactly one image")
        parts = Path(images[0]).parts
        if "train" not in parts:
            raise ValueError(f"Line {line_number}: missing train/ path segment")
        relative = Path(*parts[parts.index("train"):])
        group = relative.parts[1] if len(relative.parts) > 1 else ""
        if group not in {"unsplash", "Echo-4o"}:
            raise ValueError(f"Line {line_number}: unexpected source {group!r}")
        image = (args.image_root / relative).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        row["images"] = [str(image)]
        counts[group] += 1
        rows.append(row)
    if counts != {"unsplash": 5000, "Echo-4o": 5000}:
        raise ValueError(f"Expected 5,000 images per source, got {dict(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
