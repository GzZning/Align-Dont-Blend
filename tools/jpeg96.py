"""In-memory JPEG96 preprocessing for GenImage and AIGCDetectBenchmark."""

import numpy as np
from PIL import Image


def png2jpg(img, quality):
    """Convert PNG to JPG with specified quality"""
    from io import BytesIO

    out = BytesIO()
    img.save(out, format="jpeg", quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def prepare_image(image, dataset, label):
    """Convert to RGB and apply JPEG96 to Fake images in the named datasets."""
    image = image.convert("RGB")
    if dataset in {"GenImage", "AIGCDetectBenchmark", "AIGCDetectionBenchmark", "AIGCDetect"} and label == 1:
        image = png2jpg(image, 96)
    return image
