"""Batch inpaint images with Simple LaMa (simple-lama-inpainting)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image
from simple_lama_inpainting import SimpleLama

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _repo_root() -> Path:
    # scripts/mask/LaMa/inpaint_image.py -> parents[3] == tensorflow project root
    return Path(__file__).resolve().parents[3]


def _natural_sort_key(path: Path) -> tuple[str | int, ...]:
    """Filename key so numeric runs sort in numeric order (e.g. 65_... before 10066_...)."""
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _sorted_image_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=_natural_sort_key)


def _list_images(directory: Path, limit: int) -> list[Path]:
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files = _sorted_image_paths(files)
    return files[:limit]


def _find_mask_file(image_path: Path, mask_dir: Path) -> Path | None:
    stem = image_path.stem
    for ext in MASK_EXTENSIONS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{stem}_mask{ext}"
        if candidate.is_file():
            return candidate
    return None


def _prepare_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    m = mask.convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return m


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_input = root / "data/datasets/skyfinder_2026-04-09/images/train"
    default_mask = root / "data/datasets/skyfinder_2026-04-09/masks/train"
    default_output = root / "data/datasets/skyfinder_2026-04-09/images/no_sky"

    p = argparse.ArgumentParser(
        description="Inpaint images with Simple LaMa; writes results to output_folder."
    )
    p.add_argument(
        "-n",
        "--count",
        type=int,
        default=5,
        help="Maximum number of images to process when input_image is a directory.",
    )
    p.add_argument(
        "--input-image",
        type=Path,
        default=default_input,
        help="Path to an RGB image file, or a directory of images.",
    )
    p.add_argument(
        "--input-mask",
        type=Path,
        default=default_mask,
        help="Single mask image (grayscale; 255 = inpaint), or a directory of masks "
        "with the same stem as each input image.",
    )
    p.add_argument(
        "--output-folder",
        type=Path,
        default=default_output,
        help="Directory for inpainted outputs (created if missing).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    input_image = args.input_image.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()

    input_mask = args.input_mask.expanduser().resolve()

    if input_image.is_dir():
        image_paths = _list_images(input_image, max(0, args.count))
        if not image_paths:
            print(f"No images found under {input_image}", file=sys.stderr)
            return 1
    else:
        if not input_image.is_file():
            print(f"Not a file: {input_image}", file=sys.stderr)
            return 1
        image_paths = [input_image]

    if input_mask.is_dir():
        mask_is_dir = True
    elif input_mask.is_file():
        mask_is_dir = False
    else:
        print(f"Mask path not found: {input_mask}", file=sys.stderr)
        return 1

    output_folder.mkdir(parents=True, exist_ok=True)
    lama = SimpleLama()

    if mask_is_dir:
        pairs: list[tuple[Path, Path]] = []
        for img_path in image_paths:
            mask_path = _find_mask_file(img_path, input_mask)
            if mask_path is None:
                print(
                    f"No mask for {img_path.name} under {input_mask}; skipping.",
                    file=sys.stderr,
                )
                continue
            pairs.append((img_path, mask_path))
        pairs.sort(key=lambda t: (_natural_sort_key(t[0]), _natural_sort_key(t[1])))
        jobs: list[tuple[Path, Path | None]] = pairs
    else:
        jobs = [(img_path, None) for img_path in image_paths]

    shared_mask: Image.Image | None = None
    if not mask_is_dir:
        shared_mask = Image.open(input_mask)

    try:
        for img_path, mask_path in jobs:
            if mask_is_dir:
                assert mask_path is not None
                mask_img = Image.open(mask_path)
                try:
                    image = Image.open(img_path).convert("RGB")
                    mask_prepared = _prepare_mask(mask_img, image.size)
                finally:
                    mask_img.close()
            else:
                assert shared_mask is not None
                image = Image.open(img_path).convert("RGB")
                mask_prepared = _prepare_mask(shared_mask.copy(), image.size)

            result = lama(image, mask_prepared)
            out_path = output_folder / img_path.name
            result.save(out_path)
            print(out_path)
    finally:
        if shared_mask is not None:
            shared_mask.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
