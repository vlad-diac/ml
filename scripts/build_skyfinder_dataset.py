#!/usr/bin/env python3
"""Assemble SkyFinder into data/datasets/skyfinder_<date>/images/{train,val,test} and masks/{...}.

Reads:
  - data/unformatted/skyfinder/<CAMERA_ID>.zip  — per-camera images (JPG/PNG)
  - data/unformatted/skyfinder/skyfinder_masks.zip — skyfinder_masks/<CAMERA_ID>.png

Each camera has one static mask (same W×H as that camera's images). Every image gets a
unique stem: ``{CAMERA_ID}_{original_filename_stem}`` so pairs are unambiguous:

  images/train/1093_20140811_155409.jpg
  masks/train/1093_20140811_155409.png

Masks are saved as single-channel PNG with values in {0, 255} only.

Requires: Pillow (``pip install pillow``).
"""

from __future__ import annotations

import argparse
import datetime
import io
import random
import re
import sys
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    print("This script needs Pillow: pip install pillow", file=sys.stderr)
    raise SystemExit(1) from e

CAMERA_ZIP_RE = re.compile(r"^(\d+)\.zip$", re.IGNORECASE)
MASKS_ZIP_NAMES = frozenset({"skyfinder_masks.zip", "skyfindermasks.zip"})


def _safe_extract_member(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in zf.namelist():
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe path in archive: {name!r}")
        full = (dest / member).resolve()
        full.relative_to(dest)
    zf.extractall(dest)


def _find_masks_zip(source: Path) -> Path | None:
    for p in source.iterdir():
        if not p.is_file():
            continue
        if p.name.lower() in {n.lower() for n in MASKS_ZIP_NAMES}:
            return p
    return None


def _load_mask_png_by_camera(masks_zip: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(masks_zip, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".png"):
                continue
            stem = Path(name).stem
            if stem.isdigit():
                out[stem] = zf.read(name)
    return out


def _binarize_mask_png_to_bytes(png_bytes: bytes) -> bytes:
    """Return PNG bytes, mode L, values only 0 and 255."""
    im = Image.open(io.BytesIO(png_bytes))
    gray = im.convert("L")
    lo, hi = gray.getextrema()
    if hi <= 1:
        out = gray.point(lambda p: 255 if p > 0 else 0, mode="L")
    else:
        out = gray.point(lambda p: 255 if p > 127 else 0, mode="L")
    buf = io.BytesIO()
    out.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _iter_camera_archives(source: Path) -> list[Path]:
    zips: list[Path] = []
    for p in sorted(source.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".zip":
            continue
        if p.name.lower() in {n.lower() for n in MASKS_ZIP_NAMES}:
            continue
        if CAMERA_ZIP_RE.match(p.name):
            zips.append(p)
    return zips


def _collect_samples(
    camera_zips: list[Path],
    mask_png_by_camera: dict[str, bytes],
    binarized_cache: dict[str, bytes],
) -> tuple[list[tuple[str, bytes, bytes]], list[str]]:
    """Returns (samples, warnings). Each sample: (stem, image_bytes, mask_png_bytes)."""
    samples: list[tuple[str, bytes, bytes]] = []
    warnings: list[str] = []

    for cam_zip in camera_zips:
        m = CAMERA_ZIP_RE.match(cam_zip.name)
        assert m is not None
        camera_id = m.group(1)

        raw_mask = mask_png_by_camera.get(camera_id)
        if raw_mask is None:
            warnings.append(f"No mask skyfinder_masks/{camera_id}.png — skipping archive {cam_zip.name}")
            continue

        if camera_id not in binarized_cache:
            binarized_cache[camera_id] = _binarize_mask_png_to_bytes(raw_mask)
        mask_out = binarized_cache[camera_id]

        n_before = len(samples)
        with zipfile.ZipFile(cam_zip, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                suf = Path(info.filename).suffix.lower()
                if suf not in {".jpg", ".jpeg", ".png"}:
                    continue
                inner_stem = Path(info.filename).stem
                stem = f"{camera_id}_{inner_stem}"
                samples.append((stem, zf.read(info.filename), mask_out))

        if len(samples) == n_before:
            warnings.append(f"No images found in {cam_zip.name}")

    return samples, warnings


def _assign_splits_simple(n: int, train_r: float, val_r: float, test_r: float, rng: random.Random) -> list[str]:
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must sum to 1.0")
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 0:
        raise ValueError("Split counts invalid; adjust ratios.")
    labels = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    if len(labels) != n:
        raise RuntimeError("internal split length mismatch")
    rng.shuffle(labels)
    return labels


def _clear_split_dirs(data_root: Path) -> None:
    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            d = data_root / sub / split
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()


def _write_dataset(
    samples: list[tuple[str, bytes, bytes, str]],
    data_root: Path,
) -> None:
    for stem, img_bytes, mask_bytes, split in samples:
        ext = _image_suffix_from_bytes(img_bytes)
        img_path = data_root / "images" / split / f"{stem}{ext}"
        mask_path = data_root / "masks" / split / f"{stem}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(img_bytes)
        mask_path.write_bytes(mask_bytes)


def _image_suffix_from_bytes(img_bytes: bytes) -> str:
    """Return .jpg or .png for output filename (keeps JPEG as .jpg)."""
    if len(img_bytes) >= 3 and img_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(img_bytes) >= 8 and img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/unformatted/skyfinder"),
        help="Folder containing <ID>.zip archives and skyfinder_masks.zip",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Output root (writes images/ and masks/ underneath). "
        "Default: data/datasets/skyfinder_<YYYY-MM-DD> (today's date).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction for train split")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction for val split")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction for test split")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling before split")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing files under <data-root>/images/{train,val,test} and masks/...",
    )
    parser.add_argument(
        "--extract-masks-to",
        type=Path,
        default=None,
        help="Optional: extract skyfinder_masks.zip to this directory (safe extract only)",
    )
    args = parser.parse_args()

    if args.data_root is None:
        dated = datetime.date.today().isoformat()
        data_root_arg = Path("data/datasets") / f"skyfinder_{dated}"
    else:
        data_root_arg = args.data_root

    source = args.source.resolve()
    if not source.is_dir():
        print(f"Source not found or not a directory: {source}", file=sys.stderr)
        raise SystemExit(1)

    masks_zip = _find_masks_zip(source)
    if masks_zip is None:
        print(f"No skyfinder_masks.zip in {source}", file=sys.stderr)
        raise SystemExit(1)

    if args.extract_masks_to is not None:
        dest = args.extract_masks_to.resolve()
        with zipfile.ZipFile(masks_zip, "r") as zf:
            _safe_extract_member(zf, dest)
        print(f"Extracted masks to {dest}")

    mask_png_by_camera = _load_mask_png_by_camera(masks_zip)
    camera_zips = _iter_camera_archives(source)
    if not camera_zips:
        print(f"No camera *.zip archives in {source}", file=sys.stderr)
        raise SystemExit(1)

    binarized_cache: dict[str, bytes] = {}
    samples, warnings = _collect_samples(camera_zips, mask_png_by_camera, binarized_cache)
    for w in warnings:
        print(w, file=sys.stderr)

    if not samples:
        print("No samples collected (check masks and camera zips).", file=sys.stderr)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    labels = _assign_splits_simple(len(samples), args.train_ratio, args.val_ratio, args.test_ratio, rng)

    data_root = data_root_arg.resolve()
    if args.clean:
        _clear_split_dirs(data_root)

    packed = [(stem, img_b, mask_b, labels[i]) for i, (stem, img_b, mask_b) in enumerate(samples)]
    _write_dataset(packed, data_root)

    from collections import Counter

    c = Counter(s[3] for s in packed)
    print(f"Wrote {len(packed)} pairs under {data_root / 'images'} and {data_root / 'masks'}")
    print(f"Counts: train={c['train']} val={c['val']} test={c['test']}")


if __name__ == "__main__":
    main()
