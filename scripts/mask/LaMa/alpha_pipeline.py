"""Alpha mask pipeline: LaMa inpainting → diff map → guided filter → alpha mask + collage.

Supports parameter sweeping: pass multiple values to any tunable flag and the
pipeline generates one alpha+collage per configuration combination (cartesian
product).  LaMa inpainting is run only once per image regardless of how many
configurations are swept.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Font used for collage labels
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LABEL_H = 28          # pixel height of the label strip
_LABEL_PAD = 6         # left padding for text


# ---------------------------------------------------------------------------
# Path utilities (mirrors inpaint_image.py)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # scripts/mask/LaMa/alpha_pipeline.py -> parents[3] == tensorflow project root
    return Path(__file__).resolve().parents[3]


def _natural_sort_key(path: Path) -> tuple[str | int, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _sorted_image_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=_natural_sort_key)


def _list_images(directory: Path, limit: int, randomise: bool = False) -> list[Path]:
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files = _sorted_image_paths(files)
    if randomise:
        files = random.sample(files, min(limit, len(files)))
        files = _sorted_image_paths(files)  # keep natural order after sampling
    else:
        files = files[:limit]
    return files


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


# ---------------------------------------------------------------------------
# Pipeline config (one instance per sweep combination)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    guided_radius: int
    guided_eps: float
    gamma: float           # diff gamma
    edge_band_iters: int
    edge_blend: float
    alpha_gamma: float     # gamma applied to alpha after normalization
    sky_conf: float        # orig_mask coefficient in sky confidence floor
    sky_base: float        # constant offset in sky confidence floor

    def key(self) -> str:
        """Short filesystem-safe string identifying this configuration."""
        eps_str = f"{self.guided_eps:.0e}".replace("-0", "-")
        return (
            f"r{self.guided_radius}"
            f"_e{eps_str}"
            f"_g{self.gamma}"
            f"_b{self.edge_blend}"
            f"_band{self.edge_band_iters}"
            f"_ag{self.alpha_gamma}"
            f"_sc{self.sky_conf}"
            f"_sb{self.sky_base}"
        )

    def label(self) -> str:
        """Human-readable one-line summary for collage annotation."""
        return (
            f"r={self.guided_radius}  eps={self.guided_eps:.0e}  "
            f"g={self.gamma}  blend={self.edge_blend}  band={self.edge_band_iters}  "
            f"ag={self.alpha_gamma}  sc={self.sky_conf}  sb={self.sky_base}"
        )


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    """Load image as BGR uint8 ndarray."""
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"Could not read image: {path}")
    return img


def _load_mask(path: Path) -> np.ndarray:
    """Load mask as float32 [0,1] grayscale ndarray."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Could not read mask: {path}")
    return (m / 255.0).astype(np.float32)


def _preprocess_mask(mask: np.ndarray, dilate_iters: int) -> np.ndarray:
    """Dilate binary mask to ensure coverage of soft edges before inpainting."""
    if dilate_iters <= 0:
        return mask
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = (mask * 255).astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=dilate_iters)
    return (dilated / 255.0).astype(np.float32)


def _inpaint_lama(
    lama: SimpleLama, image_bgr: np.ndarray, mask_float: np.ndarray
) -> np.ndarray:
    """Run LaMa inpainting; inputs/outputs are BGR uint8 ndarrays."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray((mask_float * 255).astype(np.uint8))
    result_pil = lama(pil_image, pil_mask)
    result_rgb = np.array(result_pil)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def _compute_diff(
    image_bgr: np.ndarray, inpainted_bgr: np.ndarray, gamma: float
) -> np.ndarray:
    """Absolute difference → grayscale → blur → float32 [0,1] with gamma boost.

    Divides by 255 (not diff.max()) so relative contrast is preserved and the
    sky centre does not swamp faint boundary edges.  Gamma < 1 lifts those
    faint edges (poles, thin branches) without clipping strong ones.
    """
    diff = cv2.absdiff(image_bgr, inpainted_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    diff_f = (diff_blur / 255.0).astype(np.float32)
    return np.power(diff_f, gamma)


def _make_edge_band(orig_mask: np.ndarray, band_iters: int) -> np.ndarray:
    """Build a soft band around the mask boundary (dilate − erode, then blur).

    Much wider and more stable than a 1-pixel Canny edge, so the diff signal
    is concentrated near the horizon without being zeroed out almost everywhere.
    """
    mask_u8 = (orig_mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=band_iters)
    eroded = cv2.erode(mask_u8, kernel, iterations=band_iters)
    band = (dilated.astype(np.float32) - eroded.astype(np.float32)) / 255.0
    return cv2.GaussianBlur(band, (5, 5), 0).astype(np.float32)


def _apply_edge_band(diff: np.ndarray, edge_band: np.ndarray, blend: float) -> np.ndarray:
    """Blend diff with edge_band focus: diff * (blend + (1-blend)*edge_band).

    blend=0 → pure edge_band masking; blend=1 → no spatial weighting.
    blend≈0.3 keeps some global signal while strongly emphasising the boundary.
    """
    weight = blend + (1.0 - blend) * edge_band
    return (diff * weight).astype(np.float32)


def _guided_filter(
    guide_bgr: np.ndarray,
    src_float: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Edge-aware smoothing via guided filter; returns float32 [0,1].

    Uses cv2.ximgproc.guidedFilter when available (opencv-contrib-python),
    otherwise falls back to a pure numpy/cv2 per-channel implementation.
    """
    if hasattr(cv2, "ximgproc"):
        src_u8 = (np.clip(src_float, 0.0, 1.0) * 255).astype(np.uint8)
        result = cv2.ximgproc.guidedFilter(
            guide=guide_bgr,
            src=src_u8,
            radius=radius,
            eps=eps,
        )
        return (result / 255.0).astype(np.float32)

    # Pure cv2 guided filter — per color channel, then average (He et al. 2013)
    # Running per B/G/R channel preserves color edge information that a grayscale
    # guide would lose (e.g. blue sky vs green tree look similar in gray).
    ksize = (2 * radius + 1, 2 * radius + 1)
    p = np.clip(src_float, 0.0, 1.0).astype(np.float32)

    channel_results = []
    for ch in cv2.split(guide_bgr):
        I = ch.astype(np.float32) / 255.0

        mean_I = cv2.blur(I,     ksize)
        mean_p = cv2.blur(p,     ksize)
        corr_I = cv2.blur(I * I, ksize)
        cov_Ip = cv2.blur(I * p, ksize)

        var_I  = corr_I - mean_I * mean_I
        cov_Ip = cov_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.blur(a, ksize)
        mean_b = cv2.blur(b, ksize)

        channel_results.append(mean_a * I + mean_b)

    return np.clip(np.mean(channel_results, axis=0), 0.0, 1.0).astype(np.float32)


def _shape_alpha(
    guided: np.ndarray,
    orig_mask: np.ndarray,
    alpha_gamma: float,
    sky_conf: float,
    sky_base: float,
) -> np.ndarray:
    """Post-filter alpha shaping: normalize → gamma boost → sky confidence floor.

    1. Normalize to full dynamic range so the guided filter's compressed output
       becomes a usable [0,1] signal.
    2. Apply gamma on alpha (more effective than diff gamma because the guided
       filter would smooth away any diff boost before this point).
    3. Enforce a sky confidence floor so the sky interior doesn't collapse to
       black after normalization: max(alpha, orig_mask * sky_conf + sky_base).
    """
    alpha = guided.copy()
    lo, hi = float(alpha.min()), float(alpha.max())
    if hi > lo:
        alpha = (alpha - lo) / (hi - lo + 1e-6)
    alpha = np.power(alpha, alpha_gamma)
    alpha = np.maximum(alpha, orig_mask * sky_conf + sky_base)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _add_label_strip(image: np.ndarray, label: str) -> np.ndarray:
    """Append a dark text strip below the image showing parameter values."""
    strip = np.zeros((_LABEL_H, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        strip, label, (_LABEL_PAD, _LABEL_H - 8),
        _FONT, _FONT_SCALE, (220, 220, 220), _FONT_THICKNESS, cv2.LINE_AA,
    )
    return np.vstack([image, strip])


def _create_collage(
    image_bgr: np.ndarray,
    mask_float: np.ndarray,
    alpha_float: np.ndarray,
    label: str = "",
) -> np.ndarray:
    """Horizontal stack: Original | Mask | Alpha, all same height.

    If label is provided a dark annotation strip is appended below.
    """
    h = image_bgr.shape[0]
    mask_bgr = cv2.cvtColor((mask_float * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    alpha_bgr = cv2.cvtColor((alpha_float * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    panels = []
    for panel in (image_bgr, mask_bgr, alpha_bgr):
        if panel.shape[0] != h:
            panel = cv2.resize(panel, (int(panel.shape[1] * h / panel.shape[0]), h))
        panels.append(panel)

    collage = np.hstack(panels)
    if label:
        collage = _add_label_strip(collage, label)
    return collage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _multi(type_fn: Any, default: list[Any]) -> dict[str, Any]:
    """Shared kwargs for sweep-capable arguments (nargs='+')."""
    return {"nargs": "+", "type": type_fn, "default": default}


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    base = root / "data/datasets/skyfinder_2026-04-09"

    p = argparse.ArgumentParser(
        description=(
            "Per-sample pipeline: LaMa inpainting → diff map → "
            "guided-filter alpha → alpha mask + debug collage.\n\n"
            "Tunable parameters (--guided-radius, --guided-eps, --gamma, "
            "--edge-band-iters, --edge-blend) accept multiple values; "
            "one output is produced for every configuration combination."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-n", "--count",
        type=int, default=5,
        help="Maximum number of images to process (default: 5).",
    )
    p.add_argument(
        "--input-rand",
        action="store_true", default=False,
        help="Sample --count images at random from the sorted list instead of "
             "always taking the first N.",
    )
    p.add_argument(
        "--input-image",
        type=Path, default=base / "images/train",
        help="Image file or directory of images.",
    )
    p.add_argument(
        "--input-mask",
        type=Path, default=base / "masks/train",
        help="Mask file or directory of masks (same stem as images).",
    )
    p.add_argument(
        "--output-alpha",
        type=Path, default=base / "alpha/train",
        help="Directory for alpha mask outputs.",
    )
    p.add_argument(
        "--output-collage",
        type=Path, default=base / "collages/train",
        help="Directory for debug collage outputs.",
    )
    p.add_argument(
        "--dilate-iters",
        type=int, default=1,
        help="Dilation iterations on the inpainting mask (not the edge band).",
    )

    # --- sweepable parameters ---
    p.add_argument(
        "--guided-radius",
        metavar="R", **_multi(int, [20]),
        help="Guided filter radius. Multiple values trigger a sweep (default: 20).",
    )
    p.add_argument(
        "--guided-eps",
        metavar="EPS", **_multi(float, [1e-4]),
        help="Guided filter epsilon. Multiple values trigger a sweep (default: 1e-4).",
    )
    p.add_argument(
        "--gamma",
        metavar="G", **_multi(float, [0.7]),
        help="Gamma applied to diff map (< 1 boosts faint edges). "
             "Multiple values trigger a sweep (default: 0.7).",
    )
    p.add_argument(
        "--edge-band-iters",
        metavar="ITERS", **_multi(int, [4]),
        help="Dilation/erosion iterations for edge band width. "
             "Multiple values trigger a sweep (default: 4).",
    )
    p.add_argument(
        "--edge-blend",
        metavar="B", **_multi(float, [0.3]),
        help="Base blend weight: diff * (blend + (1-blend)*edge_band). "
             "0 = pure boundary, 1 = no spatial weighting (default: 0.3).",
    )
    p.add_argument(
        "--alpha-gamma",
        metavar="AG", **_multi(float, [0.7]),
        help="Gamma applied to alpha after normalization (< 1 boosts mid-tones). "
             "Multiple values trigger a sweep (default: 0.7).",
    )
    p.add_argument(
        "--sky-conf",
        metavar="SC", **_multi(float, [0.2]),
        help="orig_mask coefficient in sky confidence floor: "
             "max(alpha, orig_mask * sky_conf + sky_base). "
             "Multiple values trigger a sweep (default: 0.2).",
    )
    p.add_argument(
        "--sky-base",
        metavar="SB", **_multi(float, [0.1]),
        help="Constant offset in sky confidence floor. "
             "Multiple values trigger a sweep (default: 0.1).",
    )
    return p.parse_args()


def _build_configs(args: argparse.Namespace) -> list[PipelineConfig]:
    return [
        PipelineConfig(r, e, g, band, blend, ag, sc, sb)
        for r, e, g, band, blend, ag, sc, sb in iterproduct(
            args.guided_radius,
            args.guided_eps,
            args.gamma,
            args.edge_band_iters,
            args.edge_blend,
            args.alpha_gamma,
            args.sky_conf,
            args.sky_base,
        )
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    input_image = args.input_image.expanduser().resolve()
    input_mask = args.input_mask.expanduser().resolve()
    out_alpha = args.output_alpha.expanduser().resolve()
    out_collage = args.output_collage.expanduser().resolve()

    # Collect image paths
    if input_image.is_dir():
        image_paths = _list_images(input_image, max(0, args.count), args.input_rand)
        if not image_paths:
            print(f"No images found under {input_image}", file=sys.stderr)
            return 1
    elif input_image.is_file():
        image_paths = [input_image]
    else:
        print(f"Input image not found: {input_image}", file=sys.stderr)
        return 1

    # Validate mask path
    if not (input_mask.is_dir() or input_mask.is_file()):
        print(f"Mask path not found: {input_mask}", file=sys.stderr)
        return 1
    mask_is_dir = input_mask.is_dir()

    # Build sorted (image, mask) pairs
    jobs: list[tuple[Path, Path]] = []
    for img_path in image_paths:
        if mask_is_dir:
            mask_path = _find_mask_file(img_path, input_mask)
            if mask_path is None:
                print(
                    f"No mask for {img_path.name} under {input_mask}; skipping.",
                    file=sys.stderr,
                )
                continue
        else:
            mask_path = input_mask
        jobs.append((img_path, mask_path))

    jobs.sort(key=lambda t: (_natural_sort_key(t[0]), _natural_sort_key(t[1])))

    configs = _build_configs(args)
    multi = len(configs) > 1

    if multi:
        print(f"Parameter sweep: {len(configs)} configurations × {len(jobs)} images "
              f"= {len(configs) * len(jobs)} outputs")

    out_alpha.mkdir(parents=True, exist_ok=True)
    out_collage.mkdir(parents=True, exist_ok=True)

    lama = SimpleLama()

    for img_path, mask_path in jobs:
        image_bgr = _load_image(img_path)
        orig_mask = _load_mask(mask_path)

        # Inpaint once per image; reused across all configs.
        proc_mask = _preprocess_mask(orig_mask, args.dilate_iters)
        inpainted_bgr = _inpaint_lama(lama, image_bgr, proc_mask)

        for cfg in configs:
            diff = _compute_diff(image_bgr, inpainted_bgr, cfg.gamma)

            edge_band = _make_edge_band(orig_mask, cfg.edge_band_iters)
            diff = _apply_edge_band(diff, edge_band, cfg.edge_blend)

            guided = _guided_filter(image_bgr, diff, cfg.guided_radius, cfg.guided_eps)
            alpha = _shape_alpha(
                guided, orig_mask, cfg.alpha_gamma, cfg.sky_conf, cfg.sky_base
            )

            suffix = f"__{cfg.key()}" if multi else ""
            stem = img_path.stem

            alpha_path = out_alpha / f"{stem}{suffix}.png"
            collage_path = out_collage / f"{stem}{suffix}{img_path.suffix}"

            cv2.imwrite(str(alpha_path), (alpha * 255).astype(np.uint8))
            cv2.imwrite(
                str(collage_path),
                _create_collage(image_bgr, orig_mask, alpha, label=cfg.label()),
            )

            print(f"[alpha]   {alpha_path}")
            print(f"[collage] {collage_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
