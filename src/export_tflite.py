from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

import train as train_mod

DEFAULT_QAIRT_VERSION = "2.43.0.260127150333_193827"
TFLITE_OUT_BASE = Path("models/out/tflite")
LABELS_TXT = """# Sky segmentation: one label per class index (for interpreting thresholded masks).
# Model output is [1,H,W,1] float in [0,1] (sigmoid); values > 0.5 => class 1 (sky).
not_sky
sky
"""

_IMAGE_EXT = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Helpers shared between FP16 and int8 paths
# ---------------------------------------------------------------------------

def _default_model_from_config(config_path: Path) -> Path:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["output"]["best_model_path"])


def _sanitize_io_name(name: str) -> str:
    base = name.split("/")[0].split(":")[0]
    return base if base else "tensor"


def _fixed_shape(shape: tuple[int | None, ...], *, batch: int = 1) -> list[int]:
    out: list[int] = []
    for i, d in enumerate(shape):
        if d is None:
            out.append(batch if i == 0 else 1)
        else:
            out.append(int(d))
    return out


def _tflite_version_string() -> str:
    v = tf.__version__
    return v.split()[0] if v else "unknown"


# ---------------------------------------------------------------------------
# Calibration dataset for int8 PTQ
# ---------------------------------------------------------------------------

def _load_calibration_image(path: Path, img_size: int) -> np.ndarray:
    """Load a single image and apply the same [-1, 1] normalization used in training."""
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as im:
        im = im.convert("RGB").resize((img_size, img_size))
        arr = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0  # [-1, 1]
    return arr


def _representative_data_gen(image_dir: Path, n: int, img_size: int):
    """Yield single-image batches [1, H, W, 3] float32 for TFLite int8 calibration."""
    candidates = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXT
    )
    if not candidates:
        raise FileNotFoundError(
            f"No images ({', '.join(_IMAGE_EXT)}) found in representative data dir: {image_dir}"
        )
    paths = candidates[:n]
    print(
        f"[int8-calib] Using {len(paths)} images from {image_dir} for calibration.",
        flush=True,
    )
    for p in paths:
        img = _load_calibration_image(p, img_size)
        yield [img[np.newaxis]]  # shape [1, H, W, 3]


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _convert_fp16(model: keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def _convert_int8(
    model: keras.Model,
    representative_data_dir: Path,
    num_calibration_samples: int,
    img_size: int,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_data_gen(
        representative_data_dir, num_calibration_samples, img_size
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained .keras segmentation model to TFLite. "
            "Supports FP16 (default) and int8 PTQ quantization."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="If set without --model, use output.best_model_path from this YAML.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .keras model (default: from --config or models/best_model.keras).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Parent for export folder (default: {TFLITE_OUT_BASE}).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Subfolder name under output-dir (default: timestamp YYYY-mm-dd_HH-MM-SS). "
            "The .tflite basename defaults to <run-name>.tflite when set."
        ),
    )
    parser.add_argument(
        "--tflite-name",
        type=str,
        default=None,
        help="Basename for the .tflite file (default: <keras_stem>_<quantization>.tflite).",
    )
    parser.add_argument(
        "--quantization",
        choices=["fp16", "int8"],
        default="fp16",
        help="Quantization mode: 'fp16' (default) or 'int8' PTQ.",
    )
    parser.add_argument(
        "--representative-data",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Directory of images used to calibrate int8 PTQ (required when "
            "--quantization=int8). Typically the training images folder."
        ),
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=200,
        metavar="N",
        help="Number of images to use for int8 calibration (default: 200).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help=(
            "Input image size used during calibration for int8 (default: inferred from "
            "model input shape, falls back to 320)."
        ),
    )
    parser.add_argument(
        "--qairt-version",
        type=str,
        default=DEFAULT_QAIRT_VERSION,
        help="Value recorded under tool_versions.qairt in metadata.yaml.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve model path
    # ------------------------------------------------------------------
    model_path = args.model
    if model_path is None:
        if args.config is not None:
            cfg_path = args.config.expanduser().resolve()
            if not cfg_path.is_file():
                print(f"Config not found: {cfg_path}", file=sys.stderr)
                sys.exit(1)
            model_path = _default_model_from_config(cfg_path)
        else:
            model_path = Path("models/best_model.keras")

    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Validate int8 requirements
    # ------------------------------------------------------------------
    if args.quantization == "int8" and args.representative_data is None:
        print(
            "Error: --representative-data is required when --quantization=int8.",
            file=sys.stderr,
        )
        sys.exit(1)

    rep_data_dir: Path | None = None
    if args.representative_data is not None:
        rep_data_dir = args.representative_data.expanduser().resolve()
        if not rep_data_dir.is_dir():
            print(
                f"Representative data directory not found: {rep_data_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    base_out = args.output_dir or TFLITE_OUT_BASE
    base_out = base_out.expanduser().resolve()
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir = base_out / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    quant_tag = args.quantization  # "fp16" or "int8"

    if args.tflite_name:
        tflite_basename = args.tflite_name
        if not tflite_basename.endswith(".tflite"):
            tflite_basename = f"{tflite_basename}.tflite"
    else:
        tflite_basename = f"{model_path.stem}_{quant_tag}.tflite"

    tflite_path = export_dir / tflite_basename
    metadata_path = export_dir / "metadata.yaml"
    labels_path = export_dir / "labels.txt"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading:", model_path)
    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects=train_mod.keras_custom_objects(),
            compile=False,
        )
    except Exception as e:
        print(f"Failed to load model (custom_objects / format): {e}", file=sys.stderr)
        sys.exit(1)

    in_tensor = model.input
    out_tensor = model.output
    in_name = _sanitize_io_name(in_tensor.name)
    out_name = _sanitize_io_name(out_tensor.name)
    in_shape = _fixed_shape(tuple(model.input_shape))
    out_shape = _fixed_shape(tuple(model.output_shape))

    # Determine img_size for calibration
    img_size_for_calib: int = args.img_size or (in_shape[1] if in_shape[1] else 320)

    print("Input shape:", in_shape, f"({in_name})")
    print("Output shape:", out_shape, f"({out_name})")
    print("Quantization:", quant_tag)

    # ------------------------------------------------------------------
    # Convert
    # ------------------------------------------------------------------
    try:
        if args.quantization == "int8":
            print(
                f"[int8-calib] Calibrating with up to {args.num_calibration_samples} "
                f"samples at {img_size_for_calib}×{img_size_for_calib}...",
                flush=True,
            )
            tflite_model = _convert_int8(
                model,
                representative_data_dir=rep_data_dir,
                num_calibration_samples=args.num_calibration_samples,
                img_size=img_size_for_calib,
            )
        else:
            tflite_model = _convert_fp16(model)
    except Exception as e:
        print(f"TFLite conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    tflite_path.write_bytes(tflite_model)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    # Precision label: int8 uses fixed-point; fp16 is still floating-point
    precision_label = "int8" if args.quantization == "int8" else "float16"

    metadata = {
        "runtime": "tflite",
        "quantization": quant_tag,
        "precision": precision_label,
        "tool_versions": {
            "qairt": args.qairt_version,
            "tflite": _tflite_version_string(),
        },
        "model_files": {
            tflite_basename: {
                "inputs": {
                    in_name: {
                        "shape": in_shape,
                        # int8 model expects int8 tensors on device
                        "dtype": "int8" if args.quantization == "int8" else "float32",
                    }
                },
                "outputs": {
                    out_name: {
                        "shape": out_shape,
                        "dtype": "int8" if args.quantization == "int8" else "float32",
                    }
                },
            }
        },
        "supplementary_files": {
            "labels.txt": "Mapping of model prediction indices -> string labels.",
        },
    }

    metadata_path.write_text(
        yaml.dump(metadata, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    labels_path.write_text(LABELS_TXT, encoding="utf-8")

    print(f"Saved bundle under {export_dir}")
    print(f"  {tflite_path.name} ({len(tflite_model) / 1e6:.2f} MB)")
    print(f"  {metadata_path.name}")
    print(f"  {labels_path.name}")


if __name__ == "__main__":
    main()
