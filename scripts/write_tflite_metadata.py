#!/usr/bin/env python3
"""Write ``metadata.yaml`` (and optional ``labels.txt``) next to an existing ``.tflite`` file.

Uses the TFLite interpreter to read tensor names, shapes, and dtypes — same layout as
``src/export_tflite.py`` produces for Keras exports.

Example::

  python scripts/write_tflite_metadata.py --tflite models/out/tflite/model_fp16.tflite
  python scripts/write_tflite_metadata.py --tflite model.tflite --output-dir models/out/bundle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

DEFAULT_QAIRT_VERSION = "2.43.0.260127150333_193827"
LABELS_TXT = """# Sky segmentation: one label per class index (for interpreting thresholded masks).
# Model output is [1,H,W,1] float in [0,1] (sigmoid); values > 0.5 => class 1 (sky).
not_sky
sky
"""


def _sanitize_io_name(name: str) -> str:
    base = name.split("/")[0].split(":")[0]
    return base if base else "tensor"


def _shape_list(shape: object) -> list[int]:
    if hasattr(shape, "tolist"):
        return [int(x) for x in shape.tolist()]
    return [int(x) for x in shape]


def _dtype_str(dtype: object) -> str:
    return str(np.dtype(dtype).name)


def _tflite_version_string() -> str:
    v = tf.__version__
    return v.split()[0] if v else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, required=True, help="Path to .tflite file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metadata.yaml (default: same folder as .tflite)",
    )
    parser.add_argument(
        "--qairt-version",
        type=str,
        default=DEFAULT_QAIRT_VERSION,
        help="Recorded under tool_versions.qairt",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not write labels.txt",
    )
    args = parser.parse_args()

    tflite_path = args.tflite.expanduser().resolve()
    if not tflite_path.is_file():
        print(f"Not found: {tflite_path}", file=sys.stderr)
        raise SystemExit(1)

    out_dir = args.output_dir or tflite_path.parent
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to load TFLite: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    tflite_basename = tflite_path.name
    inputs: dict[str, dict] = {}
    outputs: dict[str, dict] = {}

    for d in interpreter.get_input_details():
        name = _sanitize_io_name(d["name"])
        inputs[name] = {
            "shape": _shape_list(d["shape"]),
            "dtype": _dtype_str(d["dtype"]),
        }

    for d in interpreter.get_output_details():
        name = _sanitize_io_name(d["name"])
        outputs[name] = {
            "shape": _shape_list(d["shape"]),
            "dtype": _dtype_str(d["dtype"]),
        }

    metadata = {
        "runtime": "tflite",
        "precision": "float",
        "tool_versions": {
            "qairt": args.qairt_version,
            "tflite": _tflite_version_string(),
        },
        "model_files": {
            tflite_basename: {
                "inputs": inputs,
                "outputs": outputs,
            }
        },
    }

    supplementary: dict[str, str] = {}
    if not args.no_labels:
        supplementary["labels.txt"] = (
            "Mapping of model prediction indices -> string labels."
        )
        (out_dir / "labels.txt").write_text(LABELS_TXT, encoding="utf-8")

    if supplementary:
        metadata["supplementary_files"] = supplementary

    meta_path = out_dir / "metadata.yaml"
    meta_path.write_text(
        yaml.dump(metadata, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Wrote {meta_path}")
    if not args.no_labels:
        print(f"Wrote {out_dir / 'labels.txt'}")


if __name__ == "__main__":
    main()
