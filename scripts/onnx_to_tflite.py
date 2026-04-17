#!/usr/bin/env python3
"""Convert an ONNX model to TFLite using onnx2tf.

Pipeline: ONNX → (onnx2tf) → ``*_float32.tflite`` and ``*_float16.tflite`` in a work
directory, then copy the chosen precision to your destination.

Install: ``pip install -r requirements.txt`` (or at minimum
``onnx2tf tf-keras onnx onnx-graphsurgeon onnxruntime``).
Optional: ``onnxsim`` (from PyPI as ``onnxsim`` / ``onnx-simplifier``) if you have a prebuilt wheel
or cmake for source builds — onnx2tf only uses it for an optional simplify step.

onnx2tf downloads a fixed-name calibration ``.npy`` into the **process working
directory** for 4D RGB inputs; failed downloads or a corrupt cache file can make
``numpy.load`` raise (pickle / invalid file). This script writes a valid
``calibration_image_sample_data_20x128x128x3_float32.npy`` into the work dir
and runs onnx2tf with ``cwd`` there so that download path is skipped.

Example::

  python scripts/onnx_to_tflite.py \\
    --onnx path/to/model.onnx \\
    --output models/out/tflite/my_model_fp16.tflite

Segmentation / dynamic shapes (often fixes onnx2tf ``Resize`` / ``tf.image.resize`` errors)::

  python scripts/onnx_to_tflite.py --onnx model.onnx --output out.tflite \\
    --static-nchw 224

See also ``--print-inputs`` and explicit ``--ois name:1,3,H,W``. If conversion still
fails, re-run with ``--keep-work-dir ./tmp_onnx`` and use onnx2tf's ``-prf`` with any
``*_auto.json`` it writes there.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# onnx2tf.utils.common_functions.download_test_image_data looks for this exact name in os.getcwd().
_ONNX2TF_CALIB_NPY = "calibration_image_sample_data_20x128x128x3_float32.npy"


def _seed_onnx2tf_calibration_npy(work_dir: Path) -> None:
    """Write a standard float32 NHWC bundle onnx2tf expects (resized per model internally)."""
    path = work_dir / _ONNX2TF_CALIB_NPY
    data = np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32)
    np.save(path, data)


def _print_onnx_inputs(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    init_names = {x.name for x in model.graph.initializer}
    print(f"ONNX inputs ({onnx_path.name}):", flush=True)
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        dims: list[str] = []
        for d in tt.shape.dim:
            if d.dim_value:
                dims.append(str(d.dim_value))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        dt = onnx.TensorProto.DataType.Name(tt.elem_type)
        print(f"  {inp.name}: [{', '.join(dims)}] {dt}", flush=True)


def _static_nchw_ois(onnx_path: Path, hw: int, batch: int) -> str:
    """Build ``-ois`` for the first 4D non-initializer input (NCHW, square H=W)."""
    import onnx

    model = onnx.load(str(onnx_path))
    init_names = {x.name for x in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        dims = list(tt.shape.dim)
        if len(dims) != 4:
            continue
        ch = dims[1].dim_value if dims[1].dim_value else 3
        return f"{inp.name}:{batch},{ch},{hw},{hw}"
    raise SystemExit(
        "No 4D graph input found for --static-nchw; use --print-inputs and --ois manually."
    )


def _find_onnx2tf() -> list[str]:
    """Return argv prefix to invoke onnx2tf (console script or ``python -m``)."""
    exe = shutil.which("onnx2tf")
    if exe:
        return [exe]
    return [sys.executable, "-m", "onnx2tf"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Input .onnx file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write this .tflite path (default: "
            "models/out/tflite/onnx/<run>/<onnx_stem>_fp16.tflite)"
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32", "both"),
        default="fp16",
        help="Which TFLite variant to copy to --output (default: fp16). "
        "Use 'both' with --output-dir instead of --output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set with --precision both, copy both tflite files here.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder under the default parent when --output is omitted.",
    )
    parser.add_argument(
        "--saved-model",
        action="store_true",
        help="Also emit TensorFlow SavedModel in the work dir (-fdosm).",
    )
    parser.add_argument(
        "--tflite-backend",
        choices=("flatbuffer_direct", "tf_converter"),
        default=None,
        help=(
            "If set, pass ``-tb`` to onnx2tf (only some versions expose this flag; "
            "omit to use onnx2tf defaults)."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        choices=("debug", "info", "warn", "error"),
        default="info",
        help="onnx2tf log level (default: info).",
    )
    parser.add_argument(
        "--keep-work-dir",
        type=Path,
        default=None,
        help="Use this folder for onnx2tf -o and do not delete it (for debugging).",
    )
    parser.add_argument(
        "--print-inputs",
        action="store_true",
        help="Print ONNX graph input names and shapes, then exit.",
    )
    parser.add_argument(
        "--static-nchw",
        type=int,
        metavar="HW",
        default=None,
        help=(
            "Pass onnx2tf -ois for the first 4D input as "
            "NAME:BATCH,C,HW,HW (C from model or 3). Combine with --batch (default 1)."
        ),
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=None,
        help="Forwarded as onnx2tf -b (batch); also used as B in --static-nchw (default 1).",
    )
    parser.add_argument(
        "--ois",
        action="append",
        default=[],
        metavar="NAME:DIM,DIM,...",
        help="Forwarded as onnx2tf -ois (repeatable), e.g. --ois input:1,3,224,224",
    )
    parser.add_argument(
        "onnx2tf_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args after -- are passed to onnx2tf (e.g. -- -ois input:1,3,224,224).",
    )
    args = parser.parse_args()
    if args.onnx2tf_args and args.onnx2tf_args[0] == "--":
        args.onnx2tf_args = args.onnx2tf_args[1:]

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        print(f"ONNX file not found: {onnx_path}", file=sys.stderr)
        raise SystemExit(1)

    if args.print_inputs:
        _print_onnx_inputs(onnx_path)
        raise SystemExit(0)

    stem = onnx_path.stem
    cmd_prefix = _find_onnx2tf()
    cleanup: Path | None = None
    if args.keep_work_dir is not None:
        work_dir = args.keep_work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = Path(tempfile.mkdtemp(prefix="onnx2tf_"))
        work_dir = cleanup

    cmd = [
        *cmd_prefix,
        "-i",
        str(onnx_path),
        "-o",
        str(work_dir),
    ]
    if args.tflite_backend is not None:
        cmd.extend(["-tb", args.tflite_backend])

    if args.static_nchw is not None:
        batch = int(args.batch) if args.batch is not None else 1
        cmd.extend(["-ois", _static_nchw_ois(onnx_path, args.static_nchw, batch)])
    for spec in args.ois:
        cmd.extend(["-ois", spec])
    if args.batch is not None and args.static_nchw is None:
        cmd.extend(["-b", str(args.batch)])

    cmd.extend(["-v", args.verbosity, *args.onnx2tf_args])
    if args.saved_model:
        cmd.append("-fdosm")

    _seed_onnx2tf_calibration_npy(work_dir)

    print("Running:", " ".join(cmd), flush=True)
    print(f"(cwd for onnx2tf: {work_dir})", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(work_dir))
    except FileNotFoundError:
        print(
            "onnx2tf not found. Install with: pip install onnx2tf",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as e:
        print("onnx2tf failed.", file=sys.stderr)
        print(
            f"Work dir (artifacts, any *_auto.json for -prf): {work_dir}",
            file=sys.stderr,
        )
        print(
            "If you saw Resize/tf.image.resize errors, try static shapes, e.g.:",
            file=sys.stderr,
        )
        print(
            f"  python scripts/onnx_to_tflite.py --onnx {onnx_path} --output ... "
            "--static-nchw 224",
            file=sys.stderr,
        )
        print(
            "Or: --print-inputs, then --ois YOUR_INPUT:1,3,H,W",
            file=sys.stderr,
        )
        if not shutil.which("onnxsim"):
            print(
                "Optional: `onnxsim` missing — onnx2tf skips ONNX simplify (usually fine). "
                "For it: install cmake then `pip install onnxsim`, or use a platform wheel.",
                file=sys.stderr,
            )
        raise SystemExit(e.returncode) from e

    fp32_name = f"{stem}_float32.tflite"
    fp16_name = f"{stem}_float16.tflite"
    fp32_path = work_dir / fp32_name
    fp16_path = work_dir / fp16_name

    missing = [p for p in (fp32_path, fp16_path) if not p.is_file()]
    if missing:
        print(
            "Expected TFLite file(s) not found after conversion. "
            f"Look in {work_dir} — onnx2tf may use different naming for this graph.",
            file=sys.stderr,
        )
        listed = list(work_dir.glob("*.tflite"))
        if listed:
            print("Found .tflite files:", *[f"  {p}" for p in sorted(listed)], sep="\n")
        if cleanup and cleanup.exists():
            shutil.rmtree(cleanup, ignore_errors=True)
        raise SystemExit(1)

    if args.precision == "both":
        out_parent = args.output_dir
        if out_parent is None:
            print(
                "With --precision both, set --output-dir to copy both files.",
                file=sys.stderr,
            )
            if cleanup and cleanup.exists():
                shutil.rmtree(cleanup, ignore_errors=True)
            raise SystemExit(1)
        out_parent = out_parent.expanduser().resolve()
        out_parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp32_path, out_parent / fp32_name)
        shutil.copy2(fp16_path, out_parent / fp16_name)
        print(f"Copied:\n  {out_parent / fp32_name}\n  {out_parent / fp16_name}")
    else:
        if args.precision == "fp16":
            chosen = fp16_path
        else:
            chosen = fp32_path

        if args.output is not None:
            dest = args.output.expanduser().resolve()
        else:
            run = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            parent = Path("models/out/tflite/onnx") / run
            parent.mkdir(parents=True, exist_ok=True)
            suffix = "fp16" if args.precision == "fp16" else "fp32"
            dest = parent / f"{stem}_{suffix}.tflite"

        if dest.suffix.lower() != ".tflite":
            dest = dest / f"{stem}_{'fp16' if args.precision == 'fp16' else 'fp32'}.tflite"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chosen, dest)
        print(f"Wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB)", flush=True)

    if cleanup is not None and cleanup.exists():
        shutil.rmtree(cleanup, ignore_errors=True)
    elif args.keep_work_dir is not None:
        print(f"onnx2tf artifacts kept under {work_dir}", flush=True)


if __name__ == "__main__":
    main()
