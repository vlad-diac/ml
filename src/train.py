from __future__ import annotations

import argparse
import csv as _csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

from dataset import get_pairs, make_tf_dataset
from model import build_model


def dice_loss(y_true, y_pred, smooth=1e-6):
    """Global (batch) Dice loss — single overlap term over the whole batch."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * tf.reduce_mean(bce) + 0.5 * dice_loss(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Alias for older checkpoints; same scalar loss as combined_loss."""
    return combined_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, threshold=0.5):
    """Kept for loading older checkpoints that serialised this function."""
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(iou)


def _make_binary_iou() -> tf.keras.metrics.BinaryIoU:
    """Factory for BinaryIoU targeting the sky class (class 1)."""
    return tf.keras.metrics.BinaryIoU(
        target_class_ids=[1], threshold=0.5, name="binary_iou"
    )


def _denormalize(image: np.ndarray) -> np.ndarray:
    """Convert [-1, 1] normalized image to [0, 1] for plt.imshow."""
    return np.clip(image * 0.5 + 0.5, 0.0, 1.0)


def keras_custom_objects() -> dict:
    return {
        "dice_loss": dice_loss,
        "combined_loss": combined_loss,
        "bce_dice_loss": bce_dice_loss,
        "iou_metric": iou_metric,
        "BinaryIoU": tf.keras.metrics.BinaryIoU,
    }


def _has_cuda_gpu() -> bool:
    """Return True if nvidia-smi exits successfully (NVIDIA GPU + driver present)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _setup_hardware(train_cfg: dict) -> None:
    """Configure mixed precision and XLA based on config + auto-detected hardware.

    mixed_precision values: auto | mixed_float16 | mixed_bfloat16 | float32
    xla values: auto | true | false

    auto (both): CUDA GPU detected  → mixed_float16 + XLA
                 no CUDA detected   → mixed_float16 only (safe for Metal / CPU)
    """
    mp_cfg = str(train_cfg.get("mixed_precision", "auto")).strip().lower()
    xla_cfg = str(train_cfg.get("xla", "auto")).strip().lower()

    cuda = _has_cuda_gpu()
    hw_label = "CUDA GPU" if cuda else "CPU / Apple Metal"
    print(f"[hardware] Detected hardware: {hw_label}")

    # Mixed precision policy
    if mp_cfg == "auto":
        policy = "mixed_float16"
    elif mp_cfg in ("mixed_float16", "mixed_bfloat16", "float32"):
        policy = mp_cfg
    else:
        print(f"[hardware] Unknown mixed_precision={mp_cfg!r}; defaulting to mixed_float16")
        policy = "mixed_float16"

    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"[hardware] Mixed precision policy: {policy}")

    # XLA (JIT compilation — only beneficial with CUDA; can cause issues on Metal)
    enable_xla: bool
    if xla_cfg == "auto":
        enable_xla = cuda
    elif xla_cfg == "true":
        enable_xla = True
    else:
        enable_xla = False

    if enable_xla:
        tf.config.optimizer.set_jit(True)
        print("[hardware] XLA (JIT) enabled")
    else:
        print("[hardware] XLA (JIT) disabled")


def _get_prev_session_dir(resume_path: Path, meta: dict) -> Path | None:
    """Derive the previous session folder from meta.json or from the checkpoint path."""
    if meta.get("session_dir"):
        p = Path(meta["session_dir"])
        if p.is_dir():
            return p
    # Fallback: epoch checkpoint is one level inside the session root.
    # e.g. models/17-04-2026-09-19-26/epoch_ckpt_ds/e02_...keras
    #      → parent.parent = models/17-04-2026-09-19-26/
    candidate = resume_path.parent.parent
    if candidate.is_dir():
        return candidate
    return None


def _get_prev_log_csv(
    meta: dict, prev_session_dir: Path | None, dataset_name: str | None
) -> Path | None:
    """Find the previous session's log CSV via meta.json or known naming patterns."""
    if meta.get("log_csv_path"):
        p = Path(meta["log_csv_path"])
        if p.is_file():
            return p
    if prev_session_dir is not None:
        candidates = [
            _with_dataset_suffix(prev_session_dir / "training_log.csv", dataset_name),
            prev_session_dir / "training_log.csv",
        ]
        for p in candidates:
            if p.is_file():
                return p
    return None


def _seed_csv_from_prev(
    new_csv: Path, prev_log_csv: Path | None, finished_epoch_0idx: int
) -> bool:
    """Write the baseline row (last completed epoch) into the new session CSV.

    The new CSV gets the header + that single seed row so that CSVLogger
    (opened with append=True) can continue writing from the correct state.
    Returns True if the seed was written successfully.
    """
    if prev_log_csv is None or not prev_log_csv.is_file():
        return False
    try:
        with prev_log_csv.open(newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except Exception as exc:
        print(f"[train] Warning: could not read previous CSV for seeding: {exc}", flush=True)
        return False
    if not rows:
        return False

    seed_row: dict | None = None
    for row in rows:
        try:
            if int(float(row.get("epoch", -1))) == finished_epoch_0idx:
                seed_row = row
                break
        except (ValueError, TypeError):
            continue
    if seed_row is None:
        seed_row = rows[-1]  # fallback: last row in file

    try:
        new_csv.parent.mkdir(parents=True, exist_ok=True)
        with new_csv.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(seed_row.keys()))
            writer.writeheader()
            writer.writerow(seed_row)
        print(
            f"[train] Seeded new CSV with epoch={seed_row.get('epoch')} "
            f"row from {prev_log_csv.name}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[train] Warning: could not write seed CSV: {exc}", flush=True)
        return False


def _copy_prev_debug_viz(
    prev_session_dir: Path | None, initial_epoch: int, new_viz_dir: Path
) -> None:
    """Copy epoch_{initial_epoch:02d}_val_sample_*.png from the previous session folder.

    `initial_epoch` is the epoch index training will start at; the last
    *completed* epoch produced files named epoch_{initial_epoch:02d}_*.png
    (because human_epoch = epoch_0idx + 1, and checkpoint e{N} → initial_epoch N+1).
    """
    if prev_session_dir is None:
        return
    prev_viz_dir = prev_session_dir / "debug_viz"
    if not prev_viz_dir.is_dir():
        return
    pattern = f"epoch_{initial_epoch:02d}_val_sample_*.png"
    files = sorted(prev_viz_dir.glob(pattern))
    if not files:
        print(
            f"[debug-viz] No files matching {pattern} found in {prev_viz_dir}; "
            "skipping copy.",
            flush=True,
        )
        return
    new_viz_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = new_viz_dir / src.name
        shutil.copy2(src, dst)
        print(f"[debug-viz] Copied from prev session: {src.name}", flush=True)


def _safe_dataset_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Dataset name must be non-empty")
    if name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Invalid dataset name: {name!r}")
    return name


def _with_dataset_suffix(path: Path, dataset: str | None) -> Path:
    if dataset is None:
        return path
    slug = _safe_dataset_name(dataset)
    return path.with_name(f"{path.stem}_{slug}{path.suffix}")


def _dataset_slug(dataset_name: str | None) -> str:
    return _safe_dataset_name(dataset_name) if dataset_name is not None else "default"


def _read_resume_initial_epoch(resume_path: Path) -> int:
    meta_path = resume_path.with_name(resume_path.stem + ".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["initial_epoch"])
    m = re.match(r"^e(\d{2})_", resume_path.name)
    if m:
        return int(m.group(1), 10) + 1
    print(
        f"Warning: no {meta_path.name} beside checkpoint; assuming initial_epoch=0 "
        "(set initial_epoch correctly or add .meta.json from training)."
    )
    return 0


class _PerEpochCheckpointWithMeta(keras.callbacks.Callback):
    """Save full model each epoch and write a sidecar .meta.json for resuming."""

    def __init__(
        self,
        directory: Path,
        *,
        img_size: int,
        batch_size: int,
        dataset_slug: str,
        epochs: int,
        config_path: Path,
        dataset_name: str | None,
        learning_rate: float,
        backbone: str,
        head: str,
        pretrained: str | None,
        freeze_backbone: bool,
        session_dir: Path,
        log_csv_path: Path,
    ) -> None:
        super().__init__()
        self._directory = directory
        self._img_size = img_size
        self._batch_size = batch_size
        self._dataset_slug = dataset_slug
        self._epochs = epochs
        self._config_path = config_path
        self._dataset_name = dataset_name
        self._learning_rate = learning_rate
        self._backbone = backbone
        self._head = head
        self._pretrained = pretrained
        self._freeze_backbone = freeze_backbone
        self._session_dir = session_dir
        self._log_csv_path = log_csv_path

    def on_epoch_end(self, epoch, logs=None) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        fname = (
            f"e{epoch:02d}_sz{self._img_size}_bs{self._batch_size}_ds{self._dataset_slug}.keras"
        )
        ckpt_path = self._directory / fname
        human_epoch = epoch + 1
        print(
            f"[checkpoint] Saving epoch model: epoch {human_epoch}/{self._epochs} -> {ckpt_path.name}",
            flush=True,
        )
        self.model.save(str(ckpt_path))
        meta = {
            "initial_epoch": epoch + 1,
            "epochs": self._epochs,
            "config_path": str(self._config_path.resolve()),
            "dataset_name": self._dataset_name,
            "img_size": self._img_size,
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "backbone": self._backbone,
            "head": self._head,
            "pretrained": self._pretrained,
            "freeze_backbone": self._freeze_backbone,
            "weights_path": str(ckpt_path.resolve()),
            "finished_epoch": epoch,
            "session_dir": str(self._session_dir.resolve()),
            "log_csv_path": str(self._log_csv_path.resolve()),
        }
        meta_path = ckpt_path.with_name(ckpt_path.stem + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"[checkpoint] Wrote {ckpt_path.resolve()} and meta {meta_path.name}",
            flush=True,
        )


def _format_metric_for_log(v: object) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{x:.4f}"


class _EpochMetricsTerminalLogger(keras.callbacks.Callback):
    """Print epoch metrics to the terminal (CSVLogger still writes the CSV)."""

    def __init__(self, total_epochs: int) -> None:
        super().__init__()
        self._total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        ne = epoch + 1
        parts = [f"{k}={_format_metric_for_log(v)}" for k, v in sorted(logs.items())]
        tail = " ".join(parts) if parts else "(no metrics)"
        print(
            f"[metrics] Epoch {ne}/{self._total_epochs} — {tail}",
            flush=True,
        )


class _ValDebugVizCallback(keras.callbacks.Callback):
    """After each epoch (optionally throttled), save val batch predictions for inspection."""

    def __init__(
        self,
        val_ds: tf.data.Dataset,
        out_dir: Path,
        *,
        show_plot: bool,
        every_n_epochs: int = 1,
        total_epochs: int,
    ) -> None:
        super().__init__()
        self._val_ds = val_ds
        self._out_dir = out_dir
        self._show_plot = show_plot
        self._every_n = max(int(every_n_epochs), 1)
        self._total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None) -> None:
        human_epoch = epoch + 1
        if human_epoch % self._every_n != 0:
            return
        import matplotlib.pyplot as plt

        print(
            f"[debug-viz] Generating validation sample panels: epoch {human_epoch}/"
            f"{self._total_epochs} (saving under {self._out_dir})",
            flush=True,
        )
        self._out_dir.mkdir(parents=True, exist_ok=True)
        for images, masks in self._val_ds.take(1):
            preds = self.model.predict(images, verbose=0)
            n = min(3, int(tf.shape(images)[0]))
            img_np = images.numpy() if isinstance(images, tf.Tensor) else images
            mask_np = masks.numpy() if isinstance(masks, tf.Tensor) else masks
            for i in range(n):
                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                axes[0].imshow(_denormalize(img_np[i]))
                axes[0].set_title("Image")
                axes[1].imshow(np.squeeze(mask_np[i]))
                axes[1].set_title("GT")
                axes[2].imshow(np.squeeze(preds[i]))
                axes[2].set_title("Pred")
                for ax in axes:
                    ax.axis("off")
                out_path = self._out_dir / f"epoch_{human_epoch:02d}_val_sample_{i}.png"
                fig.savefig(out_path, bbox_inches="tight")
                print(
                    f"[debug-viz] Saved sample {i + 1}/{n}: {out_path.resolve()}",
                    flush=True,
                )
                if self._show_plot:
                    plt.show()
                plt.close(fig)
        print(
            f"[debug-viz] Finished epoch {human_epoch} visualization batch",
            flush=True,
        )


def main(
    config_path: Path,
    dataset_name: str | None = None,
    *,
    debug_viz: bool = False,
    resume_from: Path | None = None,
) -> None:
    config_path = config_path.resolve()
    with config_path.open() as f:
        config = yaml.safe_load(f)

    data_cfg = config["dataset"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    out_cfg = config["output"]

    base = Path(data_cfg["path"])
    if dataset_name is not None:
        slug = _safe_dataset_name(dataset_name)
        root = base / "datasets" / slug
    else:
        root = base

    img_size = int(data_cfg["img_size"])
    batch_size = int(data_cfg["batch_size"])
    shuffle_buffer = int(data_cfg.get("shuffle_buffer", 1024))
    drop_remainder = bool(data_cfg.get("drop_remainder", False))
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    freeze_backbone = bool(train_cfg["freeze_backbone"])

    resume_path = resume_from
    if resume_path is None and train_cfg.get("resume_from"):
        resume_path = Path(str(train_cfg["resume_from"]))
    if resume_path is not None:
        resume_path = resume_path.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    timestamp_run_dir = bool(out_cfg.get("timestamp_run_dir", True))
    models_run_root: Path | None = None
    if timestamp_run_dir:
        models_run_root = Path("models") / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        models_run_root.mkdir(parents=True, exist_ok=True)

    def _model_output_path(cfg_value: str) -> Path:
        p = Path(cfg_value)
        if models_run_root is not None:
            return models_run_root / p.name
        return p

    best_path = _with_dataset_suffix(
        _model_output_path(out_cfg["best_model_path"]), dataset_name
    )
    final_path = _with_dataset_suffix(
        _model_output_path(out_cfg["model_path"]), dataset_name
    )
    # When using a session folder, co-locate the CSV there; otherwise use the
    # configured path so that timestamp_run_dir=false keeps the old behaviour.
    _log_csv_base = Path(out_cfg.get("log_csv", "logs/training_log.csv"))
    if models_run_root is not None:
        log_csv = _with_dataset_suffix(models_run_root / _log_csv_base.name, dataset_name)
    else:
        log_csv = _with_dataset_suffix(_log_csv_base, dataset_name)

    for p in (best_path.parent, final_path.parent, log_csv.parent):
        p.mkdir(parents=True, exist_ok=True)

    _setup_hardware(train_cfg)

    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())
    print("Data root:", root.resolve())
    if models_run_root is not None:
        print("Model run directory:", models_run_root.resolve())
    if dataset_name is not None:
        print("Dataset:", _safe_dataset_name(dataset_name))

    train_pairs = get_pairs(root, "train")
    val_pairs = get_pairs(root, "val")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    train_ds = make_tf_dataset(
        train_pairs,
        img_size,
        batch_size,
        training=True,
        shuffle_buffer_size=shuffle_buffer,
        drop_remainder=drop_remainder,
    )
    val_ds = make_tf_dataset(val_pairs, img_size, batch_size, training=False)
    val_ds = val_ds.repeat()

    steps_per_epoch = max(len(train_pairs) // batch_size, 1)
    validation_steps = max(len(val_pairs) // batch_size, 1)
    print(f"steps_per_epoch={steps_per_epoch} validation_steps={validation_steps}")

    backbone = model_cfg.get("backbone", "mobilenetv3small")
    head = model_cfg.get("head", "unet")
    pretrained = model_cfg.get("pretrained", "imagenet")

    initial_epoch = 0
    csv_append = False
    if resume_path is not None:
        print("Resuming from:", resume_path)
        initial_epoch = _read_resume_initial_epoch(resume_path)
        print(f"initial_epoch={initial_epoch} (target epochs={epochs})")
        meta_path = resume_path.with_name(resume_path.stem + ".meta.json")
        resume_meta: dict = {}
        if meta_path.is_file():
            resume_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_ds = resume_meta.get("dataset_name")
            if meta_ds is not None and meta_ds != dataset_name:
                print(
                    f"Warning: checkpoint dataset_name={meta_ds!r} "
                    f"!= current --dataset {dataset_name!r}"
                )
            if int(resume_meta.get("img_size", img_size)) != img_size:
                print(
                    "Warning: checkpoint img_size in meta differs from config "
                    f"({resume_meta.get('img_size')} vs {img_size})"
                )

        # Locate previous session folder + CSV for seeding
        prev_session_dir = _get_prev_session_dir(resume_path, resume_meta)
        prev_log_csv = _get_prev_log_csv(resume_meta, prev_session_dir, dataset_name)
        # finished_epoch_0idx: the 0-based epoch whose checkpoint we're resuming from
        finished_epoch_0idx = initial_epoch - 1
        csv_seeded = _seed_csv_from_prev(log_csv, prev_log_csv, finished_epoch_0idx)
        # CSVLogger will append rows; if we seeded the file it already has a header
        csv_append = True

        model = keras.models.load_model(
            str(resume_path),
            custom_objects=keras_custom_objects(),
            compile=True,
        )
        in_sh = model.input_shape
        if in_sh[1] is not None and int(in_sh[1]) != img_size:
            print(
                f"Warning: checkpoint input height {in_sh[1]} != config img_size {img_size}"
            )
        if in_sh[2] is not None and int(in_sh[2]) != img_size:
            print(
                f"Warning: checkpoint input width {in_sh[2]} != config img_size {img_size}"
            )
        if model.optimizer is None:
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=lr, weight_decay=weight_decay
                ),
                loss=combined_loss,
                metrics=["accuracy", _make_binary_iou()],
            )
    else:
        model = build_model(
            img_size,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=lr, weight_decay=weight_decay
            ),
            loss=combined_loss,
            metrics=["accuracy", _make_binary_iou()],
        )

    if initial_epoch >= epochs:
        print(
            f"Nothing to do: initial_epoch ({initial_epoch}) >= epochs ({epochs}). "
            "Increase training.epochs in config."
        )
        return

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val_binary_iou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_binary_iou",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_binary_iou",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(str(log_csv), append=csv_append),
        _EpochMetricsTerminalLogger(epochs),
    ]

    epoch_dir_raw = out_cfg.get("epoch_checkpoint_dir")
    if epoch_dir_raw:
        epoch_base = (
            models_run_root / Path(epoch_dir_raw).name
            if models_run_root is not None
            else Path(epoch_dir_raw)
        )
        epoch_dir = _with_dataset_suffix(epoch_base, dataset_name)
        callbacks.append(
            _PerEpochCheckpointWithMeta(
                epoch_dir,
                img_size=img_size,
                batch_size=batch_size,
                dataset_slug=_dataset_slug(dataset_name),
                epochs=epochs,
                config_path=config_path,
                dataset_name=dataset_name,
                learning_rate=lr,
                backbone=backbone,
                head=head,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                session_dir=models_run_root if models_run_root is not None else epoch_dir.parent,
                log_csv_path=log_csv,
            )
        )

    if debug_viz:
        # Always keep debug_viz inside the session folder so all artefacts
        # for a run are co-located; fall back to log_csv's parent when there
        # is no session folder (timestamp_run_dir=false).
        viz_dir = (models_run_root if models_run_root is not None else log_csv.parent) / "debug_viz"
        viz_every = int(out_cfg.get("debug_viz_every_n_epochs", 1))
        # On resume, copy the last completed epoch's images into the new session.
        if resume_path is not None:
            _copy_prev_debug_viz(prev_session_dir, initial_epoch, viz_dir)
        callbacks.append(
            _ValDebugVizCallback(
                val_ds,
                viz_dir,
                show_plot=sys.stdout.isatty(),
                every_n_epochs=viz_every,
                total_epochs=epochs,
            )
        )

    print(
        f"[train] Logging epoch metrics to CSV ({'append' if csv_append else 'new'}): "
        f"{log_csv.resolve()}",
        flush=True,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    model.save(str(final_path))
    print(f"Saved {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Named dataset under {dataset.path}/datasets/<name>/ "
            "(images/ and masks/ splits). Output checkpoints and log get this name suffix."
        ),
    )
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help=(
            "After each epoch (see output.debug_viz_every_n_epochs), save image / GT / "
            "prediction panels for three val samples under <log_csv_dir>/debug_viz/ "
            "as epoch_NN_val_sample_*.png (and show plots if stdout is a TTY)."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to a .keras checkpoint to continue training (overrides training.resume_from). "
            "Uses sibling .meta.json when present for initial_epoch and run metadata."
        ),
    )
    args = parser.parse_args()
    main(
        args.config,
        dataset_name=args.dataset,
        debug_viz=args.debug_viz,
        resume_from=args.resume,
    )
