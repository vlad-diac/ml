# Sky segmentation (TensorFlow)

Binary sky-vs-not-sky segmentation trained on image and mask pairs. The model uses a **MobileNetV3-Large** encoder (ImageNet weights) and a **Lite ASPP head** (5 parallel branches: 1×1, dilated-6, dilated-12, dilated-18, global avg pool), with **0.5 × BCE + 0.5 × global batch Dice** loss (`combined_loss`) and **IoU** as the validation metric. Images are normalized to **`[-1, 1]`**. Inference binarizes logits with **`pred > threshold`** (default **0.5**) before resizing the mask to the original image size.

A **MobileNetV3-Small + U-Net decoder** path is still available for backward compatibility by setting `backbone: mobilenetv3small` / `head: unet` in [config.yaml](config.yaml).

## Prerequisites

- **Conda** with an environment named **`sky-seg`** (required by `scripts/setup.sh`).
- Python compatible with your TensorFlow wheel (see [requirements.txt](requirements.txt): TensorFlow 2.18.x on Python 3.9–3.12, or 2.20+ on Python 3.13+).
- **Apple Silicon (M1/M2/M3):** install `tensorflow-metal` after the base requirements to enable GPU acceleration via the Metal backend:
  ```bash
  pip install -r requirements.txt
  pip install -r requirements-macos-metal.txt
  ```

## Quick start

```bash
cd /path/to/this/project
conda activate sky-seg
chmod +x scripts/*.sh   # once, if needed
./scripts/setup.sh
./scripts/train.sh
./scripts/test.sh path/to/image.jpg
```

Training reads [config.yaml](config.yaml). With **`output.timestamp_run_dir: true`** (default), each run writes **best**, **final**, and **per-epoch** artifacts under a timestamped folder:

```text
models/DD-MM-YYYY-HH-MM-SS/best_model.keras
models/DD-MM-YYYY-HH-MM-SS/final_model.keras
models/DD-MM-YYYY-HH-MM-SS/epoch_ckpt/   # if epoch_checkpoint_dir is enabled
```

Basenames come from `output.best_model_path`, `output.model_path`, and `output.epoch_checkpoint_dir`. Set **`timestamp_run_dir: false`** to keep the legacy flat layout (`models/best_model.keras`, etc.). Metrics stay in **`logs/training_log.csv`** (not under the stamp). If you pass **`--dataset <name>`**, `.keras` filenames and the epoch directory name get the usual dataset suffix.

**Debug visuals:** pass **`--debug-viz`** to save image / ground-truth / prediction panels for three validation samples **after each epoch** (or every **`output.debug_viz_every_n_epochs`** epochs), under `logs/debug_viz/` as **`epoch_NN_val_sample_*.png`** (1-based `NN`). With `--dataset`, that folder sits beside `logs/training_log_<name>.csv`. If stdout is a TTY, matplotlib may also show figures.

**Resume training:** set **`training.resume_from`** in [config.yaml](config.yaml) to a `.keras` file, or pass **`--resume path/to.ckpt.keras`**. Training appends to the same CSV log when resuming. **New runs** still create a **new** `models/<stamp>/` for outputs; the resume path can point at an older stamp. Prefer checkpoints with sibling **`.meta.json`** (`initial_epoch`, `img_size`, `dataset_name`, `backbone`, `head`, etc.). Without `.meta.json`, `initial_epoch` is inferred from filenames like `e05_...` (next epoch = 6), else it defaults to 0 with a warning.

**Per-epoch checkpoints:** when **`output.epoch_checkpoint_dir`** is set, each epoch writes `e{epoch}_sz{img_size}_bs{batch}_ds{slug}.keras` and **`.meta.json`** inside the run's epoch directory (under the timestamp folder when `timestamp_run_dir` is true). Set **`epoch_checkpoint_dir`** to **`null`** to disable.

**TFLite export:** from the project root, run `python src/export_tflite.py --config config.yaml --model path/to/best_model.keras`. Creates **`models/out/tflite/<timestamp>/`** containing the **`.tflite`** file, **`metadata.yaml`** (runtime, quantization, precision, tool versions, I/O shapes), and **`labels.txt`** (not_sky / sky).

| Mode | Command |
|------|---------|
| FP16 (default) | `python src/export_tflite.py --model best_model.keras` |
| int8 PTQ | `python src/export_tflite.py --model best_model.keras --quantization int8 --representative-data data/images/train` |

Additional export flags: **`--output-dir`**, **`--run-name`**, **`--tflite-name`**, **`--num-calibration-samples`** (default 200), **`--img-size`** (default: inferred from model), **`--qairt-version`** (metadata field).

Inference writes **`output.png`** in the current working directory by default (red overlay on predicted sky). Override with `python src/inference.py --help`.

## Project layout

| Path | Role |
|------|------|
| [config.yaml](config.yaml) | Dataset paths, image size, batch size, training hyperparameters, model options, output paths |
| [requirements.txt](requirements.txt) | Python dependencies (TensorFlow pins differ by Python version) |
| [requirements-macos-metal.txt](requirements-macos-metal.txt) | Optional **Apple Silicon** GPU support: install after base requirements |
| `data/` | Dataset root (see below) |
| `models/` | Saved `.keras` models; `models/pretrained/` reserved for any manual weights (ImageNet loads automatically via Keras) |
| `logs/` | Training CSV logs; optional `debug_viz/` PNGs when using `--debug-viz` |
| `src/` | Training, data loading, model, inference, and **TFLite export** |
| `scripts/` | Shell runners and data-prep utilities |

### Source modules

- [src/dataset.py](src/dataset.py) — `get_pairs()`, `make_tf_dataset()`: load images normalized to **`[-1, 1]`** (ImageNet-compatible), `ignore_errors`, **train-only** shuffle (buffer `min(1024, num_pairs)`), augmentation (**horizontal flip**, **brightness/contrast jitter**, **rotation ±15°**), **`repeat()`**, **foreground filter** (`sum(mask) > 10`), batch, prefetch.
- [src/model.py](src/model.py) — `build_model(img_size, backbone, head, pretrained, freeze_backbone)`. **Default:** `backbone="mobilenetv3large"` + `head="aspp"` (Lite ASPP, fine-tunes from layer 100+). Legacy: `backbone="mobilenetv3small"` + `head="unet"` (skip-connection decoder, full backbone freeze).
- [src/train.py](src/train.py) — **0.5 × BCE + 0.5 × global Dice** (`combined_loss`), **IoU** metric, **`steps_per_epoch`** / **`validation_steps`**, **`val_ds.repeat()`**, optional **`models/<DD-MM-YYYY-HH-MM-SS>/`** run dirs, best + final + per-epoch checkpoints (`.meta.json` stores `backbone`, `head`, and all hyperparameters), **`--resume`**, **`--debug-viz`**.
- [src/inference.py](src/inference.py) — Loads a saved model, converts **BGR→RGB** for consistency with training, runs prediction, saves an overlay.
- [src/export_tflite.py](src/export_tflite.py) — Loads `.keras` with **`keras_custom_objects`**, exports **FP16** or **int8 PTQ** TFLite under **`models/out/tflite/<run>/`** plus **`metadata.yaml`** and **`labels.txt`**. int8 calibration uses the same **`[-1, 1]`** normalization as training.

### Training data pipeline (summary)

| Split | Dataset behavior |
|--------|------------------|
| **Train** | Shuffle → augment (flip + brightness/contrast + rotation ±15°) → **`repeat()`** (infinite iterator) → drop masks with foreground sum ≤ 10 → batch |
| **Val** | Same foreground filter → **`repeat()`** (iterator cycles) → batch |

`model.fit` uses `steps_per_epoch = max(len(train_pairs) // batch_size, 1)` and `validation_steps = max(len(val_pairs) // batch_size, 1)`. Validation is **`repeat()`** so Keras can always draw exactly `validation_steps` batches even when **`has_foreground`** drops samples (avoids *Input ran out of data*). If you have very small objects, raise the filter threshold in `make_tf_dataset()` or remove the filter.

### Dataset layout

**Default** — under `dataset.path` in [config.yaml](config.yaml) (default `data/`):

```text
data/
  images/
    train/   val/   test/
  masks/
    train/   val/   test/
```

**Named dataset** — train with `--dataset <name>` so the root is `{dataset.path}/datasets/<name>/` with the same `images/` and `masks/` structure:

```text
data/
  datasets/
    my_run/
      images/
        train/   val/   test/
      masks/
        train/   val/   test/
```

For each split, every training image must have a matching mask PNG with the **same basename** (e.g. `foo.jpg` → `masks/.../foo.png`). Supported image extensions: `.jpg`, `.jpeg`, `.png`.

To build SkyFinder data directly into a named folder, use `scripts/build_skyfinder_dataset.py --data-root data/datasets/my_run` (plus your usual `--source` / `--clean` flags), then train with `./scripts/train.sh --dataset my_run`.

## Scripts

### `scripts/setup.sh`

- Ensures the active conda environment is **`sky-seg`**.
- Upgrades `pip` and runs `pip install -r requirements.txt`.
- Creates `models/pretrained` and `logs` if missing.
- Reminds you that **Apple Silicon** users can optionally run `pip install -r requirements-macos-metal.txt`.

### `scripts/train.sh`

Changes to the project root and runs `python src/train.py --config config.yaml`, forwarding any extra arguments (for example **`--dataset`**, **`--debug-viz`**, or **`--resume`**):

```bash
./scripts/train.sh
./scripts/train.sh --dataset my_run
./scripts/train.sh --dataset my_run --debug-viz
./scripts/train.sh --dataset my_run --resume models/08-04-2026-12-00-00/epoch_ckpt_my_run/e05_sz320_bs16_dsmy_run.keras
```

Equivalent:

```bash
python src/train.py --config config.yaml --dataset my_run
python src/train.py --config config.yaml --dataset my_run --debug-viz
python src/train.py --config config.yaml --dataset my_run --resume models/08-04-2026-12-00-00/epoch_ckpt_my_run/e05_sz320_bs16_dsmy_run.keras
```

### `scripts/test.sh`

Runs inference on one image. Usage:

```bash
./scripts/test.sh path/to/image.jpg
```

Uses `models/best_model.keras` by default. After training, best weights usually live under a **timestamped** folder, e.g. `models/08-04-2026-14-30-00/best_model.keras` (or `best_model_my_run.keras` with **`--dataset`**):

```bash
MODEL=models/08-04-2026-14-30-00/best_model_my_run.keras ./scripts/test.sh path/to/image.jpg
```

Other checkpoints:

```bash
MODEL=models/08-04-2026-14-30-00/final_model.keras ./scripts/test.sh path/to/image.jpg
```

The script passes `--config config.yaml` so input size matches training.

### `scripts/build_skyfinder_dataset.py`

Builds `images/{train,val,test}` and `masks/{...}` under `--data-root` (default `data/`) from SkyFinder-style zips under `data/unformatted/skyfinder/`. Use `--data-root data/datasets/<name>` to match `train.py --dataset <name>`. Run from the project root, for example:

```bash
python scripts/build_skyfinder_dataset.py --help
```

### `scripts/download_zenodo_dataset.py`

Downloads the Zenodo archive used for SkyFinder-style raw data. See `--help` for URLs and paths.

### `scripts/verify_dataset.py`

Prints per-split image/mask counts, checks stem alignment, and samples dimensions and mask value ranges. Run from the project root:

```bash
python scripts/verify_dataset.py
```

## Configuration reference ([config.yaml](config.yaml))

- **`dataset`**: `path`, `img_size` (default **320**), `batch_size` (default **16**)
- **`training`**: `epochs` (default **20**), `learning_rate` (default **1e-4**), `freeze_backbone` (default **false** — fine-tunes from layer 100+ for ASPP/Large), `resume_from` (optional path to `.keras`; **`--resume` CLI flag overrides**)
- **`model`**: `backbone` (**`mobilenetv3large`** or `mobilenetv3small`), `head` (**`aspp`** with Large; `unet` with Small), `pretrained` (`imagenet`, or `none` / `null` / `random` for no ImageNet weights)
- **`output`**: `model_path`, `best_model_path`, `log_csv`, `epoch_checkpoint_dir` (per-epoch `.keras` + `.meta.json`, or **`null`** to disable), **`timestamp_run_dir`** (nest model paths under `models/DD-MM-YYYY-HH-MM-SS/`), **`debug_viz_every_n_epochs`** (used with **`--debug-viz`**)

## Other files

- [train.py](train.py) (repo root) — Deprecated entrypoint; points you to `scripts/train.sh` / `src/train.py`.
- [verify_gpu.py](verify_gpu.py) — Quick check that TensorFlow sees expected devices (run with your env activated).

## Pretrained weights

No separate manual download is required for **ImageNet** initialization: `keras.applications.MobileNetV3Large(weights="imagenet")` downloads and caches weights automatically on first use.
