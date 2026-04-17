from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

import train as train_mod


def load_segmentation_model(model_path: Path) -> keras.Model:
    objs = train_mod.keras_custom_objects()
    return keras.models.load_model(
        str(model_path),
        custom_objects=objs,
        compile=False,
    )


def run(
    model_path: Path,
    image_path: Path,
    *,
    img_size: int,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    model = load_segmentation_model(model_path)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size)) / 255.0
    batch = np.expand_dims(resized.astype(np.float32), 0)

    pred = model.predict(batch, verbose=0)[0]
    mask = (pred > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask.squeeze(), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = bgr.copy()
    overlay[mask > 0] = [0, 0, 255]

    cv2.imwrite(str(output_path), overlay)
    print(f"Saved {output_path}")


def _img_size_from_config(config_path: Path | None) -> int:
    if config_path is None or not config_path.is_file():
        return 224
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return int(cfg["dataset"]["img_size"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    img_size = args.img_size if args.img_size is not None else _img_size_from_config(args.config)
    run(
        args.model,
        args.image,
        img_size=img_size,
        output_path=args.output,
        threshold=args.threshold,
    )
