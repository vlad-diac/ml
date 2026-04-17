from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
_IMAGE_EXT = {".jpg", ".jpeg", ".png"}

# Module-level layer so the RandomRotation RNG state is shared across calls.
# factor=15/360 maps to ±15 degrees (Keras factor is a fraction of 2π).
_rotator = tf.keras.layers.RandomRotation(factor=15 / 360, fill_mode="reflect")


def _path_to_str(path_tensor) -> str:
    """Normalize tf.string / numpy path to a filesystem path."""
    x = path_tensor
    if isinstance(x, bytes):
        return x.decode("utf-8")
    x = np.asarray(x).item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _decode_pair_py(image_path, mask_path, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load image + mask with Pillow (tolerates truncated JPEGs better than tf.decode_jpeg)."""
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ip = _path_to_str(image_path)
    mp = _path_to_str(mask_path)

    with Image.open(ip) as im:
        im = im.convert("RGB")
        im = im.resize((img_size, img_size), Image.BILINEAR)
        img = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0  # [-1, 1]

    with Image.open(mp) as mm:
        mm = mm.convert("L")
        mm = mm.resize((img_size, img_size), Image.NEAREST)
        m = np.asarray(mm, dtype=np.float32)
    m = (m > 127.0).astype(np.float32)[..., np.newaxis]
    return img, m


def get_pairs(root: str | Path, split: str) -> list[tuple[str, str]]:
    root = Path(root)
    img_dir = root / "images" / split
    mask_dir = root / "masks" / split

    image_paths = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT
    )
    pairs: list[tuple[str, str]] = []
    for img in image_paths:
        mask = mask_dir / f"{img.stem}.png"
        if mask.exists():
            pairs.append((str(img), str(mask)))

    if not pairs:
        raise ValueError(f"No image/mask pairs found in {img_dir} and {mask_dir}")

    return pairs


def make_tf_dataset(
    pairs: list[tuple[str, str]],
    img_size: int,
    batch_size: int,
    *,
    training: bool = False,
    shuffle_buffer_size: int = 1024,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    def load_image_mask(image_path, mask_path):
        def _np_load(ip, mp):
            return _decode_pair_py(ip, mp, img_size)

        image, mask = tf.numpy_function(
            _np_load,
            [image_path, mask_path],
            (tf.float32, tf.float32),
        )
        image.set_shape([img_size, img_size, 3])
        mask.set_shape([img_size, img_size, 1])
        return image, mask

    def _coarse_erase(image):
        """Zero out a random rectangle (5–20% of side length) on the image."""
        h = tf.cast(tf.random.uniform((), 0.05, 0.20) * img_size, tf.int32)
        w = tf.cast(tf.random.uniform((), 0.05, 0.20) * img_size, tf.int32)
        y0 = tf.random.uniform((), 0, img_size - h, dtype=tf.int32)
        x0 = tf.random.uniform((), 0, img_size - w, dtype=tf.int32)
        rows = tf.range(img_size)
        cols = tf.range(img_size)
        row_mask = tf.logical_or(rows < y0, rows >= y0 + h)
        col_mask = tf.logical_or(cols < x0, cols >= x0 + w)
        keep = tf.cast(
            row_mask[:, tf.newaxis] | col_mask[tf.newaxis, :],
            tf.float32,
        )[..., tf.newaxis]  # H×W×1
        return image * keep

    def augment(image, mask):
        # ---- Geometric transforms (applied to image AND mask) ----

        # Horizontal flip — use tf.cond so the condition lives in the TF graph
        hflip = tf.random.uniform(()) > 0.5
        image = tf.cond(hflip, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(hflip, lambda: tf.image.flip_left_right(mask), lambda: mask)

        # Vertical flip (low probability; sky is rarely inverted, but aids generalisation)
        vflip = tf.random.uniform(()) > 0.8
        image = tf.cond(vflip, lambda: tf.image.flip_up_down(image), lambda: image)
        mask = tf.cond(vflip, lambda: tf.image.flip_up_down(mask), lambda: mask)

        # Rotation ±15°: stack image+mask on channel axis so both get the
        # exact same geometric transform, then split back.
        combined = tf.concat([image, mask], axis=-1)  # H×W×4
        combined = _rotator(combined[tf.newaxis], training=True)[0]
        image = combined[..., :3]
        mask = combined[..., 3:4]
        mask = tf.cast(mask > 0.5, tf.float32)

        # Coarse dropout — erase a random rectangle on the image only (50% chance)
        erase = tf.random.uniform(()) > 0.5
        image = tf.cond(erase, lambda: _coarse_erase(image), lambda: image)

        # ---- Photometric transforms (image only) ----

        # Stronger brightness / contrast (was ±0.1 / 0.9–1.1)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Subtle hue jitter for sky-colour variation
        image = tf.image.random_hue(image, max_delta=0.02)

        # Random gamma in [0.7, 1.4] — must be applied in [0, 1] space
        gamma = tf.random.uniform((), 0.7, 1.4)
        image_01 = (image + 1.0) / 2.0
        image_01 = tf.pow(tf.clip_by_value(image_01, 1e-8, 1.0), gamma)
        image = image_01 * 2.0 - 1.0

        # Low-level Gaussian noise
        noise = tf.random.normal(tf.shape(image), stddev=0.03)
        image = image + noise

        # Final clip to keep the [-1, 1] range intact
        image = tf.clip_by_value(image, -1.0, 1.0)
        return image, mask

    def has_foreground(_image, mask):
        return tf.reduce_sum(mask) > 10.0

    effective_buffer = min(shuffle_buffer_size, max(len(pairs), 1))

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(load_image_mask, num_parallel_calls=AUTOTUNE)
    # Skip pairs that still fail (e.g. zero-byte files, non-images).
    ds = ds.ignore_errors()
    if training:
        ds = ds.shuffle(buffer_size=effective_buffer)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.repeat()
    ds = ds.filter(has_foreground)
    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
