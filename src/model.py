from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _application_weights(pretrained: str | None) -> str | None:
    if pretrained is None:
        return None
    p = str(pretrained).strip().lower()
    if p in ("", "none", "null", "random"):
        return None
    if p == "imagenet":
        return "imagenet"
    raise ValueError(f"Unsupported pretrained value: {pretrained!r}")


def _last_output_per_square_spatial_size(backbone: keras.Model) -> dict[int, object]:
    """Map H (==W) -> that layer's output; later layers overwrite earlier at the same size."""
    out: dict[int, object] = {}
    for lay in backbone.layers:
        sh = lay.output.shape
        if len(sh) != 4 or sh[1] is None or sh[2] is None:
            continue
        h, w = int(sh[1]), int(sh[2])
        if h != w:
            continue
        out[h] = lay.output
    return out


def _mobilenet_skips_and_bottleneck(backbone: keras.Model, img_size: int) -> list[object]:
    """Pick encoder outputs by spatial resolution (robust to Keras layer renames)."""
    spatial = _last_output_per_square_spatial_size(backbone)
    sizes = [
        img_size // 2,
        img_size // 4,
        img_size // 8,
        img_size // 16,
        img_size // 32,
    ]
    missing = [s for s in sizes if s not in spatial]
    if missing:
        raise ValueError(
            "MobileNetV3Small missing expected spatial levels "
            f"{missing} for img_size={img_size}. "
            f"Found sizes: {sorted(spatial.keys())}"
        )
    return [spatial[s] for s in sizes]


# ---------------------------------------------------------------------------
# U-Net decoder (original path — MobileNetV3Small)
# ---------------------------------------------------------------------------

def _build_unet_model(
    img_size: int,
    *,
    pretrained: str | None,
    freeze_backbone: bool,
) -> keras.Model:
    input_shape = (img_size, img_size, 3)
    weights = _application_weights(pretrained)

    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = not freeze_backbone

    outs = _mobilenet_skips_and_bottleneck(base_model, img_size)
    encoder = keras.Model(inputs=base_model.input, outputs=outs)
    encoder.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape)
    s1, s2, s3, s4, x = encoder(inputs)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s4])
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s3])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s2])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s1])
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# ASPP head (research spec — MobileNetV3Large)
# ---------------------------------------------------------------------------

def _aspp_branch(x: object, filters: int, kernel: int, dilation: int) -> object:
    """Single ASPP conv branch."""
    padding = "same" if kernel > 1 else "valid"
    return layers.Conv2D(
        filters, kernel, padding=padding, dilation_rate=dilation,
        use_bias=False, activation="relu",
    )(x)


def _build_aspp_model(
    img_size: int,
    *,
    pretrained: str | None,
    freeze_backbone: bool,
) -> keras.Model:
    """MobileNetV3Large backbone + Lite ASPP head (research spec).

    ASPP branches (applied in parallel to backbone output):
      - 1×1 conv
      - 3×3 dilated conv, rate=6
      - 3×3 dilated conv, rate=12
      - 3×3 dilated conv, rate=18
      - Global average pooling branch

    All branches concatenated → Conv2D(256,1) projection →
    bilinear upsample to (img_size, img_size) → Conv2D(1,1,sigmoid).

    freeze_backbone=False  → fine-tune from layer 100+ (research spec)
    freeze_backbone=True   → freeze entire backbone
    """
    input_shape = (img_size, img_size, 3)
    weights = _application_weights(pretrained)

    base_model = keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    if freeze_backbone:
        base_model.trainable = False
    else:
        # Fine-tune: freeze first 100 layers, unfreeze the rest
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    features = base_model(inputs, training=False)

    # ASPP branches — parallel
    branch_1x1 = _aspp_branch(features, 256, 1, 1)
    branch_d6 = _aspp_branch(features, 256, 3, 6)
    branch_d12 = _aspp_branch(features, 256, 3, 12)
    branch_d18 = _aspp_branch(features, 256, 3, 18)

    # Global average pooling branch — resize back to feature map spatial dims.
    # features.shape[1] / [2] are static Python ints (known at graph-build time
    # because input_shape is fully specified), so Resizing is valid here.
    feat_h = int(features.shape[1])
    feat_w = int(features.shape[2])
    gap = layers.GlobalAveragePooling2D()(features)
    gap = layers.Reshape((1, 1, gap.shape[-1]))(gap)
    gap = layers.Conv2D(256, 1, use_bias=False, activation="relu")(gap)
    gap = layers.Resizing(feat_h, feat_w, interpolation="bilinear")(gap)

    # Concatenate all branches
    x = layers.Concatenate()([branch_1x1, branch_d6, branch_d12, branch_d18, gap])

    # Project to 256 channels
    x = layers.Conv2D(256, 1, use_bias=False, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Final 1×1 conv to single-channel logit
    x = layers.Conv2D(1, 1, activation="sigmoid")(x)

    # Bilinear upsample to full input resolution using a Keras layer (tf.image.resize
    # cannot be called on KerasTensors during functional model construction).
    outputs = layers.Resizing(img_size, img_size, interpolation="bilinear")(x)

    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_model(
    img_size: int,
    *,
    backbone: str = "mobilenetv3small",
    pretrained: str | None = "imagenet",
    freeze_backbone: bool = True,
    head: str = "unet",
) -> keras.Model:
    """Build a segmentation model.

    Parameters
    ----------
    img_size:
        Square input/output resolution (e.g. 320).
    backbone:
        "mobilenetv3small" (U-Net decoder) or "mobilenetv3large" (ASPP).
    pretrained:
        "imagenet" or None / "random".
    freeze_backbone:
        True  → freeze entire backbone.
        False → fine-tune (for ASPP/Large: freeze first 100 layers only).
    head:
        "aspp"  → Lite ASPP head; requires backbone="mobilenetv3large".
        "unet"  → U-Net skip-connection decoder; requires backbone="mobilenetv3small".
    """
    b = backbone.lower().strip()
    h = head.lower().strip()

    if h == "aspp":
        if b != "mobilenetv3large":
            raise ValueError(
                f"head='aspp' requires backbone='mobilenetv3large', got {backbone!r}"
            )
        return _build_aspp_model(img_size, pretrained=pretrained, freeze_backbone=freeze_backbone)

    if h == "unet":
        if b != "mobilenetv3small":
            raise ValueError(
                f"head='unet' requires backbone='mobilenetv3small', got {backbone!r}"
            )
        return _build_unet_model(img_size, pretrained=pretrained, freeze_backbone=freeze_backbone)

    raise ValueError(f"Unsupported head: {head!r}. Choose 'aspp' or 'unet'.")
