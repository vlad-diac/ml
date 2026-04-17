from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("data")
SPLITS = ["train", "val", "test"]

def check_split(split: str) -> None:
    image_dir = ROOT / "images" / split
    mask_dir = ROOT / "masks" / split

    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"])

    print(f"\n--- {split.upper()} ---")
    print(f"Images: {len(image_files)}")
    print(f"Masks : {len(mask_files)}")

    image_stems = {p.stem for p in image_files}
    mask_stems = {p.stem for p in mask_files}

    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    if missing_masks:
        print(f"Missing masks for {len(missing_masks)} images")
        print(missing_masks[:10])

    if missing_images:
        print(f"Missing images for {len(missing_images)} masks")
        print(missing_images[:10])

    if not image_files or not mask_files:
        print("Split is empty")
        return

    sample_image = image_files[0]
    sample_mask = mask_dir / f"{sample_image.stem}.png"

    img = Image.open(sample_image)
    mask = Image.open(sample_mask)

    print("Sample image size:", img.size, "mode:", img.mode)
    print("Sample mask size :", mask.size, "mode:", mask.mode)

    mask_np = np.array(mask)
    unique_values = np.unique(mask_np)
    print("Mask unique values:", unique_values[:20], f"(count={len(unique_values)})")

for split in SPLITS:
    check_split(split)