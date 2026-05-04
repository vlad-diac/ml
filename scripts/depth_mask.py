import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Distinct BGR colors for cluster visualization, ordered near-to-far (warm → cool)
_CLUSTER_COLORS = [
    (0, 0, 255),    # red   – closest
    (0, 128, 255),  # orange
    (0, 255, 255),  # yellow
    (0, 255, 128),  # green
    (255, 128, 0),  # blue  – furthest
]


def cluster_depth(depth_uint8: np.ndarray, n_clusters: int = 5) -> tuple:
    """
    K-means cluster depth values (1-D), sort by median depth ascending (darkest = furthest first).
    Returns (labels_2d, stats_sorted, furthest_stat).
    Each stat: {id, median, std, coverage, separation}.
    """
    flat = depth_uint8.flatten().reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat)
    labels_2d = labels.reshape(depth_uint8.shape)

    stats = []
    for i in range(n_clusters):
        pixel_vals = flat[labels == i, 0]
        stats.append({
            "id": i,
            "median": float(np.median(pixel_vals)),
            "std": float(np.std(pixel_vals)),
            "coverage": float(len(pixel_vals) / len(flat)),
        })

    # Ascending median → index 0 is furthest (darkest)
    stats.sort(key=lambda s: s["median"])

    # Separation: normalised gap between furthest and next cluster median
    separation = (stats[1]["median"] - stats[0]["median"]) / 255.0 if len(stats) > 1 else 1.0
    stats[0]["separation"] = round(separation, 4)

    return labels_2d, stats, stats[0]


def generate_depth_mask(
    image_path: str,
    contour_mask: bool = False,
    contour_image: bool = False,
    n_clusters: int = 5,
) -> None:
    image = Image.open(image_path).convert("RGB")

    processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    model.eval()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth_uint8 = (depth.detach().cpu().numpy() * 255).astype(np.uint8)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- depth mask ---
    Image.fromarray(depth_uint8).save(os.path.join(output_dir, f"{base_name}_{timestamp}.png"))
    print(f"Depth mask        → {output_dir}/{base_name}_{timestamp}.png")

    # --- clustering ---
    labels_2d, stats, furthest = cluster_depth(depth_uint8, n_clusters)

    print(f"\nClusters (furthest → closest):")
    for rank, s in enumerate(stats):
        tag = " ← furthest" if rank == 0 else ""
        print(
            f"  [{rank}] median={s['median']:6.1f}  std={s['std']:5.1f}"
            f"  coverage={s['coverage']*100:5.1f}%{tag}"
        )
    print(f"\nFurthest cluster separation score: {furthest['separation']:.4f}  (0–1, higher = more distinct)")

    # --- furthest region binary mask ---
    furthest_binary = (labels_2d == furthest["id"]).astype(np.uint8) * 255
    furthest_path = os.path.join(output_dir, f"{base_name}_{timestamp}_furthest.png")
    cv2.imwrite(furthest_path, furthest_binary)
    print(f"Furthest region   → {furthest_path}")

    # --- cluster colour map (near = warm, far = cool, mapped by rank) ---
    colors = (_CLUSTER_COLORS * ((n_clusters // len(_CLUSTER_COLORS)) + 1))[:n_clusters]
    cluster_vis = np.zeros((*depth_uint8.shape, 3), dtype=np.uint8)
    for rank, s in enumerate(stats):
        cluster_vis[labels_2d == s["id"]] = colors[rank]
    clusters_path = os.path.join(output_dir, f"{base_name}_{timestamp}_clusters.png")
    cv2.imwrite(clusters_path, cluster_vis)
    print(f"Cluster map       → {clusters_path}")

    # --- optional contours (drawn around the furthest cluster) ---
    if contour_mask or contour_image:
        contours, _ = cv2.findContours(furthest_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contour_mask:
            overlay = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
            path = os.path.join(output_dir, f"{base_name}_{timestamp}_contour_mask.png")
            cv2.imwrite(path, overlay)
            print(f"Contour mask      → {path}")

        if contour_image:
            overlay = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
            path = os.path.join(output_dir, f"{base_name}_{timestamp}_contour_image.png")
            cv2.imwrite(path, overlay)
            print(f"Contour image     → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth mask + furthest-region clustering via Depth Anything.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--contour-mask", action="store_true", help="Save furthest-cluster contour on the depth mask.")
    parser.add_argument("--contour-image", action="store_true", help="Save furthest-cluster contour on the original image.")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of K-means depth clusters (default: 5).",
    )
    args = parser.parse_args()

    generate_depth_mask(
        args.image_path,
        contour_mask=args.contour_mask,
        contour_image=args.contour_image,
        n_clusters=args.n_clusters,
    )
