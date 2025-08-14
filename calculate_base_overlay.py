from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path

MIN_REGION_AREA = 1000
EDGE_MARGIN = 10
PINK_LOWER = (180, 0, 100)
PINK_UPPER = (255, 100, 255)


def mask_pink_regions(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    mask = (
        (arr[:, :, 0] >= PINK_LOWER[0])
        & (arr[:, :, 0] <= PINK_UPPER[0])
        & (arr[:, :, 1] >= PINK_LOWER[1])
        & (arr[:, :, 1] <= PINK_UPPER[1])
        & (arr[:, :, 2] >= PINK_LOWER[2])
        & (arr[:, :, 2] <= PINK_UPPER[2])
    ).astype(np.uint8)
    return mask, arr


def filter_and_draw_contours(
    mask, orig_arr, min_area=MIN_REGION_AREA, contour_color=(0, 255, 0)
):
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    mask_filtered = np.zeros_like(mask_uint8)
    contour_img = orig_arr.copy()
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask_filtered[labels == label] = 255
            region_mask = np.zeros_like(mask_uint8)
            region_mask[labels == label] = 255
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(contour_img, contours, -1, contour_color, thickness=2)
    return mask_filtered, contour_img


def is_mask_fully_visible(mask, edge_margin=EDGE_MARGIN):
    h, w = mask.shape
    top = np.any(mask[:edge_margin, :] > 0)
    bottom = np.any(mask[-edge_margin:, :] > 0)
    left = np.any(mask[:, :edge_margin] > 0)
    right = np.any(mask[:, -edge_margin:] > 0)
    return not (top or bottom or left or right)


def mask_solidity(mask):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = np.sum(mask)
    hull = cv2.convexHull(np.vstack(contours))
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def mask_aspect_ratio(mask):
    mask = (mask > 0).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(coords)
    if h == 0:
        return 0.0
    return w / h


def mask_extent(mask):
    mask = (mask > 0).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(coords)
    area = np.sum(mask)
    rect_area = w * h
    if rect_area == 0:
        return 0.0
    return area / rect_area


def mask_quality_score(mask, orig_arr):
    area = np.sum(mask > 0)
    area_score = min(area / (orig_arr.shape[0] * orig_arr.shape[1]), 1.0)
    border_penalty = 1.0 if is_mask_fully_visible(mask) else 0.5
    solidity = mask_solidity(mask)
    solidity_score = min(solidity, 1.0)
    ar = mask_aspect_ratio(mask)
    ar_score = max(1.0 - abs(ar - 0.5), 0.0)
    extent = mask_extent(mask)
    extent_score = min(extent, 1.0)
    final = (
        3 * area_score
        + 2 * border_penalty
        + 2 * solidity_score
        + 1 * ar_score
        + 1 * extent_score
    )
    return final, {
        "area": int(area),
        "area_score": float(area_score),
        "border_penalty": float(border_penalty),
        "solidity": float(solidity),
        "solidity_score": float(solidity_score),
        "aspect_ratio": float(ar),
        "ar_score": float(ar_score),
        "extent": float(extent),
        "extent_score": float(extent_score),
        "final_score": float(final),
    }


def overlay_on_original(original_path, mask, overlay_color=(0, 255, 255), alpha=0.5):
    """Overlay the chosen base mask on the original image. (default cyan for base in Stage 3)"""
    original = Image.open(original_path).convert("RGB")
    arr = np.array(original).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    if mask_img.size != (arr.shape[1], arr.shape[0]):
        mask_img = mask_img.resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
    mask_arr = np.array(mask_img)
    overlay = np.zeros_like(arr)
    overlay[:, :, :] = overlay_color
    mask_bool = mask_arr > 0
    arr[mask_bool] = (alpha * overlay[mask_bool] + (1 - alpha) * arr[mask_bool]).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(
        mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(arr, contours, -1, (0, 255, 0), thickness=2)
    return Image.fromarray(arr)


def process_base_folder(
    base_folder_path,
    input_images_paths,
    out_dir="combined_base_output",
    min_area=MIN_REGION_AREA,
):
    masks = []
    orig_arrays = []
    fnames = []
    contour_imgs = []
    valid_mask_indices = []
    scores = []
    detailed_scores = []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(
        [
            f
            for f in os.listdir(base_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    img_files = img_files[:2] 

    for i, f in enumerate(img_files):
        mask, arr = mask_pink_regions(os.path.join(base_folder_path, f))
        mask_filtered, contour_img = filter_and_draw_contours(mask, arr, min_area)
        if np.sum(mask_filtered > 0) > min_area:
            masks.append(mask_filtered)
            orig_arrays.append(arr)
            fnames.append(f)
            contour_img_pil = Image.fromarray(contour_img)
            contour_imgs.append(contour_img_pil)
            contour_img_pil.save(os.path.join(out_dir, f"{Path(f).stem}_contour.png"))
            valid_mask_indices.append(i)
            s, ds = mask_quality_score(mask_filtered, arr)
            scores.append(s)
            detailed_scores.append(ds)

    if len(masks) == 0:
        raise RuntimeError(
            "No valid base mask images found (all below area threshold)."
        )

    idx_best = int(np.argmax(scores))
    best_score_dict = detailed_scores[idx_best]
    combined_mask = masks[idx_best]
    valid_mask_indices = [valid_mask_indices[idx_best]]

    Image.fromarray(combined_mask).save(os.path.join(out_dir, "combined_mask.png"))
    combined_outputs = []
    for idx in valid_mask_indices:
        orig_img_path = input_images_paths[idx]
        out_img = overlay_on_original(orig_img_path, combined_mask)
        out_fname = f"{Path(orig_img_path).stem}_combinedbase.png"
        out_img.save(os.path.join(out_dir, out_fname))
        combined_outputs.append((out_fname, out_img))

    orig_mask_files = [os.path.join(base_folder_path, f) for f in fnames]
    contour_files = [
        os.path.join(out_dir, f"{Path(f).stem}_contour.png") for f in fnames
    ]

    return {
        "mask_files": orig_mask_files,
        "contour_images": contour_files,
        "combined_mask": os.path.join(out_dir, "combined_mask.png"),
        "combined_overlays": [
            os.path.join(out_dir, fname) for fname, _ in combined_outputs
        ],
        "detailed_scores": detailed_scores,
        "idx_best": idx_best,
        "best_score_dict": best_score_dict,
        "best_mask_file": orig_mask_files[idx_best],
        "best_mask": combined_mask,
        "best_input_idx": valid_mask_indices[0],
    }
