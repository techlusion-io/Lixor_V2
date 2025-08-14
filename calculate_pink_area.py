from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
from pathlib import Path

MIN_REGION_AREA = 1000

PINK_LOWER = (180, 0, 100)
PINK_UPPER = (255, 100, 255)

def calculate_pink_area(image_path, pink_lower=PINK_LOWER, pink_upper=PINK_UPPER):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    mask = (
        (arr[:, :, 0] >= pink_lower[0])
        & (arr[:, :, 0] <= pink_upper[0])
        & (arr[:, :, 1] >= pink_lower[1])
        & (arr[:, :, 1] <= pink_upper[1])
        & (arr[:, :, 2] >= pink_lower[2])
        & (arr[:, :, 2] <= pink_upper[2])
    )     
    pink_pixel_count = np.sum(mask)
    total_pixel_count = arr.shape[0] * arr.shape[1]
    percent_pink = (
        (pink_pixel_count / total_pixel_count) * 100 if total_pixel_count else 0.0
    )
    return pink_pixel_count, total_pixel_count, percent_pink


def mask_pink_regions(img_path, pink_lower=PINK_LOWER, pink_upper=PINK_UPPER):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    mask = (
        (arr[:, :, 0] >= pink_lower[0])
        & (arr[:, :, 0] <= pink_upper[0])
        & (arr[:, :, 1] >= pink_lower[1])
        & (arr[:, :, 1] <= pink_upper[1])
        & (arr[:, :, 2] >= pink_lower[2])
        & (arr[:, :, 2] <= pink_upper[2])
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


def combine_masks_AND(mask1, mask2):
    return cv2.bitwise_and(mask1, mask2)


def overlay_on_original(
    original_path, combined_mask, overlay_color=(255, 255, 0), alpha=0.6
):
    original = Image.open(original_path).convert("RGB")
    arr = np.array(original).astype(np.uint8)
    # Resize mask if needed
    mask_img = Image.fromarray(combined_mask)
    if mask_img.size != (arr.shape[1], arr.shape[0]):
        mask_img = mask_img.resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
    mask_arr = np.array(mask_img)
    overlay = np.zeros_like(arr)
    overlay[:, :, :] = overlay_color
    mask_bool = mask_arr > 0
    arr[mask_bool] = (alpha * overlay[mask_bool] + (1 - alpha) * arr[mask_bool]).astype(
        np.uint8
    )
    mask_uint8 = mask_arr.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(arr, contours, -1, (0, 255, 0), thickness=2)
    return Image.fromarray(arr)


def process_cork_folder(
    cork_folder_path,
    input_images_paths,
    out_dir="combined_cork_output",
    min_area=MIN_REGION_AREA,
):
    cork_masks = []
    orig_arrays = []
    cork_fnames = []
    contour_imgs = []
    mask_filtered_list = []
    valid_mask_indices = []
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cork_img_files = sorted(
        [
            f
            for f in os.listdir(cork_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    cork_img_files = cork_img_files[:2]
    for i, cork_file in enumerate(cork_img_files):
        mask, arr = mask_pink_regions(os.path.join(cork_folder_path, cork_file))
        mask_filtered, contour_img = filter_and_draw_contours(mask, arr, min_area)
        if np.sum(mask_filtered > 0) > min_area:
            cork_masks.append(mask_filtered)
            orig_arrays.append(arr)
            cork_fnames.append(cork_file)
            contour_img_pil = Image.fromarray(contour_img)
            contour_imgs.append(contour_img_pil)
            mask_filtered_list.append(mask_filtered)
            contour_img_pil.save(
                os.path.join(out_dir, f"{Path(cork_file).stem}_contour.png")
            )
            valid_mask_indices.append(i)
    if len(cork_masks) == 2:
        combined_mask = combine_masks_AND(cork_masks[0], cork_masks[1])
    elif len(cork_masks) == 1:
        combined_mask = cork_masks[0]
    else:
        raise RuntimeError(
            "No valid cork mask images found (all below area threshold)."
        )
    combined_mask_img = Image.fromarray(combined_mask)
    combined_mask_img.save(os.path.join(out_dir, "combined_mask.png"))
    combined_outputs = []
    for idx in valid_mask_indices:
        orig_img_path = input_images_paths[idx]
        out_img = overlay_on_original(orig_img_path, combined_mask)
        out_fname = f"{Path(orig_img_path).stem}_combinedcork.png"
        out_img.save(os.path.join(out_dir, out_fname))
        combined_outputs.append((out_fname, out_img))
    return {
        "cork_mask_files": [os.path.join(cork_folder_path, f) for f in cork_fnames],
        "contour_images": [
            os.path.join(out_dir, f"{Path(f).stem}_contour.png") for f in cork_fnames
        ],
        "combined_mask": os.path.join(out_dir, "combined_mask.png"),
        "combined_overlays": [
            os.path.join(out_dir, fname) for fname, _ in combined_outputs
        ],
        "images_pil": {
            "cork_masks": [
                Image.open(os.path.join(cork_folder_path, f)) for f in cork_fnames
            ],
            "contour_imgs": contour_imgs,
            "combined_mask": combined_mask_img,
            "combined_overlays": [img for _, img in combined_outputs],
        },
    }