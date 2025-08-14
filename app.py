# import streamlit as st
# import os
# import shutil
# import sys
# from pathlib import Path
# from PIL import Image, ImageOps
# import numpy as np
# import cv2
# import subprocess

# from calculate_cork import process_cork_folder
# from calculate_liquid import process_liquid_folder
# from calculate_vessel import process_vessel_folder
# from evaluation_liquid_percentage import (
#     calculate_liquid_percentage,
# )

# from calculate_base_overlay import process_base_folder

# try:
#     from ultralytics import YOLO

#     _YOLO_AVAILABLE = True
# except Exception:
#     _YOLO_AVAILABLE = False

# MIN_REGION_AREA = 1000
# FILL_ERROR_RATE = 0.05
# INPUT_IMAGES_FOLDER = "input images"
# OUTPUT_FOLDER = "output images"
# RUN_PREDICTION_SCRIPT = "RunPredictionOnFolder.py"
# MODEL_PATH = (
#     "logs/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
# )

# YOLO_BASE_WEIGHTS = "logs/best.pt"
# BASE_OUTPUT_SUBFOLDER = (
#     "Base"
# )

# CLASS_OUTPUT_SUBDIRS = ["V Cork", "Liquid GENERAL", "Vessel", BASE_OUTPUT_SUBFOLDER]

# COMBINED_OUTPUT_DIRS = [
#     "combined_base_output",
#     "combined_cork_output",
#     "combined_liquid_output",
#     "combined_vessel_output",
# ]

# st.set_page_config(page_title="LIXOR - Liquid Fill Level Detector", layout="centered")
# st.title("Liquid Fill Level Detector")


# def clear_folder(folder_path: str):
#     """Deletes all files/subfolders in a folder; creates it if missing."""
#     if os.path.exists(folder_path):
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print(f"Failed to delete {file_path}. Reason: {e}")
#     else:
#         os.makedirs(folder_path, exist_ok=True)


# def ensure_dir(p: str):
#     os.makedirs(p, exist_ok=True)

# ensure_dir(INPUT_IMAGES_FOLDER)
# ensure_dir(OUTPUT_FOLDER)
# for sub in CLASS_OUTPUT_SUBDIRS:
#     ensure_dir(os.path.join(OUTPUT_FOLDER, sub))

# def resize_mask_to_shape(mask: np.ndarray, target_shape):
#     """Resize uint8 mask (H,W) to match target (H,W,3) or (H,W)."""
#     from PIL import Image as PILImage

#     if mask is None:
#         return None
#     th, tw = (
#         (target_shape[0], target_shape[1]) if len(target_shape) >= 2 else target_shape
#     )
#     if mask.shape[0] == th and mask.shape[1] == tw:
#         return mask
#     mask_pil = PILImage.fromarray(mask.astype(np.uint8))
#     mask_pil = mask_pil.resize((tw, th), PILImage.NEAREST)
#     return np.array(mask_pil)


# def show_mask_debug(mask_file: str, label: str):
#     """Stage 2 preview: display the raw class overlay and estimate pink area."""
#     if mask_file and os.path.exists(mask_file):
#         mask_img = Image.open(mask_file)
#         st.image(mask_img, caption=f"Raw Mask: {label}", use_container_width=True)
#         arr = np.array(mask_img)
#         mask_area = np.sum((arr[:, :, 0] > 180) & (arr[:, :, 2] > 100))
#         st.info(f"Mask area for {label}: {mask_area} pixels")
#         return mask_area
#     else:
#         st.warning(f"No mask file found for {label}")
#         return 0


# def overlay_mask_on_img(
#     arr: np.ndarray,
#     mask: np.ndarray,
#     color: np.ndarray,
#     alpha=0.5,
#     contour_color=(0, 255, 0),
# ):
#     """Stage 3 preview: overlay a binary mask color onto original RGB."""
#     if mask is None:
#         return arr
#     arr_overlay = arr.copy()
#     m = mask > 0
#     arr_overlay[m] = (alpha * color + (1 - alpha) * arr_overlay[m]).astype(np.uint8)
#     contours, _ = cv2.findContours(
#         (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     cv2.drawContours(arr_overlay, contours, -1, contour_color, thickness=2)
#     return arr_overlay

# def run_yolo_and_emit_base_overlays(input_img_paths):
#     """
#     Runs YOLO v11-seg using weights at YOLO_BASE_WEIGHTS and writes pink-magenta overlays
#     ONLY for the 'base' class into: output images/Base/<stem>.png

#     - Creates/clears the 'Base' folder each run (avoids stale files).
#     - Uses imgsz=640 explicitly (prevents imgsz=None errors in some Ultralytics builds).
#     - Filters strictly to class id 1 (your YAML: 0=bottle, 1=base).
#     """
#     base_dir = os.path.join(OUTPUT_FOLDER, BASE_OUTPUT_SUBFOLDER)
#     ensure_dir(base_dir)

#     for f in os.listdir(base_dir):
#         try:
#             os.remove(os.path.join(base_dir, f))
#         except Exception:
#             pass

#     if not _YOLO_AVAILABLE:
#         st.warning(
#             "YOLO not available — cannot generate Base overlays. Install `ultralytics`."
#         )
#         return

#     if not os.path.exists(YOLO_BASE_WEIGHTS):
#         st.warning(
#             f"YOLO base weights not found at {YOLO_BASE_WEIGHTS}. Skipping Base overlays."
#         )
#         return

#     model = YOLO(YOLO_BASE_WEIGHTS)
#     device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

#     BASE_CLS_ID = 1

#     for img_path in input_img_paths:
#         save_path = os.path.join(base_dir, f"{Path(img_path).stem}.png")

#         try:
#             results = model.predict(
#                 source=img_path,
#                 conf=0.25,
#                 iou=0.5,
#                 imgsz=640,
#                 verbose=False,
#                 device=device,
#             )
#         except Exception as e:
#             st.warning(f"YOLO base prediction failed on {Path(img_path).name}: {e}")
#             Image.open(img_path).convert("RGB").save(save_path)
#             continue

#         if not results or results[0].masks is None or len(results[0].masks.data) == 0:
#             Image.open(img_path).convert("RGB").save(save_path)
#             continue

#         res = results[0]
#         pil = Image.open(img_path).convert("RGB")
#         rgb = np.array(pil)
#         h, w = rgb.shape[:2]

#         union = np.zeros((h, w), dtype=np.uint8)
#         for inst_idx in range(len(res.masks.data)):
#             cls_id = (
#                 int(res.boxes.cls[inst_idx].item())
#                 if (res.boxes is not None and res.boxes.cls is not None)
#                 else -1
#             )
#             if cls_id != BASE_CLS_ID:
#                 continue
#             m = res.masks.data[inst_idx].cpu().numpy().astype(np.uint8)
#             m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
#             union = np.maximum(union, m)


#         rgb_overlay = rgb.copy()
#         if union.any():
#             m = union.astype(bool)
#             rgb_overlay[m, 0] = 255
#             rgb_overlay[m, 1] = 0
#             rgb_overlay[m, 2] = 255
#         Image.fromarray(rgb_overlay).save(save_path)

# uploaded_files = st.file_uploader(
#     "Upload **2 images** (jpg/png):",
#     type=["jpg", "jpeg", "png"],
#     accept_multiple_files=True,
# )

# if uploaded_files:
#     clear_folder(INPUT_IMAGES_FOLDER)
#     clear_folder(OUTPUT_FOLDER)

#     if len(uploaded_files) != 2:
#         st.warning("Please upload exactly 2 images.")
#     else:
#         for file in uploaded_files:
#             filename = Path(file.name).stem
#             ext = file.type.split("/")[-1]
#             save_path = os.path.join(INPUT_IMAGES_FOLDER, f"{filename}.{ext}")
#             img = Image.open(file)
#             img = ImageOps.exif_transpose(img)
#             img.save(save_path)
#         for d in COMBINED_OUTPUT_DIRS:
#             clear_folder(d)

#         st.subheader("Stage 1: Input Images")
#         with st.expander("CLick to see uploaded images"):
#             cols = st.columns(2)
#             for i, file in enumerate(uploaded_files):
#                 with cols[i]:
#                     st.image(
#                         file,
#                         caption=f"Input {i+1}: {file.name}",
#                         use_container_width=True,
#                     )

#         if st.button("Run Segmentation"):
#             for d in COMBINED_OUTPUT_DIRS:
#                 clear_folder(d)
#             st.info("Running segmentation...")
#             cmd = [
#                 sys.executable,
#                 RUN_PREDICTION_SCRIPT,
#                 "--inputdir",
#                 INPUT_IMAGES_FOLDER,
#                 "--outdir",
#                 OUTPUT_FOLDER,
#                 "--trainedmodel",
#                 MODEL_PATH,
#                 "--gpu",
#                 "1",
#                 "--freeze",
#                 "1",
#             ]
#             try:
#                 subprocess.run(cmd, capture_output=True, text=True, check=True)
#             except subprocess.CalledProcessError as e:
#                 st.error(f"Segmentation script failed:\n{e.stderr}")
#                 st.stop()

#             input_img_paths = [
#                 os.path.join(
#                     INPUT_IMAGES_FOLDER, Path(f.name).stem + "." + f.type.split("/")[-1]
#                 )
#                 for f in uploaded_files
#             ]

#             run_yolo_and_emit_base_overlays(input_img_paths)

#             st.header("Stage 2: Detecting Raw masks for liquid...")
#             with st.expander("Click to view staged Images"):
#                 mask_classes = [
#                     ("V Cork", "Cork"),
#                     ("Liquid GENERAL", "Liquid"),
#                     ("Vessel", "Vessel"),
#                     ("Base", "Base"),  # NEW
#                 ]
#                 for mask_folder, label in mask_classes:
#                     st.subheader(f"Segmentation Class: {label}")
#                     folder_path = os.path.join(OUTPUT_FOLDER, mask_folder)
#                     if os.path.exists(folder_path):
#                         for i, img_path in enumerate(input_img_paths):
#                             mask_file = None
#                             for f in os.listdir(folder_path):
#                                 if Path(f).stem.startswith(Path(img_path).stem):
#                                     mask_file = os.path.join(folder_path, f)
#                                     break
#                             show_mask_debug(
#                                 mask_file,
#                                 f"{label} - Input {i+1} ({Path(img_path).name})",
#                             )
#                     else:
#                         st.warning(f"Folder '{mask_folder}' does not exist.")

#                 cork_folder = os.path.join(OUTPUT_FOLDER, "V Cork")
#                 try:
#                     cork_result = process_cork_folder(
#                         cork_folder_path=cork_folder,
#                         input_images_paths=input_img_paths,
#                         out_dir="combined_cork_output",
#                         min_area=MIN_REGION_AREA,
#                     )
#                 except Exception as e:
#                     st.warning(f"Could not process cork masks: {e}")
#                     cork_result = {
#                         "mask_files": [],
#                         "best_mask": None,
#                         "best_mask_file": None,
#                         "best_input_idx": None,
#                         "detailed_scores": [],
#                         "best_score_dict": None,
#                     }

#                 liquid_folder = os.path.join(OUTPUT_FOLDER, "Liquid GENERAL")
#                 try:
#                     liquid_result = process_liquid_folder(
#                         liquid_folder_path=liquid_folder,
#                         input_images_paths=input_img_paths,
#                         out_dir="combined_liquid_output",
#                         min_area=MIN_REGION_AREA,
#                     )
#                 except Exception as e:
#                     st.warning(f"Could not process liquid masks: {e}")
#                     liquid_result = {
#                         "mask_files": [],
#                         "best_mask": None,
#                         "best_mask_file": None,
#                         "best_input_idx": None,
#                         "detailed_scores": [],
#                         "best_score_dict": None,
#                     }

#                 vessel_folder = os.path.join(OUTPUT_FOLDER, "Vessel")
#                 try:
#                     vessel_result = process_vessel_folder(
#                         vessel_folder_path=vessel_folder,
#                         input_images_paths=input_img_paths,
#                         out_dir="combined_vessel_output",
#                         min_area=MIN_REGION_AREA,
#                     )
#                 except Exception as e:
#                     st.warning(f"Could not process vessel masks: {e}")
#                     vessel_result = {
#                         "mask_files": [],
#                         "best_mask": None,
#                         "best_mask_file": None,
#                         "best_input_idx": None,
#                         "detailed_scores": [],
#                         "best_score_dict": None,
#                     }

#                 base_folder = os.path.join(OUTPUT_FOLDER, "Base")
#                 try:
#                     base_result = process_base_folder(
#                         base_folder_path=base_folder,
#                         input_images_paths=input_img_paths,
#                         out_dir="combined_base_output",
#                         min_area=MIN_REGION_AREA,
#                     )
#                 except Exception as e:
#                     st.warning(f"Could not process base masks: {e}")
#                     base_result = {
#                         "mask_files": [],
#                         "best_mask": None,
#                         "best_mask_file": None,
#                         "best_input_idx": None,
#                         "detailed_scores": [],
#                         "best_score_dict": None,
#                     }

#             st.header("Stage 3: Getting The Best Masks...")
#             orig_mask_imgs = []
#             orig_titles = []

#             # CORK
#             if cork_result["best_mask"] is not None:
#                 idx = cork_result["best_input_idx"]
#                 mask = cork_result["best_mask"]
#                 img = Image.open(input_img_paths[idx]).convert("RGB")
#                 arr = np.array(img).astype(np.uint8)
#                 mask_rs = resize_mask_to_shape(mask, arr.shape)
#                 orig_mask_imgs.append(
#                     overlay_mask_on_img(arr, mask_rs, np.array([255, 255, 0]), 0.5)
#                 )
#                 orig_titles.append(
#                     f"Cork mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
#                 )
#             else:
#                 orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
#                 orig_titles.append("No valid cork mask.")

#             # LIQUID
#             if liquid_result["best_mask"] is not None:
#                 idx = liquid_result["best_input_idx"]
#                 mask = liquid_result["best_mask"]
#                 img = Image.open(input_img_paths[idx]).convert("RGB")
#                 arr = np.array(img).astype(np.uint8)
#                 mask_rs = resize_mask_to_shape(mask, arr.shape)
#                 orig_mask_imgs.append(
#                     overlay_mask_on_img(arr, mask_rs, np.array([255, 128, 0]), 0.5)
#                 )
#                 orig_titles.append(
#                     f"Liquid mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
#                 )
#             else:
#                 orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
#                 orig_titles.append("No valid liquid mask.")

#             # VESSEL
#             if vessel_result["best_mask"] is not None:
#                 idx = vessel_result["best_input_idx"]
#                 mask = vessel_result["best_mask"]
#                 img = Image.open(input_img_paths[idx]).convert("RGB")
#                 arr = np.array(img).astype(np.uint8)
#                 mask_rs = resize_mask_to_shape(mask, arr.shape)
#                 orig_mask_imgs.append(
#                     overlay_mask_on_img(arr, mask_rs, np.array([255, 0, 255]), 0.5)
#                 )
#                 orig_titles.append(
#                     f"Vessel mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
#                 )
#             else:
#                 orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
#                 orig_titles.append("No valid vessel mask.")

#             # BASE
#             if base_result["best_mask"] is not None:
#                 idx = base_result["best_input_idx"]
#                 mask = base_result["best_mask"]
#                 img = Image.open(input_img_paths[idx]).convert("RGB")
#                 arr = np.array(img).astype(np.uint8)
#                 mask_rs = resize_mask_to_shape(mask, arr.shape)
#                 orig_mask_imgs.append(
#                     overlay_mask_on_img(arr, mask_rs, np.array([0, 255, 255]), 0.5)
#                 )
#                 orig_titles.append(
#                     f"Base mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
#                 )
#             else:
#                 orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
#                 orig_titles.append("No valid base mask.")

#             st.markdown("#### Best Mask Overlays")
#             cols = st.columns(4)
#             for i, (img, title) in enumerate(zip(orig_mask_imgs, orig_titles)):
#                 with cols[i]:
#                     st.image(img, caption=title, use_container_width=True)

#             st.subheader("🍾 Liquid Fill Detection Result:")

#             def to_u8(m):
#                 return ((m > 0).astype(np.uint8) * 255) if m is not None else None

#             liquid_u8 = to_u8(liquid_result["best_mask"])
#             vessel_u8 = to_u8(vessel_result["best_mask"])
#             cork_u8 = to_u8(cork_result["best_mask"])
#             base_u8 = to_u8(base_result["best_mask"])

#             ref_arr = np.array(Image.open(input_img_paths[0]).convert("RGB"))
#             H, W = ref_arr.shape[:2]
#             if liquid_u8 is not None:
#                 liquid_u8 = resize_mask_to_shape(liquid_u8, (H, W))
#             if vessel_u8 is not None:
#                 vessel_u8 = resize_mask_to_shape(vessel_u8, (H, W))
#             if cork_u8 is not None:
#                 cork_u8 = resize_mask_to_shape(cork_u8, (H, W))
#             if base_u8 is not None:
#                 base_u8 = resize_mask_to_shape(base_u8, (H, W))

#             base_inside_vessel = (
#                 cv2.bitwise_and(base_u8, vessel_u8)
#                 if (base_u8 is not None and vessel_u8 is not None)
#                 else None
#             )

#             area = lambda m: (
#                 int(np.sum((m > 0).astype(np.uint8))) if m is not None else 0
#             )
#             vessel_area = area(vessel_u8)
#             liquid_area = area(liquid_u8)
#             cork_area = area(cork_u8)
#             base_area = area(base_inside_vessel)

#             if vessel_area > 0 and base_area > 0:
#                 overlap_ratio = base_area / vessel_area
#                 if overlap_ratio > 0.60 or overlap_ratio < 0.02:
#                     base_area = 0

#             denom = max(vessel_area - cork_area - base_area, 1)
#             fill_percent = (liquid_area / denom) * 100.0
#             err = FILL_ERROR_RATE * 100.0
#             final_fill_percentage = fill_percent + err

#             if final_fill_percentage < 20:
#                 percentage_color = "red"
#             elif final_fill_percentage < 80:
#                 percentage_color = "#f2c200"
#             else:
#                 percentage_color = "green"

#             st.markdown(
#                 f"""
#                 <div style='font-size:2.3rem; font-weight: bold; color: {percentage_color};'>
#                     Liquid Filled: {final_fill_percentage:.2f}%
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#             with st.expander("Show Quality Scores"):
#                 st.markdown("### Cork Mask Quality Scores")
#                 if cork_result["mask_files"]:
#                     for i, score_dict in enumerate(cork_result["detailed_scores"]):
#                         st.write(
#                             f"Cork Image {i+1} ({Path(cork_result['mask_files'][i]).name if cork_result['mask_files'][i] else 'None'}):"
#                         )
#                         st.json(score_dict)
#                     if cork_result["best_mask_file"]:
#                         st.success(
#                             f"Best cork mask: {Path(cork_result['best_mask_file']).name} (area: {cork_result['best_score_dict']['area']})"
#                         )
#                 else:
#                     st.info("No valid cork mask found.")

#                 st.markdown("### Liquid Mask Quality Scores")
#                 if liquid_result["mask_files"]:
#                     for i, score_dict in enumerate(liquid_result["detailed_scores"]):
#                         st.write(
#                             f"Liquid Image {i+1} ({Path(liquid_result['mask_files'][i]).name if liquid_result['mask_files'][i] else 'None'}):"
#                         )
#                         st.json(score_dict)
#                     if liquid_result["best_mask_file"]:
#                         st.success(
#                             f"Best liquid mask: {Path(liquid_result['best_mask_file']).name} (area: {liquid_result['best_score_dict']['area']})"
#                         )
#                 else:
#                     st.info("No valid liquid mask found.")

#                 st.markdown("### Vessel Mask Quality Scores")
#                 if vessel_result["mask_files"]:
#                     for i, score_dict in enumerate(vessel_result["detailed_scores"]):
#                         st.write(
#                             f"Vessel Image {i+1} ({Path(vessel_result['mask_files'][i]).name if vessel_result['mask_files'][i] else 'None'}):"
#                         )
#                         st.json(score_dict)
#                     if vessel_result["best_mask_file"]:
#                         st.success(
#                             f"Best vessel mask: {Path(vessel_result['best_mask_file']).name} (area: {vessel_result['best_score_dict']['area']})"
#                         )
#                 else:
#                     st.info("No valid vessel mask found.")


import streamlit as st
import os
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2
import subprocess

from calculate_cork import process_cork_folder
from calculate_liquid import process_liquid_folder
from calculate_vessel import process_vessel_folder
from evaluation_liquid_percentage import (
    calculate_liquid_percentage,
)

from calculate_base_overlay import process_base_folder

try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

MIN_REGION_AREA = 1000
FILL_ERROR_RATE = 0.05
INPUT_IMAGES_FOLDER = "input images"
OUTPUT_FOLDER = "output images"
RUN_PREDICTION_SCRIPT = "RunPredictionOnFolder.py"
MODEL_PATH = (
    "logs/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
)

YOLO_BASE_WEIGHTS = "logs/best.pt"
BASE_OUTPUT_SUBFOLDER = "Base"

CLASS_OUTPUT_SUBDIRS = ["V Cork", "Liquid GENERAL", "Vessel", BASE_OUTPUT_SUBFOLDER]

COMBINED_OUTPUT_DIRS = [
    "combined_base_output",
    "combined_cork_output",
    "combined_liquid_output",
    "combined_vessel_output",
]

st.set_page_config(page_title="LIXOR - Liquid Fill Level Detector", layout="centered")
st.title("Liquid Fill Level Detector")


def clear_folder(folder_path: str):
    """Deletes all files/subfolders in a folder; creates it if missing."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path, exist_ok=True)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# Ensure base folders exist at app start
ensure_dir(INPUT_IMAGES_FOLDER)
ensure_dir(OUTPUT_FOLDER)
for sub in CLASS_OUTPUT_SUBDIRS:
    ensure_dir(os.path.join(OUTPUT_FOLDER, sub))


def resize_mask_to_shape(mask: np.ndarray, target_shape):
    """Resize uint8 mask (H,W) to match target (H,W,3) or (H,W)."""
    from PIL import Image as PILImage

    if mask is None:
        return None
    th, tw = (
        (target_shape[0], target_shape[1]) if len(target_shape) >= 2 else target_shape
    )
    if mask.shape[0] == th and mask.shape[1] == tw:
        return mask
    mask_pil = PILImage.fromarray(mask.astype(np.uint8))
    mask_pil = mask_pil.resize((tw, th), PILImage.NEAREST)
    return np.array(mask_pil)


def show_mask_debug(mask_file: str, label: str):
    """Stage 2 preview: display the raw class overlay and estimate pink area."""
    if mask_file and os.path.exists(mask_file):
        mask_img = Image.open(mask_file)
        st.image(mask_img, caption=f"Raw Mask: {label}", use_container_width=True)
        arr = np.array(mask_img)
        mask_area = np.sum((arr[:, :, 0] > 180) & (arr[:, :, 2] > 100))
        st.info(f"Mask area for {label}: {mask_area} pixels")
        return mask_area
    else:
        st.warning(f"No mask file found for {label}")
        return 0


def overlay_mask_on_img(
    arr: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha=0.5,
    contour_color=(0, 255, 0),
):
    """Stage 3 preview: overlay a binary mask color onto original RGB."""
    if mask is None:
        return arr
    arr_overlay = arr.copy()
    m = mask > 0
    arr_overlay[m] = (alpha * color + (1 - alpha) * arr_overlay[m]).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(arr_overlay, contours, -1, contour_color, thickness=2)
    return arr_overlay


def run_yolo_and_emit_base_overlays(input_img_paths):
    """
    Runs YOLO v11-seg using weights at YOLO_BASE_WEIGHTS and writes pink-magenta overlays
    ONLY for the 'base' class into: output images/Base/<stem>.png

    - Creates/clears the 'Base' folder each run (avoids stale files).
    - Uses imgsz=640 explicitly (prevents imgsz=None errors in some Ultralytics builds).
    - Filters strictly to class id 1 (your YAML: 0=bottle, 1=base).
    """
    base_dir = os.path.join(OUTPUT_FOLDER, BASE_OUTPUT_SUBFOLDER)
    ensure_dir(base_dir)

    for f in os.listdir(base_dir):
        try:
            os.remove(os.path.join(base_dir, f))
        except Exception:
            pass

    if not _YOLO_AVAILABLE:
        st.warning(
            "YOLO not available — cannot generate Base overlays. Install `ultralytics`."
        )
        return

    if not os.path.exists(YOLO_BASE_WEIGHTS):
        st.warning(
            f"YOLO base weights not found at {YOLO_BASE_WEIGHTS}. Skipping Base overlays."
        )
        return

    model = YOLO(YOLO_BASE_WEIGHTS)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    BASE_CLS_ID = 1

    for img_path in input_img_paths:
        save_path = os.path.join(base_dir, f"{Path(img_path).stem}.png")

        try:
            results = model.predict(
                source=img_path,
                conf=0.25,
                iou=0.5,
                imgsz=640,
                verbose=False,
                device=device,
            )
        except Exception as e:
            st.warning(f"YOLO base prediction failed on {Path(img_path).name}: {e}")
            Image.open(img_path).convert("RGB").save(save_path)
            continue

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            Image.open(img_path).convert("RGB").save(save_path)
            continue

        res = results[0]
        pil = Image.open(img_path).convert("RGB")
        rgb = np.array(pil)
        h, w = rgb.shape[:2]

        union = np.zeros((h, w), dtype=np.uint8)
        for inst_idx in range(len(res.masks.data)):
            cls_id = (
                int(res.boxes.cls[inst_idx].item())
                if (res.boxes is not None and res.boxes.cls is not None)
                else -1
            )
            if cls_id != BASE_CLS_ID:
                continue
            m = res.masks.data[inst_idx].cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            union = np.maximum(union, m)

        rgb_overlay = rgb.copy()
        if union.any():
            m = union.astype(bool)
            rgb_overlay[m, 0] = 255
            rgb_overlay[m, 1] = 0
            rgb_overlay[m, 2] = 255
        Image.fromarray(rgb_overlay).save(save_path)


uploaded_files = st.file_uploader(
    "Upload **2 images** (jpg/png):",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    clear_folder(INPUT_IMAGES_FOLDER)
    clear_folder(OUTPUT_FOLDER)

    if len(uploaded_files) != 2:
        st.warning("Please upload exactly 2 images.")
    else:
        for file in uploaded_files:
            filename = Path(file.name).stem
            ext = file.type.split("/")[-1]
            save_path = os.path.join(INPUT_IMAGES_FOLDER, f"{filename}.{ext}")
            img = Image.open(file)
            img = ImageOps.exif_transpose(img)
            img.save(save_path)
        for d in COMBINED_OUTPUT_DIRS:
            clear_folder(d)

        st.subheader("Stage 1: Input Images")
        with st.expander("CLick to see uploaded images"):
            cols = st.columns(2)
            for i, file in enumerate(uploaded_files):
                with cols[i]:
                    st.image(
                        file,
                        caption=f"Input {i+1}: {file.name}",
                        use_container_width=True,
                    )

        if st.button("Run Segmentation"):
            for d in COMBINED_OUTPUT_DIRS:
                clear_folder(d)
            st.info("Running segmentation...")
            cmd = [
                sys.executable,
                RUN_PREDICTION_SCRIPT,
                "--inputdir",
                INPUT_IMAGES_FOLDER,
                "--outdir",
                OUTPUT_FOLDER,
                "--trainedmodel",
                MODEL_PATH,
                "--gpu",
                "1",
                "--freeze",
                "1",
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                st.error(f"Segmentation script failed:\n{e.stderr}")
                st.stop()

            input_img_paths = [
                os.path.join(
                    INPUT_IMAGES_FOLDER, Path(f.name).stem + "." + f.type.split("/")[-1]
                )
                for f in uploaded_files
            ]

            run_yolo_and_emit_base_overlays(input_img_paths)

            st.header("Stage 2: Detecting Raw masks for liquid...")
            with st.expander("Click to view staged Images"):
                mask_classes = [
                    ("V Cork", "Cork"),
                    ("Liquid GENERAL", "Liquid"),
                    ("Vessel", "Vessel"),
                    ("Base", "Base"),  # NEW
                ]
                for mask_folder, label in mask_classes:
                    st.subheader(f"Segmentation Class: {label}")
                    folder_path = os.path.join(OUTPUT_FOLDER, mask_folder)
                    if os.path.exists(folder_path):
                        for i, img_path in enumerate(input_img_paths):
                            mask_file = None
                            for f in os.listdir(folder_path):
                                if Path(f).stem.startswith(Path(img_path).stem):
                                    mask_file = os.path.join(folder_path, f)
                                    break
                            show_mask_debug(
                                mask_file,
                                f"{label} - Input {i+1} ({Path(img_path).name})",
                            )
                    else:
                        st.warning(f"Folder '{mask_folder}' does not exist.")

                cork_folder = os.path.join(OUTPUT_FOLDER, "V Cork")
                try:
                    cork_result = process_cork_folder(
                        cork_folder_path=cork_folder,
                        input_images_paths=input_img_paths,
                        out_dir="combined_cork_output",
                        min_area=MIN_REGION_AREA,
                    )
                except Exception as e:
                    st.warning(f"Could not process cork masks: {e}")
                    cork_result = {
                        "mask_files": [],
                        "best_mask": None,
                        "best_mask_file": None,
                        "best_input_idx": None,
                        "detailed_scores": [],
                        "best_score_dict": None,
                    }

                liquid_folder = os.path.join(OUTPUT_FOLDER, "Liquid GENERAL")
                try:
                    liquid_result = process_liquid_folder(
                        liquid_folder_path=liquid_folder,
                        input_images_paths=input_img_paths,
                        out_dir="combined_liquid_output",
                        min_area=MIN_REGION_AREA,
                    )
                except Exception as e:
                    st.warning(f"Could not process liquid masks: {e}")
                    liquid_result = {
                        "mask_files": [],
                        "best_mask": None,
                        "best_mask_file": None,
                        "best_input_idx": None,
                        "detailed_scores": [],
                        "best_score_dict": None,
                    }

                vessel_folder = os.path.join(OUTPUT_FOLDER, "Vessel")
                try:
                    vessel_result = process_vessel_folder(
                        vessel_folder_path=vessel_folder,
                        input_images_paths=input_img_paths,
                        out_dir="combined_vessel_output",
                        min_area=MIN_REGION_AREA,
                    )
                except Exception as e:
                    st.warning(f"Could not process vessel masks: {e}")
                    vessel_result = {
                        "mask_files": [],
                        "best_mask": None,
                        "best_mask_file": None,
                        "best_input_idx": None,
                        "detailed_scores": [],
                        "best_score_dict": None,
                    }

                base_folder = os.path.join(OUTPUT_FOLDER, "Base")
                try:
                    base_result = process_base_folder(
                        base_folder_path=base_folder,
                        input_images_paths=input_img_paths,
                        out_dir="combined_base_output",
                        min_area=MIN_REGION_AREA,
                    )
                except Exception as e:
                    st.warning(f"Could not process base masks: {e}")
                    base_result = {
                        "mask_files": [],
                        "best_mask": None,
                        "best_mask_file": None,
                        "best_input_idx": None,
                        "detailed_scores": [],
                        "best_score_dict": None,
                    }

            st.header("Stage 3: Getting The Best Masks...")
            orig_mask_imgs = []
            orig_titles = []

            # CORK
            if cork_result["best_mask"] is not None:
                idx = cork_result["best_input_idx"]
                mask = cork_result["best_mask"]
                img = Image.open(input_img_paths[idx]).convert("RGB")
                arr = np.array(img).astype(np.uint8)
                mask_rs = resize_mask_to_shape(mask, arr.shape)
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask_rs, np.array([255, 255, 0]), 0.5)
                )
                orig_titles.append(
                    f"Cork mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
                )
            else:
                orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
                orig_titles.append("No valid cork mask.")

            # LIQUID
            if liquid_result["best_mask"] is not None:
                idx = liquid_result["best_input_idx"]
                mask = liquid_result["best_mask"]
                img = Image.open(input_img_paths[idx]).convert("RGB")
                arr = np.array(img).astype(np.uint8)
                mask_rs = resize_mask_to_shape(mask, arr.shape)
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask_rs, np.array([255, 128, 0]), 0.5)
                )
                orig_titles.append(
                    f"Liquid mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
                )
            else:
                orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
                orig_titles.append("No valid liquid mask.")

            # VESSEL
            if vessel_result["best_mask"] is not None:
                idx = vessel_result["best_input_idx"]
                mask = vessel_result["best_mask"]
                img = Image.open(input_img_paths[idx]).convert("RGB")
                arr = np.array(img).astype(np.uint8)
                mask_rs = resize_mask_to_shape(mask, arr.shape)
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask_rs, np.array([255, 0, 255]), 0.5)
                )
                orig_titles.append(
                    f"Vessel mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
                )
            else:
                orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
                orig_titles.append("No valid vessel mask.")

            # BASE
            if base_result["best_mask"] is not None:
                idx = base_result["best_input_idx"]
                mask = base_result["best_mask"]
                img = Image.open(input_img_paths[idx]).convert("RGB")
                arr = np.array(img).astype(np.uint8)
                mask_rs = resize_mask_to_shape(mask, arr.shape)
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask_rs, np.array([0, 255, 255]), 0.5)
                )
                orig_titles.append(
                    f"Base mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
                )
            else:
                orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
                orig_titles.append("No valid base mask.")

            st.markdown("#### Best Mask Overlays")
            cols = st.columns(4)
            for i, (img, title) in enumerate(zip(orig_mask_imgs, orig_titles)):
                with cols[i]:
                    st.image(img, caption=title, use_container_width=True)

            st.subheader("🍾 Liquid Fill Detection Result:")

            def to_u8(m):
                return ((m > 0).astype(np.uint8) * 255) if m is not None else None

            liquid_u8 = to_u8(liquid_result["best_mask"])
            vessel_u8 = to_u8(vessel_result["best_mask"])
            cork_u8 = to_u8(cork_result["best_mask"])
            base_u8 = to_u8(base_result["best_mask"])

            ref_arr = np.array(Image.open(input_img_paths[0]).convert("RGB"))
            H, W = ref_arr.shape[:2]
            if liquid_u8 is not None:
                liquid_u8 = resize_mask_to_shape(liquid_u8, (H, W))
            if vessel_u8 is not None:
                vessel_u8 = resize_mask_to_shape(vessel_u8, (H, W))
            if cork_u8 is not None:
                cork_u8 = resize_mask_to_shape(cork_u8, (H, W))
            if base_u8 is not None:
                base_u8 = resize_mask_to_shape(base_u8, (H, W))

            base_inside_vessel = (
                cv2.bitwise_and(base_u8, vessel_u8)
                if (base_u8 is not None and vessel_u8 is not None)
                else None
            )

            area = lambda m: (
                int(np.sum((m > 0).astype(np.uint8))) if m is not None else 0
            )
            vessel_area = area(vessel_u8)
            liquid_area = area(liquid_u8)
            cork_area = area(cork_u8)
            base_area = area(base_inside_vessel)

            if vessel_area > 0 and base_area > 0:
                overlap_ratio = base_area / vessel_area
                if overlap_ratio > 0.60 or overlap_ratio < 0.02:
                    base_area = 0

            denom = max(vessel_area - cork_area - base_area, 1)
            fill_percent = (liquid_area / denom) * 100.0
            err = FILL_ERROR_RATE * 100.0
            final_fill_percentage = fill_percent + err

            if final_fill_percentage < 20:
                percentage_color = "red"
            elif final_fill_percentage < 80:
                percentage_color = "#f2c200"
            else:
                percentage_color = "green"

            st.markdown(
                f"""
                <div style='font-size:2.3rem; font-weight: bold; color: {percentage_color};'>
                    Liquid Filled: {final_fill_percentage:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Show Quality Scores"):
                st.markdown("### Cork Mask Quality Scores")
                if cork_result["mask_files"]:
                    for i, score_dict in enumerate(cork_result["detailed_scores"]):
                        st.write(
                            f"Cork Image {i+1} ({Path(cork_result['mask_files'][i]).name if cork_result['mask_files'][i] else 'None'}):"
                        )
                        st.json(score_dict)
                    if cork_result["best_mask_file"]:
                        st.success(
                            f"Best cork mask: {Path(cork_result['best_mask_file']).name} (area: {cork_result['best_score_dict']['area']})"
                        )
                else:
                    st.info("No valid cork mask found.")

                st.markdown("### Liquid Mask Quality Scores")
                if liquid_result["mask_files"]:
                    for i, score_dict in enumerate(liquid_result["detailed_scores"]):
                        st.write(
                            f"Liquid Image {i+1} ({Path(liquid_result['mask_files'][i]).name if liquid_result['mask_files'][i] else 'None'}):"
                        )
                        st.json(score_dict)
                    if liquid_result["best_mask_file"]:
                        st.success(
                            f"Best liquid mask: {Path(liquid_result['best_mask_file']).name} (area: {liquid_result['best_score_dict']['area']})"
                        )
                else:
                    st.info("No valid liquid mask found.")

                st.markdown("### Vessel Mask Quality Scores")
                if vessel_result["mask_files"]:
                    for i, score_dict in enumerate(vessel_result["detailed_scores"]):
                        st.write(
                            f"Vessel Image {i+1} ({Path(vessel_result['mask_files'][i]).name if vessel_result['mask_files'][i] else 'None'}):"
                        )
                        st.json(score_dict)
                    if vessel_result["best_mask_file"]:
                        st.success(
                            f"Best vessel mask: {Path(vessel_result['best_mask_file']).name} (area: {vessel_result['best_score_dict']['area']})"
                        )
                else:
                    st.info("No valid vessel mask found.")
