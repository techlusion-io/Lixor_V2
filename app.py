import streamlit as st
import os
import shutil
import sys
import subprocess
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2
import subprocess
import sys

try:
    import cv2
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.0.76"]
    )
    import cv2

from calculate_cork import process_cork_folder
from calculate_liquid import process_liquid_folder
from calculate_vessel import process_vessel_folder
from evaluation_liquid_percentage import calculate_liquid_percentage

MIN_REGION_AREA = 1000
INPUT_IMAGES_FOLDER = "input images"
OUTPUT_FOLDER = "output images"
RUN_PREDICTION_SCRIPT = "RunPredictionOnFolder.py"
MODEL_PATH = (
    "logs/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
)

st.set_page_config(
    page_title="LIXOR - Liquid Fill Level Detector", layout="centered"
)
st.title("Liquid Fill Level Detector")


def clear_folder(folder_path):
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


def resize_mask_to_shape(mask, target_shape):
    from PIL import Image as PILImage

    if mask is None:
        return None
    if mask.shape != target_shape[:2]:
        mask_pil = PILImage.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((target_shape[1], target_shape[0]), PILImage.NEAREST)
        return np.array(mask_pil)
    return mask


def show_mask_debug(mask_file, label):
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


def overlay_mask_on_img(arr, mask, color, alpha=0.5, contour_color=(0, 255, 0)):
    if mask is None:
        return arr
    arr_overlay = arr.copy()
    arr_overlay[mask > 0] = (
        alpha * color + (1 - alpha) * arr_overlay[mask > 0]
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(arr_overlay, contours, -1, contour_color, thickness=2)
    return arr_overlay


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
        # st.success(
        #     "Files uploaded and old input/output files cleared. Ready to run segmentation."
        # )

        st.subheader("Stage 1: Input Images")
        with st.expander("CLick to see uploaded images"):
            cols = st.columns(2)
            for i, file in enumerate(uploaded_files):
                with cols[i]:
                    st.image(
                        file, caption=f"Input {i+1}: {file.name}", use_container_width=True
                    )

        if st.button("Run Segmentation"):
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
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # st.text("Model output:\n" + result.stdout)
            except subprocess.CalledProcessError as e:
                st.error(f"Segmentation script failed:\n{e.stderr}")
                st.stop()

            input_img_paths = [
                os.path.join(
                    INPUT_IMAGES_FOLDER, Path(f.name).stem + "." + f.type.split("/")[-1]
                )
                for f in uploaded_files
            ]

            # --- Show model's raw output masks for Cork, Liquid, Vessel ---
            st.header("Stage 2: Detecting Raw masks for liquid...")
            with st.expander("Click to view staged Images"):
                mask_classes = [
                    ("V Cork", "Cork"),
                    ("Liquid GENERAL", "Liquid"),
                    ("Vessel", "Vessel"),
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
                                mask_file, f"{label} - Input {i+1} ({Path(img_path).name})"
                            )
                    else:
                        st.warning(f"Folder '{mask_folder}' does not exist.")

                # -- Cork --
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

                # -- Liquid --
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

                # -- Vessel --
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
            st.header("Stage 3: Getting The Best Masks...")
            # --- Show overlays ---
            orig_mask_imgs = []
            orig_titles = []
            # CORK
            if cork_result["best_mask"] is not None:
                idx = cork_result["best_input_idx"]
                mask = cork_result["best_mask"]
                img = Image.open(input_img_paths[idx]).convert("RGB")
                arr = np.array(img).astype(np.uint8)
                mask_rs = resize_mask_to_shape(mask, arr.shape)
                # st.info(
                #     f"Best cork mask: Image {idx+1} ({Path(input_img_paths[idx]).name}), Mask area: {np.sum(mask_rs > 0)}"
                # )
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
                # st.info(
                #     f"Best liquid mask: Image {idx+1} ({Path(input_img_paths[idx]).name}), Mask area: {np.sum(mask_rs > 0)}"
                # )
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask_rs, np.array([255, 128, 0]), 0.5)
                )
                orig_titles.append(
                    f"Liquid mask on image {idx+1} ({Path(input_img_paths[idx]).name})"
                )
            else:
                orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
                orig_titles.append("No valid liquid mask.")
            # VESSEL always on liquid-image if liquid exists
            if (
                liquid_result["best_input_idx"] is not None
                and vessel_result["best_mask"] is not None
            ):
                liquid_idx = liquid_result["best_input_idx"]
                vessel_mask = vessel_result["best_mask"]
                vessel_img = Image.open(input_img_paths[liquid_idx]).convert("RGB")
                arr = np.array(vessel_img).astype(np.uint8)
                mask = resize_mask_to_shape(vessel_mask, arr.shape)
                # st.info(
                #     f"Vessel mask displayed on image {liquid_idx+1} ({Path(input_img_paths[liquid_idx]).name}), Mask area: {np.sum(mask > 0)}"
                # )
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask, np.array([255, 0, 255]), 0.5)
                )
                orig_titles.append(
                    f"Vessel mask (magenta) on liquid-image {liquid_idx+1} ({Path(input_img_paths[liquid_idx]).name})"
                )
            elif vessel_result["best_mask"] is not None:
                vessel_idx = vessel_result["best_input_idx"]
                vessel_mask = vessel_result["best_mask"]
                vessel_img = Image.open(input_img_paths[vessel_idx]).convert("RGB")
                arr = np.array(vessel_img).astype(np.uint8)
                mask = resize_mask_to_shape(vessel_mask, arr.shape)
                st.info(
                    f"Vessel mask fallback on image {vessel_idx+1} ({Path(input_img_paths[vessel_idx]).name}), Mask area: {np.sum(mask > 0)}"
                )
                orig_mask_imgs.append(
                    overlay_mask_on_img(arr, mask, np.array([255, 0, 255]), 0.5)
                )
                orig_titles.append(f"Vessel mask (magenta) fallback overlay")
            else:
                orig_mask_imgs.append(np.zeros((100, 100, 3), dtype=np.uint8))
                orig_titles.append("No valid vessel mask.")

            st.markdown("#### Best Mask Overlays")
            cols = st.columns(3)
            for i, (img, title) in enumerate(zip(orig_mask_imgs, orig_titles)):
                with cols[i]:
                    st.image(img, caption=title, use_container_width=True)

            st.subheader("🍾 Liquid Fill Detection Result:")
            liquid_mask = liquid_result["best_mask"]
            vessel_mask = vessel_result["best_mask"]
            cork_mask = cork_result["best_mask"]
            vessel_area = np.sum(vessel_mask > 0)
            liquid_area = np.sum(liquid_mask > 0)
            cork_area = np.sum(cork_mask > 0) if "cork_mask" in locals() else 0  # Use cork if available

            denom = max(vessel_area - cork_area, 1)
            fill_percent = (liquid_area / denom) * 100 

            if fill_percent < 20:
                percentage_color = "red"
            elif fill_percent < 80:
                percentage_color = "#f2c200"
            else:
                percentage_color = "green"

            st.markdown(
                f"""
                <div style='font-size:2.3rem; font-weight: bold; color: {percentage_color};'>
                    Liquid Filled: {fill_percent:.2f}%
                </div>
                """, unsafe_allow_html=True
            )

            # --- Quality scores --
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
            # st.header(" 🍾 Liquid Fill Percentage Estimation")

            # liquid_mask = liquid_result["best_mask"]
            # vessel_mask = vessel_result["best_mask"]
            # cork_mask = cork_result["best_mask"]

            # if (liquid_mask is not None) and (vessel_mask is not None):
            #     percent, details = calculate_liquid_percentage(
            #         liquid_mask=liquid_mask, vessel_mask=vessel_mask, cork_mask=cork_mask
            #     )
            #     percent_str = f"{percent:.2f}%"
            #     st.markdown(
            #         f"<div style='text-align:center; font-size:2.5em; color:#1a79d4; font-weight:bold;'>"
            #         f"Liquid Fill: {percent_str}</div>",
            #         unsafe_allow_html=True,
            #     )
            #     # st.write("Area details:", details)
            # else:
            #     st.warning("Cannot estimate liquid percentage: missing mask(s).")
