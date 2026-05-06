import os
import json
from PIL import Image
import base64
import io

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img
)

# -----------------------------
# PATH CONFIG
# -----------------------------
INPUT_ROOT = "images"
OUTPUT_ROOT = "outputs_omni"

# -----------------------------
# LOAD MODELS (ONCE)
# -----------------------------
print("🚀 Loading models...")

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')

caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)

print("✅ Models loaded\n")


# -----------------------------
# CONVERT TO JSON
# -----------------------------
def convert_to_json(elements, w, h):

    compos = []

    for idx, elem in enumerate(elements):

        bbox = elem['bbox']  # normalized

        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        comp = {
            "id": idx,
            "class": "Compo",
            "height": y2 - y1,
            "width": x2 - x1,
            "position": {
                "column_min": x1,
                "row_min": y1,
                "column_max": x2,
                "row_max": y2
            },
            "text": elem.get("content"),
            "label": elem.get("type")
        }

        compos.append(comp)

    return {"compos": compos}


# -----------------------------
# PROCESS SINGLE IMAGE
# -----------------------------
def process_image(image_path, output_img_path, output_json_path):

    try:
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # --- CONFIG ---
        box_threshold = 0.05
        iou_threshold = 0.1
        imgsz = 640
        use_paddleocr = False

        box_overlay_ratio = img_w / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # --- OCR ---
        ocr_bbox_rslt, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=use_paddleocr
        )

        text, ocr_bbox = ocr_bbox_rslt

        # --- OMNIPARSER ---
        encoded_img, label_coords, elements = get_som_labeled_img(
            image,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )

        # --- SAVE VIS IMAGE ---
        vis_image = Image.open(io.BytesIO(base64.b64decode(encoded_img)))
        vis_image.save(output_img_path)

        # --- SAVE JSON ---
        json_data = convert_to_json(elements, img_w, img_h)

        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f" Saved: {output_img_path}")
        print(f" Saved: {output_json_path}")

    except Exception as e:
        print(f" Error processing {image_path}")
        print(e)


# -----------------------------
# BATCH RUN
# -----------------------------
def run_all():

    for root, dirs, files in os.walk(INPUT_ROOT):

        for file in files:

            if not file.endswith(".png"):
                continue

            input_path = os.path.join(root, file)

            # --- folder structure preserve ---
            relative_path = os.path.relpath(root, INPUT_ROOT)
            output_dir = os.path.join(OUTPUT_ROOT, relative_path)

            os.makedirs(output_dir, exist_ok=True)

            name, ext = os.path.splitext(file)

            # outputs
            output_img = os.path.join(output_dir, f"{name}_omni.png")
            output_json = os.path.join(output_dir, f"{name}_omni.json")

            process_image(input_path, output_img, output_json)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    run_all()