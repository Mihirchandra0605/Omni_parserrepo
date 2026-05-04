import os
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
# PROCESS SINGLE IMAGE
# -----------------------------
def process_image(image_path, output_path):

    try:
        image = Image.open(image_path).convert("RGB")

        img_w, img_h = image.size

        # --- CONFIG (same as gradio) ---
        box_threshold = 0.05
        iou_threshold = 0.1
        imgsz = 640
        use_paddleocr = False

        # --- DRAW CONFIG ---
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
        dino_labeled_img, _, _ = get_som_labeled_img(
            image,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=False,  # doesn't matter now
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )

        # --- DECODE IMAGE ---
        vis_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))

        # --- SAVE ---
        vis_image.save(output_path)

        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {image_path}")
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

            # --- build output path ---
            relative_path = os.path.relpath(root, INPUT_ROOT)
            output_dir = os.path.join(OUTPUT_ROOT, relative_path)

            os.makedirs(output_dir, exist_ok=True)

            name, ext = os.path.splitext(file)
            output_file = f"{name}_omni{ext}"

            output_path = os.path.join(output_dir, output_file)

            process_image(input_path, output_path)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    run_all()