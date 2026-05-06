import os
import json
from PIL import Image, ImageDraw
import base64
import io

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img
)

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "images/bootstrap/runtime_source_image_bootstrap_1280_720.png"   
OUTPUT_DIR = "debug_omni"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
print(" Loading models...")

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')

caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)

print(" Models loaded\n")


# -----------------------------
# RUN OMNIPARSER
# -----------------------------
def run_omni(image_path):

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # CONFIG (same as your batch script)
    box_threshold = 0.05
    iou_threshold = 0.1
    imgsz = 640
    use_paddleocr = False

    box_overlay_ratio = w / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # OCR
    ocr_bbox_rslt, _ = check_ocr_box(
        image,
        display_img=False,
        output_bb_format='xyxy',
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr
    )

    text, ocr_bbox = ocr_bbox_rslt

    # OMNI
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

    # decode visualization
    vis_img = Image.open(io.BytesIO(base64.b64decode(encoded_img)))
    vis_img.save(os.path.join(OUTPUT_DIR, "omni_vis.png"))

    return elements, (w, h)


# -----------------------------
# CONVERT TO YOUR JSON FORMAT
# -----------------------------
def convert_to_json(elements, w, h):

    compos = []

    for idx, elem in enumerate(elements):

        bbox = elem['bbox']  # normalized [0,1]

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
# RECONSTRUCT IMAGE FROM JSON
# -----------------------------
from PIL import ImageFont

def draw_from_json(image_path, json_data):

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for comp in json_data["compos"]:

        pos = comp["position"]
        comp_id = comp["id"]

        x1 = pos["column_min"]
        y1 = pos["row_min"]
        x2 = pos["column_max"]
        y2 = pos["row_max"]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw ID background box (for readability)
        text = str(comp_id)
        text_bbox = draw.textbbox((x1, y1), text, font=font)

        tx1, ty1, tx2, ty2 = text_bbox

        draw.rectangle([tx1, ty1, tx2, ty2], fill="red")

        # Draw text
        draw.text((x1, y1), text, fill="white", font=font)

    return image


# -----------------------------
# MAIN TEST
# -----------------------------
def main():

    elements, (w, h) = run_omni(IMAGE_PATH)

    print(f"Detected elements: {len(elements)}")

    json_data = convert_to_json(elements, w, h)

    # save json
    json_path = os.path.join(OUTPUT_DIR, "omni.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f" JSON saved: {json_path}")

    # reconstruct image
    reconstructed = draw_from_json(IMAGE_PATH, json_data)
    reconstructed.save(os.path.join(OUTPUT_DIR, "reconstructed.png"))

    print(" Reconstructed image saved")

    print("\n Compare:")
    print("- omni_vis.png (original Omni output)")
    print("- reconstructed.png (from JSON)")


if __name__ == "__main__":
    main()