from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn
import os
import json
import base64

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img
)

app = FastAPI()

# --- LOAD MODELS ---
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')

caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)


# --- SAVE JSON ---
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


@app.post("/parse")
async def parse_ui(file: UploadFile = File(...)):

    # --- READ IMAGE ---
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

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

    # --- STEP 1: OCR ---
    ocr_bbox_rslt, _ = check_ocr_box(
        image,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr
    )

    text, ocr_bbox = ocr_bbox_rslt

    # --- STEP 2: OMNIPARSER ---
    dino_labeled_img, _, parsed_content_list = get_som_labeled_img(
        image,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,   # IMPORTANT
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz
    )

    # --- DECODE VISUAL IMAGE ---
    vis_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))

    # --- CLEAN + CONVERT JSON ---
    clean_components = []

    for item in parsed_content_list:

        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        cx, cy, w, h = bbox

        # --- CONVERT NORMALIZED → PIXELS ---
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        # --- CLAMP TO IMAGE BOUNDS (CRITICAL) ---
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        # --- VALIDATION ---
        if x2 <= x1 or y2 <= y1:
            continue

        # --- OPTIONAL: REMOVE TINY NOISE ---
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        comp = {
            "id": len(clean_components),
            "class": "Compo",
            "height": y2 - y1,
            "width": x2 - x1,
            "position": {
                "column_min": x1,
                "row_min": y1,
                "column_max": x2,
                "row_max": y2
            },
            "text": item.get("content", ""),
            "label": item.get("type", "")
        }

        clean_components.append(comp)

    # --- SAVE OUTPUT ---
    output_dir = "outputs/test_case_1"
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "omni.json")
    img_path = os.path.join(output_dir, "omni_vis.jpg")

    save_json({"components": clean_components}, json_path)
    vis_image.save(img_path)

    print(f"✅ Saved JSON: {json_path}")
    print(f"✅ Saved Image: {img_path}")
    print(f"📦 Clean Components: {len(clean_components)}")

    return {"components": clean_components}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)