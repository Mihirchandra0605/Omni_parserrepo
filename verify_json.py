import json
import cv2
import os


def draw_from_json(json_path, image_path, output_path):

    # --- LOAD JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    components = data["components"]

    # --- LOAD IMAGE ---
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found:", image_path)
        return

    # --- DRAW BOXES ---
    for comp in components:

        pos = comp["position"]

        x1 = pos["column_min"]
        y1 = pos["row_min"]
        x2 = pos["column_max"]
        y2 = pos["row_max"]

        label = comp.get("label", "")
        text = comp.get("text", "")

        # --- DRAW RECTANGLE ---
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- DRAW LABEL ---
        display_text = label if label else text[:10]

        cv2.putText(
            image,
            display_text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # --- SAVE OUTPUT ---
    cv2.imwrite(output_path, image)

    print("✅ Visualization saved at:", output_path)
    print("📦 Components drawn:", len(components))


# -----------------------------
# 🔧 CHANGE PATHS HERE
# -----------------------------
json_path = "outputs/test_case_1/omni.json"
image_path = "test_images/test1.png"   # original input image
output_path = "outputs/test_case_1/json_visualization.jpg"

draw_from_json(json_path, image_path, output_path)