# code A. Hoenen, 2025, CC BY-NC 4.0 https://creativecommons.org/licenses/by-nc/4.0/

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# Set up directories
input_dir = "dataset/graphs/images/val"
output_img_dir = "dataset/synthetic_graphs_on_synthetic_pages/images/val"
output_lbl_dir = "dataset/synthetic_graphs_on_synthetic_pages/labels/val"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

def create_random_text_background_with_margin(full_width=640, full_height=640, margin_ratio=0.15, line_count=20):
    # Create full white canvas
    full_img = Image.new('RGB', (full_width, full_height), color='white')

    # Compute inner content area (90% of full size)
    margin_x = int(full_width * margin_ratio)
    margin_y = int(full_height * margin_ratio)
    inner_width = full_width - 2 * margin_x
    inner_height = full_height - 2 * margin_y

    # Create inner text content area
    inner_img = Image.new('RGB', (inner_width, inner_height), color='white')
    draw = ImageDraw.Draw(inner_img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i in range(line_count):
        text = ''.join(random.choices(" abcdefghijklmnopqrstuvwxyz", k=random.randint(30, 70)))
        draw.text((10, i * 30), text, font=font, fill='black')

    # Paste inner content into white canvas
    full_img.paste(inner_img, (margin_x, margin_y))

    return np.array(full_img), margin_x, margin_y, inner_width, inner_height
# Generate synthetic background (fake page with text)
def create_random_text_background(width=640, height=640, line_count=20):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i in range(line_count):
        text = ''.join(random.choices(" abcdefghijklmnopqrstuvwxyz", k=random.randint(30, 70)))
        draw.text((10, 30 * i), text, font=font, fill='black')

    return np.array(img)

# Convert to YOLO format
def convert_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return [x_center, y_center, w / img_w, h / img_h]

label = 0
N_AUG = 3

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        obj_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if obj_img is None:
            continue

        for aug_idx in range(N_AUG):
            # Create background
            bg_img, margin_x, margin_y, content_w, content_h = create_random_text_background_with_margin()
            bg_h, bg_w, _ = bg_img.shape

            # Resize object image to 100–300 px width
            # Resize object image
            scale = random.uniform(0.2, 0.4)
            new_w = int(bg_w * scale)
            aspect_ratio = obj_img.shape[0] / obj_img.shape[1]
            new_h = int(new_w * aspect_ratio)
            resized_obj = cv2.resize(obj_img, (new_w, new_h))

            # Placement range inside content area (respecting margin)
            max_x = margin_x + content_w - new_w
            min_x = margin_x

            max_y = margin_y + content_h - new_h
            min_y = margin_y

            if max_x < min_x or max_y < min_y:
                print(f"⚠️ Skipped {filename} at augmentation {aug_idx+1}: object too large for margins.")
                continue

            x_offset = random.randint(min_x, max_x)
            y_offset = random.randint(min_y, max_y)

            # Paste object onto background
            # If object has 4 channels, use alpha blending
            if resized_obj.shape[2] == 4:
                alpha = resized_obj[:, :, 3] / 255.0  # normalize alpha to 0–1
                for c in range(3):  # for R, G, B channels
                    bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                        alpha * resized_obj[:, :, c] +
                        (1 - alpha) * bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                    )
            else:
                # No alpha channel, paste directly
                bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_obj

            # Convert to YOLO format
            yolo_bbox = convert_to_yolo(x_offset, y_offset, new_w, new_h, bg_w, bg_h)

            # Save image
            out_name = f"{base_name}_comp{aug_idx+1}"
            out_img_path = os.path.join(output_img_dir, out_name + ".png")
            out_lbl_path = os.path.join(output_lbl_dir, out_name + ".txt")
            cv2.imwrite(out_img_path, bg_img)

            # Save label
            with open(out_lbl_path, "w") as f:
                f.write(f"{label} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

print("✅ Synthetic images with random text backgrounds created.")
