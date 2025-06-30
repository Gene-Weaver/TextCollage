# inference.py

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json

def get_bounding_boxes(model_path, image_path):
    """
    Performs object detection on an image using a CPU.
    """
    model = YOLO(model_path)
    results = model(image_path, device='cpu')

    if not results or not results[0].boxes:
        return {}

    boxes = results[0].boxes.xyxy.tolist()
    class_ids = results[0].boxes.cls.tolist()
    class_names_map = results[0].names

    bounding_boxes = {name: [] for name in class_names_map.values()}
    for box, class_id in zip(boxes, class_ids):
        class_name = class_names_map[int(class_id)]
        bounding_boxes[class_name].append(box)

    return bounding_boxes

def merge_overlapping_boxes(boxes):
    """
    Merges overlapping bounding boxes into single encompassing boxes.
    """
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=int)
    while True:
        merged_in_pass = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                box1, box2 = boxes[i], boxes[j]
                if (box1[0] < box2[2] and box1[2] > box2[0] and
                    box1[1] < box2[3] and box1[3] > box2[1]):
                    merged_box = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
                    boxes[i] = merged_box
                    boxes = np.delete(boxes, j, axis=0)
                    merged_in_pass = True
                    break
                else:
                    j += 1
            if merged_in_pass:
                break
            else:
                i += 1
        if not merged_in_pass:
            break
    return boxes.tolist()

def create_condensed_collage(detected_objects, original_image, class_names_to_include, all_class_names):
    """
    Creates a collage by condensing objects and returns the image and new coordinates for all possible classes.
    """
    collage_positions = {name: [] for name in all_class_names}
    crops = []
    for class_name in class_names_to_include:
        if class_name in detected_objects and detected_objects[class_name]:
            for box in detected_objects[class_name]:
                x1, y1, x2, y2 = map(int, box)
                crops.append({'img': original_image[y1:y2, x1:x2], 'orig_box': (x1, y1, x2, y2), 'class_name': class_name})

    if not crops:
        return None, collage_positions

    rows = []
    crops.sort(key=lambda c: c['orig_box'][1])
    for crop in crops:
        y1, y2 = crop['orig_box'][1], crop['orig_box'][3]
        placed = False
        for row in rows:
            if max(row['y_min'], y1) < min(row['y_max'], y2):
                row['crops'].append(crop); row['y_min'] = min(row['y_min'], y1); row['y_max'] = max(row['y_max'], y2); placed = True; break
        if not placed:
            rows.append({'crops': [crop], 'y_min': y1, 'y_max': y2})

    row_images, max_width = [], 0
    current_y = 0
    for row in rows:
        row['crops'].sort(key=lambda c: c['orig_box'][0])
        row_h = max(c['img'].shape[0] for c in row['crops'])
        row_w = sum(c['img'].shape[1] for c in row['crops'])
        current_x = 0
        for c in row['crops']:
            h, w = c['img'].shape[:2]
            new_box = [current_x, current_y, current_x + w, current_y + h]
            collage_positions[c['class_name']].append(new_box)
            current_x += w
        row_images.append({'w': row_w, 'h': row_h})
        max_width = max(max_width, row_w)
        current_y += row_h

    total_height = sum(r['h'] for r in row_images)
    final_collage = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    
    y_offset = 0
    for i, row in enumerate(rows):
        x_offset = 0
        for crop_data in row['crops']:
            h, w = crop_data['img'].shape[:2]
            final_collage[y_offset:y_offset+h, x_offset:x_offset+w] = crop_data['img']
            x_offset += w
        y_offset += row_images[i]['h']

    return final_collage, collage_positions

def resize_image_for_engine(image, engine):
    """
    Resizes an image according to predefined rules and returns the image and scale factor.
    """
    scale = 1.0
    h, w = image.shape[:2]
    print(f"Applying resizing rule for engine: '{engine}'...")
    print(f"  - Original dimensions (HxW): {h}x{w}")

    if engine == 'gemini':
        print("  - Gemini rule: No resize needed."); return image, scale
    
    elif engine == 'claude':
        print("  - Claude rule: Longest edge to 1568px.")
        longest_side = max(h, w)
        if longest_side > 1568:
            scale = 1568 / longest_side
            resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            print(f"  - Resized to (HxW): {resized.shape[0]}x{resized.shape[1]}"); return resized, scale
        print("  - Image already compliant, no resize needed."); return image, scale

    elif engine == 'gpt':
        print("  - GPT rule: Longest to 2048, then shortest to 768.")
        longest_side = max(h, w)
        if longest_side > 2048:
            scale = 2048 / longest_side; image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]; print(f"    - Intermediate dimensions (HxW): {h}x{w}")
        else: print("    - Step 1: Longest side compliant, no resize.")
        
        shortest_side = min(h, w)
        if shortest_side > 768:
            scale_step2 = 768 / shortest_side; image = cv2.resize(image, (int(w * scale_step2), int(h * scale_step2)), interpolation=cv2.INTER_AREA)
            scale *= scale_step2
            print(f"  - Final dimensions (HxW): {image.shape[0]}x{image.shape[1]}")
        else: print("    - Step 2: Shortest side compliant, no resize.")
        return image, scale
    
    return image, scale

def draw_overlay_on_collage(image, positions):
    """Draws labeled bounding boxes on the collage image for verification."""
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    color_map = {
        'ruler': (255, 0, 0), 'barcode': (0, 255, 0), 'colorcard': (0, 0, 255),
        'label': (255, 255, 0), 'map': (255, 0, 255), 'envelope': (0, 255, 255),
        'photo': (255, 255, 255), 'attached_item': (0, 165, 255), 'weights': (128, 0, 128)
    }
    
    for class_name, bboxes in positions.items():
        if not bboxes: continue
        color = color_map.get(class_name, (0, 0, 0))
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_name}: [{x1},{y1},{x2},{y2}]"
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(overlay, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            text_color = (0, 0, 0) if class_name in ['envelope', 'label', 'photo'] else (255, 255, 255)
            cv2.putText(overlay, label, (x1, y1 - 5), font, font_scale, text_color, 1)
            
    return overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLOv12 inference and create a condensed collage.')
    parser.add_argument('--model', type=str, default='runs/detect/yolo12n/weights/best.pt', help='Path to the trained .pt model.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image for inference.')
    parser.add_argument('--output', type=str, default='collage_output.jpg', help='Path to save the final collage image.')
    parser.add_argument('--engine', type=str, default='gemini', choices=['gemini', 'claude', 'gpt', 'original'], help='The AI engine to resize the output for.')
    parser.add_argument('--overlay', action='store_true', help='If set, draws labeled bounding boxes on the final collage for verification.')
    parser.add_argument('--hide-long-objects', action='store_true', help='If set, excludes objects with an aspect ratio (w/h) > 2.0 from the collage.')
    args = parser.parse_args()
    
    all_possible_classes = ['ruler', 'barcode', 'colorcard', 'label', 'map', 'envelope', 'photo', 'attached_item', 'weights']
    final_output = {"position_original": {}, "position_collage": {}}

    print(f"Running inference on {args.image}...")
    detected_objects = get_bounding_boxes(args.model, args.image)
    if not detected_objects: print("No objects were detected in the image."); exit()

    print("\n--- Merging Overlapping Boxes ---")
    merged_objects = {}
    for class_name, bboxes in detected_objects.items():
        if bboxes:
            merged_bboxes = merge_overlapping_boxes(bboxes)
            print(f"  Class: '{class_name}' -> Original: {len(bboxes)}, Merged: {len(merged_bboxes)}")
            merged_objects[class_name] = merged_bboxes
    
    final_output["position_original"] = {k: v for k, v in merged_objects.items() if v}
    
    # --- Aspect Ratio Filtering ---
    objects_for_collage = {}
    if args.hide_long_objects:
        print("\n--- Filtering Long Objects (Aspect Ratio < 0.5) ---")
        for class_name, bboxes in merged_objects.items():
            if not bboxes: continue
            kept_boxes = []
            for i, box in enumerate(bboxes):
                width = box[2] - box[0]
                height = box[3] - box[1]
                ratio = width / height if height > 0 else float('inf')
                if (ratio > 0.25) and (ratio < 6):
                    kept_boxes.append(box)
                else:
                    print(f"  - Discarding '{class_name}' box {i+1} (ratio: {ratio:.2f})")
            if kept_boxes:
                objects_for_collage[class_name] = kept_boxes
    else:
        objects_for_collage = merged_objects
    # --- End of Filtering ---

    print("\n--- Bounding Box Aspect Ratios (w/h) of objects FOR COLLAGE ---")
    for class_name, bboxes in objects_for_collage.items():
        if bboxes:
            for i, box in enumerate(bboxes):
                width = box[2] - box[0]
                height = box[3] - box[1]
                ratio = width / height if height > 0 else float('inf')
                print(f"  Class '{class_name}', Box {i+1}: {ratio:.2f}")

    classes_for_collage = ['barcode', 'label', 'map']
    classes_for_collage = None
    if classes_for_collage is None:
        classes_for_collage = all_possible_classes
    
    original_image = cv2.imread(args.image)
    if original_image is None: print(f"Error: Could not load image at {args.image}"); exit()

    print(f"\n--- Creating Condensed Collage for: {classes_for_collage} ---")
    collage_image, pre_resize_positions = create_condensed_collage(objects_for_collage, original_image, classes_for_collage, all_possible_classes)
    
    if collage_image is not None:
        resized_collage, scale = resize_image_for_engine(collage_image, args.engine)
        
        final_collage_positions = {name: [] for name in all_possible_classes}
        for class_name, bboxes in pre_resize_positions.items():
            if bboxes:
                for box in bboxes:
                    scaled_box = [int(coord * scale) for coord in box]
                    final_collage_positions[class_name].append(scaled_box)
        final_output["position_collage"] = final_collage_positions
        
        if args.overlay:
            print("\n--- Drawing Overlay on Final Collage ---")
            resized_collage = draw_overlay_on_collage(resized_collage, final_collage_positions)
        
        cv2.imwrite(args.output, resized_collage)
        print(f"\nSuccessfully created and saved condensed collage to '{args.output}'")
    else:
        print("Condensed collage could not be created.")

    print("\n--- Final Bounding Box Data (JSON) ---")
    print(json.dumps(final_output, indent=2))