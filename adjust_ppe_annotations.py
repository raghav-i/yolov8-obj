import os
import json


def adjust_ppe_annotations(original_annotations_dir, crop_mapping_path, output_dir):
    with open(crop_mapping_path, 'r') as f:
        crop_mapping = json.load(f)


    for crop_id, metadata in crop_mapping.items():
        original_image = metadata["original_image"]
        original_annotation_path = os.path.join(original_annotations_dir, f"{os.path.splitext(original_image)[0]}.txt")

        if not os.path.exists(original_annotation_path):
            continue

        with open(original_annotation_path, 'r') as f:
            original_lines = f.readlines()

        adjusted_lines = []

        original_width = metadata["original_width"]  
        original_height = metadata["original_height"]  
        xmin = metadata["xmin"]
        ymin = metadata["ymin"]
        crop_width = metadata["crop_width"]
        crop_height = metadata["crop_height"]

     
        for line in original_lines:
            class_id, x_center, y_center, w, h = map(float, line.split())

            abs_x_center = x_center * original_width
            abs_y_center = y_center * original_height
            abs_w = w * original_width
            abs_h = h * original_height

            box_xmin = abs_x_center - (abs_w / 2)
            box_ymin = abs_y_center - (abs_h / 2)
            box_xmax = abs_x_center + (abs_w / 2)
            box_ymax = abs_y_center + (abs_h / 2)

            if (box_xmin >= xmin and box_xmax <= metadata["xmax"] and
                box_ymin >= ymin and box_ymax <= metadata["ymax"]):
                
            
                new_xmin = box_xmin - xmin
                new_ymin = box_ymin - ymin
                new_xmax = box_xmax - xmin
                new_ymax = box_ymax - ymin

   
                new_x_center = (new_xmin + new_xmax) / 2 / crop_width
                new_y_center = (new_ymin + new_ymax) / 2 / crop_height
                new_w = (new_xmax - new_xmin) / crop_width
                new_h = (new_ymax - new_ymin) / crop_height

                adjusted_lines.append(f"{int(class_id)} {new_x_center} {new_y_center} {new_w} {new_h}\n")

        output_path = os.path.join(output_dir, "labels", f"{crop_id}.txt")
        with open(output_path, 'w') as f:
            f.writelines(adjusted_lines)

adjust_ppe_annotations(
    original_annotations_dir="datasets/annotations",
    crop_mapping_path="datasets/cropped_ppe_dataset/crop_mapping.json",
    output_dir="datasets/cropped_ppe_dataset"
)