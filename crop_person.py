import cv2
import os
import json
from ultralytics import YOLO

def crop_person_images(image_dir, output_dir, person_model_weights):
    person_model = YOLO(person_model_weights)

    cropped_images_dir = os.path.join(output_dir, "images")
    cropped_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(cropped_images_dir, exist_ok=True)
    os.makedirs(cropped_labels_dir, exist_ok=True)

    crop_mapping = {}

    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            original_height, original_width = image.shape[:2]  

            
            results = person_model(image)
            person_boxes = results[0].boxes.xyxy.cpu().numpy()

            for i, box in enumerate(person_boxes):
                xmin, ymin, xmax, ymax = map(int, box)
                cropped_image = image[ymin:ymax, xmin:xmax]

                base_name = os.path.splitext(image_file)[0]
                crop_id = f"{base_name}_person_{i}"
                crop_image_path = os.path.join(cropped_images_dir, f"{crop_id}.jpg")
                cv2.imwrite(crop_image_path, cropped_image)

                crop_mapping[crop_id] = {
                    "original_image": image_file,
                    "original_width": original_width, 
                    "original_height": original_height,  
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "crop_width": xmax - xmin,
                    "crop_height": ymax - ymin
                }

    with open(os.path.join(output_dir, "crop_mapping.json"), 'w') as f:
        json.dump(crop_mapping, f, indent=4)

crop_person_images(
    image_dir="datasets/images",
    output_dir="datasets/cropped_ppe_dataset",
    person_model_weights="detect/person_detection3/weights/best.pt"
)