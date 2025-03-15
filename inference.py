import cv2
import os
import argparse
from ultralytics import YOLO

def draw_boxes(image, detections, class_colors):
    for class_name, conf, (x1, y1, x2, y2) in detections:
        color = class_colors.get(class_name, (0, 255, 0))  
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def run_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)
    
    class_colors = {
        "person": (0, 255, 0),          # Green
        "hard-hat": (255, 0, 0),        # Blue
        "gloves": (0, 0, 255),          # Red
        "mask": (255, 255, 0),          # Cyan
        "glasses": (255, 0, 255),       # Magenta
        "boots": (0, 255, 255),         # Yellow
        "vest": (128, 0, 128),          # Purple
        "ppe-suit": (0, 128, 128),      # Teal
        "ear-protector": (128, 128, 0), # Olive
        "safety-harness": (255, 165, 0) # Orange
    }
    
    ppe_classes = ["safety-harness", "hard-hat", "gloves", "glasses", "mask", "boots" , "vest", "ppe-suit", "ear-protector"]

    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        all_detections = []
        h, w = image.shape[:2]
        
        person_results = person_model(image)[0]
        for person_box in person_results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, person_box)
            all_detections.append(("person", 1.0, (x1, y1, x2, y2)))
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            ppe_results = ppe_model(crop)[0]
            for ppe_box, conf, cls_id in zip(ppe_results.boxes.xyxy.cpu().numpy(),
                                            ppe_results.boxes.conf.cpu().numpy(),
                                            ppe_results.boxes.cls.cpu().numpy()):
                px1, py1, px2, py2 = map(int, ppe_box)
                orig_coords = (
                    x1 + px1,
                    y1 + py1,
                    x1 + px2,
                    y1 + py2
                )
                class_name = ppe_classes[int(cls_id)]
                all_detections.append((class_name, conf, orig_coords))
        
        output_img = image.copy()
        draw_boxes(output_img, all_detections, class_colors)
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, output_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Detection Inference Pipeline")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save annotated output images")
    parser.add_argument("--person_det_model", type=str, required=True,
                       help="Path to person detection model weights")
    parser.add_argument("--ppe_detection_model", type=str, required=True,
                       help="Path to PPE detection model weights")
    
    args = parser.parse_args()
    
    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        person_model_path=args.person_det_model,
        ppe_model_path=args.ppe_detection_model
    )

#use it like this on cli
#python inference.py   --input_dir /absolute/input/directory/path   
# --output_dir  /absolute/output/directory/path
#  --person_det_model /absolute/model.pt/path
#  --ppe_detection_model /absolute/model.pt/path