from ultralytics import YOLO

def train_ppe_detection():
 
    model = YOLO("yolov8m.pt") 
  
    results = model.train(
        data="/home/raghav/Documents/yolov8-obj/datasets/cropped_ppe_dataset/ppe_dataset.yaml",
        epochs=300,              
        patience=0,            
        batch=8,               
        imgsz=150,             
        lr0=0.01,              
    )

if __name__ == "__main__":
    train_ppe_detection()