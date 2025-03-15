from ultralytics import YOLO

def train_person_detection():

    model = YOLO("yolov8n.pt") 

    results = model.train(
        data="datasets/dataset.yaml",  
        epochs=50,                     
        imgsz=640,                    
        batch=8,                       
        name="person_detection",       
        single_cls=True               
    )

if __name__ == "__main__":
    train_person_detection()