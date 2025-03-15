#can use this for regular and cropped dataset

import os
import random
import shutil

def split_dataset(image_dir, annotation_dir, train_ratio=0.8):

    train_image_dir = os.path.join("datasets/cropped_ppe_dataset/train/images")
    train_label_dir = os.path.join("datasets/cropped_ppe_dataset/train/labels")
    val_image_dir = os.path.join("datasets/cropped_ppe_dataset/val/images")
    val_label_dir = os.path.join("datasets/cropped_ppe_dataset/val/labels")

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)  

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for file in train_files:
        shutil.move(os.path.join(image_dir, file), os.path.join(train_image_dir, file))
        annotation_file = os.path.splitext(file)[0] + '.txt'
        shutil.move(os.path.join(annotation_dir, annotation_file), os.path.join(train_label_dir, annotation_file))

    for file in val_files:
        shutil.move(os.path.join(image_dir, file), os.path.join(val_image_dir, file))
        annotation_file = os.path.splitext(file)[0] + '.txt'
        shutil.move(os.path.join(annotation_dir, annotation_file), os.path.join(val_label_dir, annotation_file))

    print(f"Dataset split into {len(train_files)} training samples and {len(val_files)} validation samples.")

split_dataset("datasets/cropped_ppe_dataset/images", "datasets/cropped_ppe_dataset/labels", train_ratio=0.8)