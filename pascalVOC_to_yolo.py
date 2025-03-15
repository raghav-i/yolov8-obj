import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

def convert_annotation(xml_file, output_dir, classes):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            print(f"Warning: No size information in {xml_file}. Skipping and deleting.")
            return True 

        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)


        objects = root.findall('object')
        if not objects:
            print(f"Warning: No objects found in {xml_file}. Skipping and deleting.")
            return True

        for obj in objects:
            class_name = obj.find('name').text
            if class_name not in classes:
                print(f"Warning: Class '{class_name}' not in classes list. Skipping object in {xml_file}.")
                continue

            class_id = classes.index(class_name)
            bbox = obj.find('bndbox')
            if bbox is None:
                print(f"Warning: No bounding box found for object in {xml_file}. Skipping object.")
                continue

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
            with open(output_file, 'a') as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        return False  
    except ET.ParseError as e:
        print(f"Error: Invalid XML file {xml_file}. Skipping and deleting. Error: {e}")
        return True 
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {xml_file}. Skipping and deleting. Error: {e}")
        return True  

def delete_corresponding_image(xml_file, image_dir):
    base_name = os.path.splitext(os.path.basename(xml_file))[0]

    for ext in ['.jpg', '.png', '.jpeg']:
        image_file = os.path.join(image_dir, base_name + ext)
        if os.path.exists(image_file):
            os.remove(image_file)
            print(f"Deleted corresponding image file: {image_file}")

def main():
    parser = ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format")
    parser.add_argument("input_dir", help="Directory containing PascalVOC XML files")
    parser.add_argument("output_dir", help="Directory to save YOLOv8 annotations")
    parser.add_argument("image_dir", help="Directory containing corresponding images")
    args = parser.parse_args()

    classes = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    xml_files = [f for f in os.listdir(args.input_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} XML files in {args.input_dir}.")

    for xml_file in xml_files:
        xml_path = os.path.join(args.input_dir, xml_file)
        should_delete = convert_annotation(xml_path, args.output_dir, classes)

        if should_delete:
            os.remove(xml_path)
            print(f"Deleted invalid XML file: {xml_path}")
            delete_corresponding_image(xml_path, args.image_dir)

    txt_files = [f for f in os.listdir(args.output_dir) if f.endswith('.txt')]
    print(f"Generated {len(txt_files)} TXT files in {args.output_dir}.")

if __name__ == "__main__":
    main()