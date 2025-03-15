# YOLOV8 - Person and PPE Detection

This project involves using the yolov8 models to detect persons and PPE kits for construction workers. 

The models used for the person detection and PPE detection were the yolov8n and yolov8m respectively.

Results can be viewed in the `` person_detection_results `` folder and the `` ppe_detection_results `` folder.

For the combined working and inference, kindly check the image results in the ``inference_results`` folder.

The ``best.pt`` models for both categories are located in:

 Person: ``person_detection_results/weights``

 PPE: ``ppe_detection_results/weights``

A detailed PDF report is also available for the making and working of the project, titled ``Report_YOLOv8OBJ.pdf``.

Execution of files in order:

    1. ``pascalVOC_to_yolo.py``
    2. ``split_data.py``
    3. ``train_person_model.py`` 
    4. ``crop_person.py``
    5. ``adjust_ppe_annotations.py``
    6. ``split_data.py``
    7. ``train_ppe_model.py``
    8. ``inference.py``


To run inference: 

```bash
$ python inference.py   
$ --input_dir /absolute/input/directory/path   
$ --output_dir  /absolute/output/directory/path
$ --person_det_model /absolute/model.pt/path
$ --ppe_detection_model /absolute/model.pt/path
```