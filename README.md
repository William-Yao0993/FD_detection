# Pretrained Stomata traits network in YOLOv8 structure

This repository contains models and weights of detecting and segmenting Stomata or its aperture in Canola based on [YOLOv8](https://docs.ultralytics.com/) network structure. 
These models are embedded into an GUI for Canola trait statistical analysis. The full detail is listed in this application called [SCAN](https://github.com/William-Yao0993/SCAN). 




## Dataset

There are two datasets used in training classes: ***Stomata*** and ***pore*** in Canola. The datasets are hosted in Roboflow. 

| name | area | Train | Validation | link|
|------|------|-------|------------|-----|
| Canola Leave Surface | Adaxial | 504 | 123 | [FD Project 1](https://app.roboflow.com/danila-lab/fd-project-1)|
|                      | Abaxial | 241 | 49 |[FD Project 1](https://app.roboflow.com/danila-lab/fd-project-1)|
| Extracted Canola Stoma| Adaxial **AND** Abaxial | 404 | 136 |[Pore Segmentation](https://app.roboflow.com/danila-lab/pore-segmentation)

## Models

The section contains a series of YOLOv8 models that have been trained and fine-tuned  for different task in Canola trait measurement. The model name with `_bt` suffix is the model contains the best performance after different fine-tuning strategies on the current task and target.   
| name| task |target | template | date| format | training size |
|-----|------|-------|----------|-----|--------|--------------|
|[detect_2023_12_14](models/detect_2023_12_14.pt)|detect|Stomata|yolov8n.pt|2023-12-14|pytorch| 640x480|
|[detect_tuned_2024_01_05](models/detect_tuned_2024_01_05.pt)|detect|Stomata|yolov8s.pt|2023-01-05|pytorch| 640x480|
|[detect_bt_2024_01_10](models/detect_bt_2024_01_10.pt)|detect|stomata|yolov8s.pt|2023-01-10|pytorch| 640x480|
|[seg_2024_01_09](models/seg_2024_01_09.pt)|segment|stomata|yolov8n-seg.pt|2023-01-09|pytorch| 640x480|
|[seg_2024_01_10](models/seg_2024_01_10.pt)|segment|stomata|yolov8n-seg.pt|2023-01-10|pytorch| 640x480|
|[seg_2024_02_07](models/seg_2024_02_07.pt)|segment|stomata|yolov8n-seg.pt|2023-02-07|pytorch| 640x480|
|[seg_2024_02_08](models/seg_2024_02_08.pt)|segment|stomata|yolov8n-seg.pt|2023-02-08|pytorch| 640x480|
|[seg_2024_02_21](models/seg_2024_02_21.pt)|segment|stomata|yolov8n-seg.pt|2023-02-21|pytorch| 640x480|
|[seg_bt_2024_03_01](models/seg_bt_2024_03_01.pt)|segment|Stomata|yolov8n-seg.pt|2023-03-01|pytorch|1024x798|
|[pore_2024_03_06](models/pore_2024_03_06.pt)|segment|pore|yolov8n-seg.pt|2023-03-06|pytorch|640x480|
|[pore_bt_2024_03_25](models/pore_bt_2024_03_25.pt)|segment|pore|yolov8n-seg.pt|2023-03-25|pytorch|108x81|


## Best Parameter Setting 
In this section, we address the best hyperparameters setting in YOLOv8 training process that has been fine-tuned for stomata detection and segmentation.
|name|optimizer|dropout|learning rate| cross-validation|
|----|---------|-------|-------------|-----------------|
|[Stomata Detection](hyp/stomata/detect/best_hyperparameters.yaml)| AdamW| False| 0.001-0.0001|False| 
|[Stomata Segmentation](hyp/stomata/segment/best_hyperparameters.yaml)| AdamW| True| 0.0128-0.01249|True (k=5)| 

## Further Interests
The pretrained Canola models can provide some accurate prediction results 
## Citation

If you use this project in your research or wish to refer to the baseline results, please cite us.

```bibtex
@article {Yao2024.06.12.598768,
	author = {Yao, Lingtian and von Caemmerer, Susanne and Danila, Florence R},
	title = {Automated and high throughput measurement of leaf stomatal traits in canola},
	elocation-id = {2024.06.12.598768},
	year = {2024},
	doi = {10.1101/2024.06.12.598768},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/06/14/2024.06.12.598768},
	eprint = {https://www.biorxiv.org/content/early/2024/06/14/2024.06.12.598768.full.pdf},
	journal = {bioRxiv}
}

```

## References
1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
2. Dwyer, B., Nelson, J., Hansen, T., et. al. (2024). Roboflow (Version 1.0) [Software]. Available from https://roboflow.com. computer vision.