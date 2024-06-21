#!/usr/bin/env python
# coding: utf-8

# # YOLOv8 Stamata Idenification
import comet_ml
import datetime
experiment = comet_ml.Experiment(
    project_name= "FD_detection_train"
)
experiment.set_name(f'det_train_{datetime.datetime.now().isoformat()}')
import os
os.environ["OMP_NUM_THREADS"] = '8'
import ultralytics
ultralytics.checks()

#from roboflow import Roboflow
# rf = Roboflow(api_key="nWVwAVDNDqpsgtR5bTD0")
# project = rf.workspace("danila-lab").project("fd-project-1")
# dataset = project.version(4).download("yolov8")


# Keep track of performance of the model
# comet_ml.config.save(api_key = 'Qn88wPyWB3PmP0hTUAwhg1h9u')
from ultralytics import YOLO
print("Experiment Start:", datetime.datetime.now())

# Load & Train a model
SERVER_PATH = "/mnt/data/dayhoff/home/u6771897/FD_detection/datasets/FD-Project-1-4/data_dayhoff.yaml"
LOCAL_PATH = f'datasets\FD-Project-1-4\data.yaml'
model_path = 'models/detect_tuned_2024_01_05.pt'
hyp_path = 'hyp/detect/best_hyperparameters.yaml'
model = YOLO(model_path, task='detect')
result = model.train(data = SERVER_PATH, 
                     epochs = 999, 
                     batch = 32,
                     patience = 100,
                     cfg = hyp_path,
                     save = True,
                     )
print("Experiment End: ", datetime.datetime.now())