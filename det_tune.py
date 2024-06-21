# This is the file to fine-tuning the detection model
# The hyperparameters will be recorded for further usage
import comet_ml
import datetime
experiment = comet_ml.Experiment(
    project_name='FD_stamota_dectection'
)
experiment.set_name(f"tuning{datetime.datetime.now().strftime('%Y%m%d')}")

import numpy as np
import pandas as pd 
from ultralytics import YOLO 

model_path = 'models/detect_2023_12_14.pt'
model = YOLO(model_path)
SERVER_PATH = "/mnt/data/dayhoff/home/u6771897/FD_detection/datasets/FD-Project-1-4/data_dayhoff.yaml"
LOCAL_PATH = f'datasets\FD-Project-1-4\data.yaml'
model.tune(data = SERVER_PATH,
           epochs = 10,
           iterations= 100,
           optimizer= 'AdamW',
           plots= False,
           save = False,
           val = False,
           lr0 = 1E-3,
           lrf = 1E-6,
           )
