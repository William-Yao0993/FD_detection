import comet_ml
import datetime
experiment = comet_ml.Experiment(
    project_name="FD_Seg_Tuning"
)
experiment.set_name(f'tuning{datetime.datetime.now().strftime("%Y%m%d%H")}')
import os 

os.environ['OMP_NUM_THREADS'] = '8'

# import sys
# print(sys.path)
import ultralytics
ultralytics.checks()
from ultralytics import YOLO


# Fine-Tuning 
SERVER_PATH = "/mnt/data/dayhoff/home/u6771897/FD_detection/datasets/FD-Project-1-5/data_dayhoff.yaml"
LOCAL_PATH = f'datasets\FD-Project-1-5\data.yaml'
model_path = r'models/seg_2024_02_21.pt'
hyp_path = r'hyp/segment/best_hyperparameters.yaml'
model = YOLO(model= model_path,task= 'segment')
model.tune(data = SERVER_PATH,
           imgsz = 1024,
           batch = 16,
           patience= 30,
           epochs = 100,
           iterations= 300,
           optimizer= 'AdamW',
           plots= False,
           save = False,
           val = True,
           dropout = 0.1,
           device = 0,
           project= os.path.join('runs/segment', f'tune_{datetime.datetime.today().isoformat()}'),
           )

