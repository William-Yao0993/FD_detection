import datetime
import comet_ml
experiment = comet_ml.Experiment(
    project_name= 'FD_pore_Segmentation', 
)
experiment.set_name(f'train{datetime.datetime.now().date().strftime("%Y%m%d")}')
import os
from ultralytics import YOLO
from pathlib import Path
def main():
    SERVER_PATH = "/mnt/data/dayhoff/home/u6771897/FD_detection/datasets/Pore-Segmentation-1"
    SERVER_YAML = os.path.join(SERVER_PATH, 'data_dayhoff.yaml')
    weights = 'models/seg_2024_03_01.pt'
    model = YOLO('yolov8n-seg.yaml').load(weights)
    model.train(
        data= SERVER_YAML,
        epochs= 9999,
        patience= 100,
        batch=32,
        #imgsz= 1024, # imgsz = Int | (h,w)
        rect= False,
        workers= 12,
        project= os.path.join('runs/segment', f'pore_train_{datetime.datetime.today().isoformat()}'),
        val= True,
        save= True,
        plots= True,
        device= [0,1]
    )

if __name__ == '__main__':
    # os.environ['OMP_NUM_THREADS'] = '28'
    os.environ['NCCL_P2P_DISABLE'] = '1'   
    main()