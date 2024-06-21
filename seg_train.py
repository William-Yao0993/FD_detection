#!/usr/bin/env python
# coding: utf-8

# This is a Segmentation model that 
# applies K-Fold Cross Validation before training stage
# K-Fold strategy give more variety in a small dataset 
# to gain better model 
import datetime
import comet_ml
experiment = comet_ml.Experiment(
    project_name= 'FD_stamota_Segmentation', 
)
experiment.set_name(f'trial{datetime.datetime.now().date().strftime("%Y%m%d")}')

#os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import shutil 
from pathlib import Path 
from collections import Counter
import yaml
import numpy as np
import pandas as pd 
#from sklearn.model_selection import KFold
from ultralytics import YOLO
import torch
import os
# # Extend Mutil-Processes Timeout in torch CUDAs 
# os.environ['NCCL_BLOCKING_WAIT'] ='0' # not to enforce timeout
# dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo',
#                         init_method='env://',
#                         timeout=datetime.timedelta(minutes=120),
#                         rank = torch.cuda.device_count(),
#                         world_size=1)

def main():
    SERVER_PATH = "/mnt/data/dayhoff/home/u6771897/FD_detection/datasets/FD-Project-1-7"
    SERVER_YAML = os.path.join(SERVER_PATH, 'data_dayhoff.yaml')
    LOCAL_PATH = 'datasets\FD-Project-1-7'
    LOCAL_YAML = os.path.join(LOCAL_PATH, 'data.yaml')

    model_path = r'models/seg_2024_02_21.pt'
    hyp_path = r'hyp/segment/best_hyperparameters.yaml'
    model = YOLO(Path(model_path), task = 'segment')
    model.train(
        data= SERVER_YAML,
        cfg= hyp_path,
        epochs= 999,
        patience= 30,
        batch=20,
        imgsz= 1024, # imgsz = Int | (h,w)
        rect= False,
        workers= 12,
        project= os.path.join('runs/segment', f'train_{datetime.datetime.today().isoformat()}'),
        val= True,
        save= True,
        plots= True,
        device= [0,1]
    )
# def multiGPUs_setup(rank,world_size):
#         torch.distributed.init_process_group(
#         backend='nccl', init_method='/mnt/data/dayhoff/home/u6771897/FD_detection', 
#         timeout=None, world_size=world_size, rank=rank, 
#         store=None, group_name='', pg_options=None)
#         if torch.cuda.is_available():
#             torch.cuda.set_device(rank % torch.cuda.device_count())
if __name__ == '__main__':
#     print('Avaiable GPU Count: ',torch.cuda.device_count())
#     world_size = 2
#     for rank in range(world_size):
#         multiGPUs_setup(rank,world_size)


    # os.environ['OMP_NUM_THREADS'] = '28'
    os.environ['NCCL_P2P_DISABLE'] = '1'   
    main()

    # torch.multiprocessing.freeze_support()
    # main()

# from roboflow import Roboflow
# rf = Roboflow(api_key="nWVwAVDNDqpsgtR5bTD0")
# project = rf.workspace("danila-lab").project("fd-project-1")
# dataset = project.version(5).download("yolov8")





# # Step1: Generate Summary Table for K-Fold 
# # Extract all txt file except test directory
# dataset_path = Path(SERVER_PATH)
# labels = [file for file in dataset_path.rglob('*labels/*.txt') 
#           if 'train' in file.parts or 
#           'valid' in file.parts]
# labels = sorted(labels)

# # Create a summary table
# # e.g. :  
# #                                                stomata
# # 0013_jpg.rf.d1086e0ebe3e7e29202889249696f998  11
# # 0015_jpg.rf.35dce3fe663c82f7e1d1dc2bdb3b52a9  15
# # 0017_jpg.rf.452fca0f61431d658bc27c2f539ff356  20
# # 0018_jpg.rf.91c233249106af453552ae99e6831571  16
# # 0019_jpg.rf.cbd4b9bc5984e6750c9ddf107ecc3eec  18
# # ...                                           ..
# # 03_jpg.rf.a939958d6bd8bfc889655887a6068916    38
# # 04_jpg.rf.deb489c7a90e0a436abe13d7708ce665    36
# # 05_jpg.rf.5574334b8ad2a0587859a92f68f168da    21
# # 05_jpg.rf.7a3a4559230bf1fe6e8962697c5d1c0e    37
# # 06_jpg.rf.d55f0fb6cf8ef7271442231abeacac33    39
# with open (SERVER_YAML, 'r', encoding= 'utf8') as file:
#     classes = yaml.safe_load(file)['names']
# cls_idx = [i for i in range(len(classes))]

# indx = [label.stem for label in labels]
# labels_df = pd.DataFrame([], columns = cls_idx, index = indx)

# for label in labels: 
#     counter = Counter()
#     with open (label, 'r') as file:
#         lines = file.readlines()
#     for line in lines:
#         counter[int(line.split(' ')[0])] += 1
#     labels_df.loc[label.stem] = counter
# labels_df = labels_df.fillna(0.0)


# # Step 2: Dataset Split
# _k = 10
# kf = KFold(n_splits=_k, shuffle= True, random_state= 425)
# kfolds = list(kf.split(labels_df))

# folds = [f'split_{n}' for n in range (1, _k+1)]
# folds_df = pd.DataFrame(index = indx, columns = folds)

# for i, (train, val) in enumerate(kfolds, start=1):
#     folds_df[f'split_{i}'].loc[labels_df.iloc[train].index] = 'train'
#     folds_df[f'split_{i}'].loc[labels_df.iloc[val].index] = 'val'

# # Step 3: Calculate cls Distribution 
# fold_dist = pd.DataFrame(index = folds, columns = cls_idx)

# for n, (train, val) in enumerate(kfolds, start=1):
#     train_totoal = labels_df.iloc[train].sum()
#     val_total = labels_df.iloc[val].sum()
#     ratio = val_total / (train_totoal + 1E-7)
#     fold_dist.loc[f'split_{n}'] = ratio

# # Step 4: Prepare YAML and dataset for each split
# imgs = [file for file in dataset_path.rglob('**/*.jpg') 
#         if 'test' not in file.parts] # Img files container

# save_path = Path(dataset_path.parent / f'{datetime.date.today().isoformat()}_{_k}-Fold')
# save_path.mkdir(parents= True, exist_ok= True)
# yamls = []

# for split in folds_df.columns:
#     split_dir = save_path / split
#     split_dir.mkdir(parents= True, exist_ok= True)
#     (split_dir / 'train' / 'images').mkdir(parents= True, exist_ok=True)
#     (split_dir / 'train' / 'labels').mkdir(parents= True, exist_ok=True)
#     (split_dir / 'val' / 'images').mkdir(parents= True, exist_ok=True)
#     (split_dir / 'val' / 'labels').mkdir(parents= True, exist_ok=True)

#     dataset_yaml = split_dir / f'{split}_data.yaml'
#     yamls.append(dataset_yaml)

#     with open(dataset_yaml, 'w') as file:
#         yaml.safe_dump({
#             'path' : split_dir.as_posix(),
#             'train': 'train',
#             'val' : 'val',
#             'names' : classes
#         }, file)

# for img, lbl in zip(imgs, labels):
#     for split, k in folds_df.loc[img.stem].items():
#         img_to_path = save_path / split / k / 'images'
#         lbl_to_path = save_path / split / k / 'labels'

#         shutil.copy(img, img_to_path / img.name)
#         shutil.copy(lbl, lbl_to_path / lbl.name)

# folds_df.to_csv(save_path / "kfold_datasplit.csv")
# fold_dist.to_csv(save_path / "kfold_label_distribution.csv")

# print("K-Fold Split Successed!")

# # Training Experiment
# print(f'Experiment start: {datetime.datetime.now()}')
# detect_path = 'models/detect_2023_12_14.pt'

# # Build from detection model and transfer weights
# model = YOLO('yolov8s-seg.yaml',task = 'segment').load(detect_path)
# hyp_path = 'hyp/segment/best_hyperparameters.yaml'
# results = {}
# for i in range(_k):
#     dataset_yaml = yamls[i]
#     model.train(data = dataset_yaml,
#                       epochs = 100,
#                       batch = 32,
#                       cfg = hyp_path,
#                       save = True,
#                       project = f'Seg_train_{_k}-Fold_{datetime.date.today().isoformat()}',
#                       )
#     results[i] = model.metrics
# print (f"Experiment end: {datetime.datetime.now()}")