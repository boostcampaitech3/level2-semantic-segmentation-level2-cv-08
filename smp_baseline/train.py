import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

#!pip install albumentations==0.4.6
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch

#!pip install webcolors
import webcolors

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_schduler
import segmentation_models_pytorch as smp

#from losses_pytorch.dice_loss import ExpLog_loss

import argparse
import wandb
from loss import get_loss, get_scheduler
from dataset import load_dataset
from model import build_model
from utils import *

def main(args):
    print('pytorch version: {}'.format(torch.__version__))
    print('torchvision version: {}'.format(torchvision.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config 읽기
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    model_config = {}
    model_config["decoder"] = config["decoder"]
    model_config["encoder"] = config["encoder"]
    # Hyperparameter
    name = config["name"]
    seed = config["seed"]
    train_config = config["train"]
    num_epochs = train_config["epoch"]
    learning_rate = train_config["learning_rate"]

    # seed 고정
    set_seed(seed)

    # 모델
    model, preprocessing_fn = build_model(model_config)

    # DataLoader
    train_loader, val_loader = load_dataset(args, train_config, preprocessing_fn)
    
    def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device):
        print(f'Start training..')
        n_class = 11
        best_loss = 9999999
        criterion2 = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()

            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(data_loader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                
                # device 할당
                model = model.to(device)
                
                # inference
                outputs = model.forward(images) #['out']
                
                # loss 계산 (cross entropy loss)
                loss1 = criterion(outputs, masks)
                loss2 = criterion2(outputs, masks)
                loss = 0.2 * loss2 + 0.8 * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                
                # step 주기에 따른 loss 출력
                if (step + 1) % 4 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                            Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                    wandb.log({"train/loss": round(loss.item(),4), "train/mIoU": round(mIoU,4)})
            scheduler.step()
                
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % val_every == 0:
                avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
                if avrg_loss < best_loss:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {saved_dir}")
                    best_loss = avrg_loss
                    save_model(model, saved_dir, file_name=name+".pt")

    def validation(epoch, model, data_loader, criterion, device):
        print(f'Start validation #{epoch}')
        model.eval()

        with torch.no_grad():
            n_class = 11
            total_loss = 0
            cnt = 0
            
            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(data_loader):
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model.forward(images) #['out']
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
            
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]
            
            avrg_loss = total_loss / cnt
            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            print(f'IoU by class : {IoU_by_class}')
            iou_dict = {}
            for i in range(len(IoU_by_class)):
                iou_dict.update(IoU_by_class[i])
            iou_df = pd.DataFrame(list(iou_dict.items()), columns = ["Class", "IoU"])
            wandb.log({"val/class_IoU": iou_df})
            wandb.log({"val/loss": avrg_loss, "val/accuracy": round(acc, 4), "val/mIoU": round(mIoU, 4)})
            
        return avrg_loss

    # 모델 저장 함수 정의
    val_every = 1

    saved_dir = args.saved_dir
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model, output_path)

    # Loss function 정의
    criterion = get_loss(train_config["criterion"])

    # Optimizer 정의
    op = getattr(optim, train_config["optimizer"])
    optimizer = op(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
    scheduler = get_scheduler(train_config["scheduler"], optimizer)

    wandb.login()
    wandb.init(project="drivingyouth-SEG", entity="hbage", config=args)

    wandb.run.name = name
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', type=str, default="./saved")
    parser.add_argument('--config_dir', type=str, default="./config/deeplabv3+_swin.yaml")
    arg = parser.parse_args()
    main(arg)
    