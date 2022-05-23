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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

#from losses_pytorch.dice_loss import ExpLog_loss

import argparse
import wandb

def main(args):
    plt.rcParams['axes.grid'] = False

    print('pytorch version: {}'.format(torch.__version__))
    print('torchvision version: {}'.format(torchvision.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameter
    batch_size = args.batch_size   # Mini-batch size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # seed 고정
    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    dataset_path  = '../data'
    anns_file_path = dataset_path + '/' + 'train_all.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    for cat_it in categories:
        cat_names.append(cat_it['name'])

    # Count annotations
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)

    category_names = list(sorted_df.Categories)

    def get_classname(classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

    class CustomDataLoader(Dataset):
        """COCO format"""
        def __init__(self, data_dir, mode = 'train', transform = None):
            super().__init__()
            self.mode = mode
            self.transform = transform
            self.coco = COCO(data_dir)
            
        def __getitem__(self, index: int):
            # dataset이 index되어 list처럼 동작
            image_id = self.coco.getImgIds(imgIds=index)
            image_infos = self.coco.loadImgs(image_id)[0]
            
            # cv2 를 활용하여 image 불러오기
            images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            images /= 255.0
            
            if (self.mode in ('train', 'val')):
                ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
                anns = self.coco.loadAnns(ann_ids)

                # Load the categories in a variable
                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)

                # masks : size가 (height x width)인 2D
                # 각각의 pixel 값에는 "category id" 할당
                # Background = 0
                masks = np.zeros((image_infos["height"], image_infos["width"]))
                # General trash = 1, ... , Cigarette = 10
                anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
                for i in range(len(anns)):
                    className = get_classname(anns[i]['category_id'], cats)
                    pixel_value = category_names.index(className)
                    masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
                masks = masks.astype(np.int8)
                            
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    transformed = self.transform(image=images, mask=masks)
                    images = transformed["image"]
                    masks = transformed["mask"]
                return images, masks, image_infos
            
            if self.mode == 'test':
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    transformed = self.transform(image=images)
                    images = transformed["image"]
                return images, image_infos
        
        def __len__(self) -> int:
            # 전체 dataset의 size를 return
            return len(self.coco.getImgIds())

    # train.json / validation.json / test.json 디렉토리 설정
    train_path = dataset_path + '/stratified_kfold/train_fold1.json'
    val_path = dataset_path + '/stratified_kfold/val_fold1.json'
    test_path = dataset_path + '/test.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))


    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose([
                            #A.RandomSizedCrop(min_max_height=(126, 256), height=512, width=512, p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            #A.RandomBrightnessContrast(),
                            #A.HueSaturationValue(),
                            ToTensorV2()
                            ])

    val_transform = A.Compose([
                            #A.HorizontalFlip(p=0.5),
                            #A.VerticalFlip(p=0.5),
                            #A.RandomBrightnessContrast(),
                            #A.HueSaturationValue(),
                            ToTensorV2()
                            ])

    test_transform = A.Compose([
                            ToTensorV2()
                            ])

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)


    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)



    model = models.segmentation.fcn_resnet101(pretrained_backbone=False)
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    #model = models.segmentation.deeplabv3_resnet101(pretrained_backbone=True, num_classes=11)
    #model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained_backbone=True, num_classes=11)
    import segmentation_models_pytorch as smp
    
    model = smp.PSPNet(
        encoder_name="tu-hrnet_w64",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                      # model output channels (number of classes in your dataset)
    )
    
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
                    save_model(model, saved_dir, file_name=args.best_name)

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
    # criterion = nn.CrossEntropyLoss()
    criterion = smp.losses.DiceLoss('multiclass')

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    run_name = vars(args).pop("run_name")
    wandb.login()
    wandb.init(project="drivingyouth-SEG", entity="hbage", config=args)

    wandb.run.name = run_name
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--saved_dir', type=str, default="./saved")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--best_name', type=str, default="deeplabv3_resnet101_best_model(backbone_pretrained).pt")
    parser.add_argument('--run_name', type=str, default="deeplabv3_resnet101_best_model(backbone_pretrained)")
    arg = parser.parse_args()
    main(arg)