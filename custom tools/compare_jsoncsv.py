import os
import json
import warnings
from xmlrpc.client import boolean 
warnings.filterwarnings('ignore')

from utils import label_accuracy_score, add_hist

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
import shutil
import csv
from tqdm import tqdm, trange

import argparse

def main(args):
    dataset_path  = args.dataset_dir
    anns_file_path = dataset_path + '/' + args.json_dir

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    coco = COCO(anns_file_path)
    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']

    # read prediction csv
    output = pd.read_csv(args.csv_dir)
    output_img = output["image_id"]
    output_anno = output["PredictionString"]
    output = []
    print("Preprocessing CSV file....")
    for i in trange(len(output_anno)):
        m = output_anno[i]
        m = list(map(int, m.split()))
        m = [m[i:i+512] for i in range(0, len(m), 512)]
        output.append(np.array(m))

    # miou calculate
    print("Calculating mIoUs....")
    n_class = 11
    miou_list = []
    for i in trange(len(output_img)):
        file_name = imgs[i]['file_name']
        name = file_name.split("/")[-1]
        masks = np.zeros((imgs[i]["height"], imgs[i]["width"]))
        for ann in anns:
            if ann['image_id'] == imgs[i]['id']:
                pixel_value = ann["category_id"]
                masks[coco.annToMask(ann) == 1] = pixel_value
                masks = masks.astype(np.int8)
        hist = np.zeros((n_class, n_class))
        hist = add_hist(hist, masks, output[i], n_class=n_class)
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        if mIoU < args.threshold:
            miou_list.append(i)
            src = os.path.join(dataset_path, file_name)
    print(f"MIoU가 {args.threshold}를 넘지않는 data의 개수:", len(miou_list))

    save_dir = args.save_dir
    existing = None
    if os.path.exists(save_dir) and args.con:
        with open(save_dir, "r") as json_file:
            existing = json.load(json_file)

    new_json = {}
    new_json['info'] = dataset['info']
    new_json['licenses'] = dataset['licenses']
    new_json['categories'] = categories


    new_imgs = [imgs[i] for i in miou_list]
    new_json['images'] = new_imgs

    image_list = [imgs[i]['id'] for i in miou_list]
    new_anno = [anns[i] for i in range(len(anns)) if anns[i]['image_id'] in image_list]
    new_json['annotations'] = new_anno

    if existing:
        new_json['images'].extend(existing['images'])
        new_json['annotations'].extend(existing['annotations'])

    with open(save_dir, "w") as json_file:

        json.dump(new_json, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--json_dir', type=str, default="cv_val.json")
    parser.add_argument('--csv_dir', type=str, default="./submission/cv_val.csv")
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default="inspect_1.json")
    parser.add_argument('--con', type=bool, default=False)
    arg = parser.parse_args()
    main(arg)