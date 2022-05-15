# level2-semantic-segmentation-level2-cv-08
level2-semantic-segmentation-level2-cv-08 created by GitHub Classroom

## 1. 프로젝트 개요

------

### 1.1 프로젝트 주제

![Untitled](C:\Users\Administrator1\Downloads\Untitled.png)

대량 생산, 대량 소비의 시대로 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

### 1.2 프로젝트 개요

(프로젝트 구현 내용, 컨셉, 교육 내용과의 관련성 등)

- EDA

  - analysis tool

    - ensemble

      hard voting기법으로 여러 모델에서 나온 output ensemble

- 컨셉

  - 분석한 내용을 바탕으로 모델 선정, augmentation, ensemble을 적용하여 최대 mAP 달성

- 교육 내용과의 관련성

  - segmentation의 다양한 모델 실험
  - transformer 계열 backbone 사용

### 1.3 활용 장비 및 재료

- 협업 툴 : GitHub, WandB, Notion
- 개발 환경
  - OS : Ubuntu 18.04
  - GPU : V100
  - 언어 : Python 3.7
  - dependency : Pytorch 1.7.1

### 1.4 프로젝트 구조 및 사용 데이터셋의 구조도

- 프로젝트 구조
  - 각 폴더에는 해당하는 library file들이 존재하며, mmdetection/configs/custom에 모델을 실험한 config파일이 존재
  - z_customs 폴더에는 Kfold, pseudo labeling, 결과 분석 툴, 앙상블 등의 파일이 존재

```
├── 📂 mmsegmentation
│   ├── 📂 configs
│   │   └── 📂 custom
│   ├── 📂 tools
│   │   ├── 📝 train.py
│   │   └── 📝 test.py
│   └── 📂 data
│				├── 📂 training
│				└── 📂 testing
└── 📂 custom tools
    ├── 📝 ensemble.ipynb
		├── 📝 Change2mmsegFormat
		└── etc
```

- 사용 데이터셋의 구조도
  - 10개의 클래스
    - `Background` , `General trash`, `Paper`, `Paper pack`, `Metal,` `Glass,` `Plastic`, `Styrofoam`, `Plastic bag`, `Battery`, `Clothing`,
  - annotation file
    - images:
      - id: 파일 안에서 image 고유 id, ex) 1
      - height: 512
      - width: 512
      - file*name: ex) train*/002.jpg
    - annotations:
      - id: 파일 안에 annotation 고유 id, ex) 1
      - segmentation: masking 되어 있는 고유의 좌표
      - bbox: 객체가 존재하는 박스의 좌표 (x*min, y*min, w, h)
      - area: 객체가 존재하는 영역의 크기
      - category_id: 객체가 해당하는 class의 id
      - image_id: annotation이 표시된 이미지 고유 id

### ● 기대 효과

만들어진 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것

## 2. 프로젝트 팀 구성 및 역할

------

- 김준 : 오분류 클래스 분석 및 augmentation 실험
- 윤서연: smp 사용, 데이터 클렌징 전처리
- 이재홍: Cross-Validation 코드 작성, CRF (Conditional Random Field) 적용하여 실험
- 이준혁: augmentation 실험 및 데이터 relabeling
- 허석: mmsegmentation 환경설정 및 Swin - Upernet 모델 선정

## 3. 프로젝트 수행 절차 및 방법

------

- 프로젝트 사전 기획

  - 1주차 - 데이터 전수조사 및 모델 선정
  - 2주차 - 모델 학습 후 mIoU 0.5이하 데이터 전수조사 및 data augmentation 실험
  - 3주차 - CRF 적용 및 ensemble 실험

- 프로젝트 수행 및 완료

  ![p stage2_복사본-001](C:\Users\Administrator1\Downloads\p stage2_복사본-001.png)

## 4. 프로젝트 수행 결과

------

### ○ 탐색적 분석 및 전처리 (학습데이터 소개)

- 전수조사
  - 오분류 데이터 삭제 및 수정

### ○ 모델 개요

- HRNet
  - backbone : `HRNet`
  - head : `FCN`
  - Loss
    - loss : `CrossEntropyLoss`
- UperNet
  - backbone : `Swin-L`, `Swin-base`
  - head : `Uperhead`, `auxiliaryhead`
  - Loss
    - loss : `Focal Loss`, `CrossEntropyLoss`
    - auxiliary loss : `CrossEntropyLoss`
- DeepLabV3
  - backbone : `ResNet101`
  - Loss
    - cls_loss : `CrossEntropyLoss`

### ○ 모델 선정 및 분석

- smp, mmsegmentation, torchvision library를 이용하여 다양한 모델 성능 시험
- wandb를 통한 mIoU 상위 모델 중심으로 실험 진행(UperNet, DeepLabV3)

### ■ 모델 성능

- 최종 public mAP: 0.7145
- 최종 private mAP: 0.7262

