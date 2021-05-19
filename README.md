# 프로젝트 진행 방향

2 Major Tasks:

1. Pneumonia **Detection**
2. Pneumonia **Type** Classification

## Pneumonia **Detection**

- 데이터셋 이미지에서 ROI (Anchor Boxes) 를 추출하고, 각 박스 Classification
- [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) 데이터셋 / [ChestX-ray14](https://www.kaggle.com/nih-chest-xrays/data) 이용해야 할듯. (Bbox information 이 있음)
- Evaluation Metric: mAP
- RetinaNet 사용해보자.
- [논문리뷰](https://www.notion.so/Deep-Learning-for-Automatic-Pneumonia-Detection-aa3fe19cc48e46ab9006a6be12d13cd8) 에서 얻을 수 있는 아이디어들: Image Augmentation, Ensemble, Multi-task Learning
- 인풋: 전체 이미지, 아웃풋: 예측 박스, 박스 별 라벨

## Pneumonia Type Classification

- 세부적으로 어떤 폐렴인지 구별하는 것 (Viral, Bacterial, CoVid-19, etc.)
- [https://github.com/shervinmin/DeepCovid](https://github.com/shervinmin/DeepCovid) 참조 (데이터셋도 있음)
- 인풋: 전체 이미지, 아웃풋: 라벨

### Dataset Details

- RSNA 데이터셋을 대충 보니, 여기는 폐렴 / 폐렴 아님 2가지 밖에 없음.
    - 만약 bbox 데이터가 있으면, 폐렴이라는 뜻.
    - 즉, 모델의 아웃풋에 박스가 있다면, 그 박스는 모두 폐렴 의심 박스임.
    - 이미지 → 모델 → `[(박스 위치1, 폐렴 점수1), (박스 위치2, 폐렴 점수2), ...]`
- Covid / 세부 폐렴 데이터셋에는 Bbox 라벨링이 없음.
    - 따라서, Type Classifier 모델의 인풋은 전체 이미지여야하고,
    - 이미지 → 모델 → 폐렴 Type

따라서, Detection 으로 얻은 Box 들의 각각 type 을 classify 하는 모델을 학습하기는 어렵다.

## Ideas To Aggregate Both Models

가장 이상적인 활용: 

- Detection 으로 폐렴 의심되는 지역 (박스) 추출 → Type classifier 로 각 추출된 지역의 폐렴 type 예측
- **불가능**. 각 의심 지역의 이미지만 추출해서 그거를 예측하면 제대로된 결과가 나오지 않을 것임.
    - 혹 방법이 있을까? (semi-supervised learning methods?)
        - Segmentation 을 통해 Heatmap 을 구하고, 그에 따라 폐렴 감염 위치 (bounding boxes)를 labelling 해서 새로운 Covid-19 데이터셋을 만드는 방법도 있겠다. (TF 팀에서 읽었던 논문에 Heatmap 구하는게 있었으니)

가능한 활용 1:

- Detection 으로 폐렴 의심되는 지역 (박스) 추출, 해당 환자의 폐렴 타입 예측.
- 각 박스 별 폐렴 타입예측이 아닌, 전체 이미지에 대한 폐렴 타입 예측
- 문제: Evaluation 이 힘듬.
    - ChestXray-14 / RSNA 데이터셋으로 Evaluation: mAP 는 구할 수 있으나 (box prediction eval), 세부 타입에 대한 evaluation 불가.

    - Covid Dataset 으로 Evaluation: F1, accuracy 는 구할 수 있으나 mAP (box prediction eval) 불가.
    - 두가지 따로 하는 방법도 있기는 하다.

**가능한 활용 2 (semi-supervised Learning style: pseudo labelling):**

- Type classifier 학습 후 Labeler 로서 활용.
    1. RSNA / ChestXray-14 데이터셋에서, target box 가 하나뿐이고, 해당 Label 이 폐렴인 이미지들에 대해서 폐렴 type 예측 → 예측된 결과를 label 로 사용.
    2. Modify 된 데이터셋으로 Detection 모델 학습.
        - Evaluation: mAP (on modified dataset / original dataset), F1 / Accuracy on Covid Dataset

**활용 방안 2가 가장 가능성 있어 보이긴 하다.**

image path, label, bbox coordinate
