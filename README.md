# 프로젝트 진행 방향

### Main Team Member
 - 장두혁 https://github.com/justin95214/
 - 최세현 https://github.com/syncdoth/
 - 이성빈 https://github.com/Noah-irooom

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


# Detection

0. 각자의 PC의 tensorflow/tensorflow-gpu 가 되는 환경을 만들어 준다.

맥북의 터미널과 우분투 기반 터미널 기준이긴 하지만 Colab에서 서버까지 들어가셨으면,

bash명령어는 앞에 ! 를 붙여 쓰면 되는 것같아서 동일하다고 생각됩니다.

코랩환경 참고 부분

https://colab.research.google.com/drive/1FCkzeWjFsjxdh0PZbmBGfveirXL-0739?usp=sharing

https://colab.research.google.com/drive/1FCkzeWjFsjxdh0PZbmBGfveirXL-0739?usp=sharing

코랩에서 `/home/server/duhyeuk` 부분이 `/content/data_server` 였나... ()

코랩에서는 tensorflow-gpu를 아래와같은 버전을 설치해주셔야합니다.

(Colab에서 돌아가는 것을 확인함)

```bash
!pip install tensorflow-gpu=2.3.0
```

1. retinanet-keras 폴더를 찾아서, 

![경로.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/%EA%B2%BD%EB%A1%9C.png)

그림과 같이 경로를 들어간다. 경로를 들어가는 방법은 "cd(스페이스바) 폴더명" 으로 하시면됩니다.

## Train 하는 방법

1. class명을 적힌 csv파일과 Annotation format의 csv파일을 준비한다.

2. 아래의 teminal창에 아래와 같은 명령어를 작성한다.

class명을 적은 csv

![class_anno.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/class_anno.png)

Annotation format의 csv

![img_anno.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/img_anno.png)

```bash
python train.py --gpu=0 --steps=1000 -workers=0 Annotation format의 csv class명을 적은 csv
(파일명을 적을땐 항상 파일이 있는 경로를 명시해줘야합니다.) 
1. Annotation csv파일은 항상 /home/server/duhyeuk/ 경로에 있음
2. class명의 csv파일은 항상 /home/server/duhyeuk/class/meta/ 경로에 있음
```

![line.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/line.png)

3/27일 기준으로~

현재 쓰는 파일은 /home/server/duhyeuk/meta_ver6.csv 과  /home/server/duhyeuk/class/meta/mappingclass_3class.csv 이 사용됩니다.

나머지는 epochs와 steps, workers는 학습하실 때 조정하시면 됩니다. 

그러면 각 스텝마다 h5 모델을 저장하게된다.

## Test 하는 방법

1. 학습한 모델을 Inference 할 수있게 변환한다
2. Inference 한다

변환하는 과정

```bash
python keras_retinanet/bin/convert_model.py ../snapshots/resnet50_csv_02.h5 ../snapshots/resnet50_csv_10_infer.h5

```

학습 과정에 train.py파일이 있는 경로상태에서 conver_model.py 파일을 실행시킨다. 

실행시키는 방법을 위에 처럼 하면되는데

이 순서 >>   명령어python    py파일명    변환시킬모델h5    변환된모델h5(작명) 

변환된모델h5(작명) 에 변환된 모델이 생기는데 이것을 쓰면 됩니다.

추론하는 과정

경로를  /home/server/duhyeuk/ 으로 가시면

![path.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/path.png)

model_path 부분에 생성한 모델을 경로와 같이 입력해주시고, 

만약에 class명의 csv과 동일하게 labels_to_names 부분을 그림과같은 규칙대로 작성


![instruction.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/instruction.png)

추론할 이미지도 작은 따옴표안에 경로와 같이 입력해주시고 파일을 저장뒤에 아래와같은 명령어로 실행시켜주시면 됩니다.

```bash
python retinanet_test.py
```

![xray.png](https://github.com/justin95214/Eagle-Eye-Pneumonia-Detection/blob/master/resource/xray.png)


[Object Detection 성능평가 Metric ](https://www.notion.so/Object-Detection-Metric-589e4ac95bc446f297d4ddf44fd45663)

```jsx
python [evaluate.py](http://evaluate.py/) --backbone=resnet101 csv ~/O-E-E/duhyeuk/meta-handmade-test.csv ./meta/mappingclass_3class.csv ~/O-E-E/sehyun/snapshots/un_PN_infer_res101_ep25.h5 --gpu=1 —-use_tc=True
```
