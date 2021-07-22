# 프로젝트 진행 방향

### Main Team Member
 - 이성빈 https://github.com/Noah-irooom
 - 장두혁 https://github.com/justin95214/
 - 최세현 https://github.com/syncdoth/

2 Major Tasks:

1. Pneumonia **Detection**
2. Pneumonia **Type** Classification
3. Multi-GPU 코드 개선

## Pneumonia **Detection**

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
