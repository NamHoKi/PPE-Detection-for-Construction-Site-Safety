# PPE Detection for Construction Site Safety using YoloV5
## object detection toy project
1. 공사 현장 안전 장비 미착용으로 인한 사고 방지 (스마트 안전 통합 관제 시스템)
2. 배운 내용으로 데이터 셋 구축 및 서비스 구현

skill - Yolov5, Deep Sort

## Introduction

- There are many safety accidents without wearing safety equipment (safety helmets, seat belts, safety shoes, etc.) at the construction site until 2023, and a pilot company has been selected and implemented since February 2022. 

![reference1](https://user-images.githubusercontent.com/48282708/229698705-21ebde2f-5dea-4a0c-b695-0a7ec6b745bf.jpg)

- It was seen that an employee who did not wear safety equipment at the construction site was notified to the construction site manager when he was found not wearing it for a certain period of time on CCTV. 
- Therefore, we are working on the project with the goal of making the current Object Detection class and practice similar to the actual practice.

- references:

    - [reference_1](http://news.kmib.co.kr/article/view.asp?arcid=0016070557&code=61121111&cp=nv),
    [reference_2](https://newsis.com/view/?id=NISX20230127_0002172017),
    [reference_3](https://www.hankyung.com/society/article/202302032530Y)

---

- Number of participants: 5 people

- Technology Stack: Python, ,Pytorch(ObjectDetection), Pyqt5, Yolov5, Yolov7, Deepsort, MMdetection

- My Role: Data Collection and Purification (Labeling and Bounding Box using CVAT), Data Segmentation, Model Training, GUI Implementation using Pyqt5 and Tracking Connection using Deepsort.

---


<hr>
-*- encoding - python 3.8 -*
I contributed gui.py

<hr>

## Step

### 1. Dataset
데이터 다운로드 및 데이터 선택
이미지 검수 및 이미지 제거 또는 CVAT로 수정작업

### 2. Train & Inference
yolov5모델을 선택하여 training

### 3. GUI & 구현기능
- GUI - PyQt5 : 메인화면 및 기능 구현
- Tracking - DeepSort : 사람 객체 인식 후 ID 부여, 사람 객체 인식 -> 장구류 착용 여부 확인
- Timer : 장구류 미착용 사람 발견 시, 타이머 작동
- Alarm : 일정시간 미착용 시 이메일 발송 


<hr>

## 1. Dataset
### Label

- There are 7 classes to detect from the dataset:
    
   ![week1_labels](https://user-images.githubusercontent.com/48282708/229698707-b929e5b0-1e38-4690-ad19-3853901d9c2a.png)
    
    - 'Safety_Belt', 'No_Safety_Belt', 'Safety_Shoes', 'No_Safety_shoes', 'Safety_helmet', 'No_Safety_helmet', 'Person'

```
label_dict = {
    0: 'belt',
    1: 'no_belt',
    2: 'shoes',
    3: 'no_shoes',
    4: 'helmet',
    5: 'no_helmet',
    6: 'person'
}
```



### Dataset
### Person Label
```
Yolov5 - detect.py - pretrained model (yolo5s, default) - label:0
```

- [AI 허브(공사 현장 안전장비 인식 이미지)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=163)

- [roboflow(Personal Protective Equipment - Combined Model)](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/browse?queryText=class%3A%22NO-Safety+Vest%22&pageSize=50&startingIndex=500&browseQuery=true)

- [roboflow(Hard Hat Workers Dataset) - 안전모 미착용](https://public.roboflow.com/object-detection/hard-hat-workers/2)


- [roboflow(clothes detect) - 안전조끼 미착용](https://universe.roboflow.com/zhang-ya-ying/clothes-detect-fevqm/dataset/5)

- [roboflow(site2)](https://app.roboflow.com/changwoo-kim-vvfty/site2/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

- [roboflow(whole_dataset) - 안전화 미착용](https://universe.roboflow.com/business-qcddc/whole_dataset/dataset/4)


<hr>

## 2. Train & Inference
### [Yolo v5](https://github.com/ultralytics/yolov5)
### Pytorch version
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Results

### 1. Yolov5
The training of Yolov5n model was done for 100 epochs and wa completed in about 5 hours. After training, we get the following results:

<details><summary>VM Environment</summary>
GPU : Tesla V100, Memory : 112GB, CPU : Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz 
</details>  

### Data set
|Type|Images|
|:----:|:----:|
|Train|11,495|
|Valid|1,437|

- Train image data is 11,495 and Valid image data is 1,437.

### Annotation Information(Bounding Box)
|Label|Train|Valid|
|:----:|:----:|:----:|
|Safety_Belt|7,841|1,006|
|No_Safety_Belt|14,820|2,111|
|Safety_Shoes|8,979|1,156|
|No_Safety_Shoes|6,607|881|
|Safety_Helmet|13,747|2,480|
|No_Safety_Helmet|6,474|811|
|Person|25,855|3,235|
|Total|84,323|11,680|

- When determining the number of bounding boxes, 84,323 bounding boxes for Train and 11,680 bounding boxes for Valid.


### 1.1. Yolov5 & Yolov7 result Table
|Model|Hyperparameter|Batch_size|Epochs|optimizer|mAP0.5|mAP0.5-0.95|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Yolov5n|Hyp.scratch-low|16|100|SGD|0.83782|0.49619|
|Yolov5n|Hyp.scratch-low|16|100|AdamW|0.81|0.469|
|Yolov5s|Hyp.scratch-high|84|100|SGD|0.86399|0.53363|
|Yolov5l|Hyp.scratch-high|16|30|SGD|0.8715|0.55632|
|Yolov5m|Hyp.scratch-low|32|100|SGD|0.88586|0.5667|
|Yolov5x|Hyp.scratch-med|16|100|Adam|0.88726|0.57239|
|Yolov7|Hyp.scratch.p5|16|200|SGD|0.933|0.598|

- When we looked at Yolov5's model with n, s, l, m, x, and Yolov7, we found that mAP 0.5~0.95 did not have more than 0.6 values.

### 1.2. Yolov5 of confusion_matrix result(Yolov5x):

![confusion_matrix](https://user-images.githubusercontent.com/48282708/229697502-11d06cc0-6a9e-4ee0-a1d7-e3381d8d33fa.png)

### 1.3. Yolov5 of Val_batch_label result(Yolov5x):

![val_batch2_pred](https://user-images.githubusercontent.com/48282708/229697832-a68c307f-11dc-4803-ace0-1c5035596cb5.jpg)

- This is the image when I trained with Train and printed out the image of Validation.

### 1.4. Yolov5 of result(Yolov5x):

![results](https://user-images.githubusercontent.com/48282708/229697976-498c5b72-d9f9-404b-a77f-e1d84982844f.png)

- In mAP 0.5 it has a value of more than 0.8, but in mAP 0.5 to 0.95, when 100 epochs were turned, it was found to converge from 0.55 to 0.6.
- So even when we turned Yolov7 200 epochs, we could see that it converges below 0.6. I've only watched 1-Stage, so I'm trying to see if I get different results when I watch 2-Stage on MMdetection. We're also figuring out if the data is weird.

<hr>

## 3. GUI & 구현기능  

```
pip install pyqt5
```


- Implementation Features
    - Explore other janggu objects within the person object (check whether a person wears janggu or not)
       - Person Make sure that the center value of another object is in the bounding box
    - Object check function that you want to recognize (the program user selects whether to wear the right equipment for the workplace)
    - Email notification service when not wearing it on the screen for a certain period of time or more

![result_GUI](https://user-images.githubusercontent.com/48282708/229698477-8aaa372a-f7d6-414d-afd6-df0344bd0a83.png)

- reference: https://www.youtube.com/watch?v=JZoFy66h8aY

- The Yolov5 pt file and the bounding box tracked people with Deepsort are implemented by calculating IOU, wearing a safety helmet, seat belt, and safety shoes using the GUI to indicate whether it is safe or not, and sending an e-mail to the manager for a certain period of time. 

![week1_email](https://user-images.githubusercontent.com/48282708/229698393-b4f25388-c37d-48b9-b75f-844f258187b5.png)

- Have the operator send an e-mail message to the administrator when safety equipment is not worn for a certain period of time.

- When you send an e-mail, capture the person who is not wearing it as shown in the picture above and send it together by e-mail.

### [동영상 플레이어 참고](https://oceancoding.blogspot.com/2020/07/blog-post_22.html)

<hr>

## Reference
https://koreascience.kr/article/JAKO201915555313326.pdf
https://ysyblog.tistory.com/m/294 (Python email 이미지 첨부)
