# object detection toy project
1. 공사 현장 안전 장비 미착용으로 인한 사고 방지 (스마트 안전 통합 관제 시스템)
2. 배운 내용으로 데이터 셋 구축 및 서비스 구현
3. Tracking 기존 코드에 추가하는중

https://github.com/kcw0331/MS-AI-School/tree/main/Computer-Vision/Construction-Site-Safety-PPE-Alert-Detection

<참고자료>

http://news.kmib.co.kr/article/view.asp?arcid=0016070557&code=61121111&cp=nv

https://newsis.com/view/?id=NISX20230127_0002172017

https://www.hankyung.com/society/article/202302032530Y


<hr>
-*- encoding - python 3.8 -*
I contributed gui.py

<hr>

## Step

### 1. Dataset
데이터 다운로드 및 데이터 선택
이미지 검수 및 이미지 제거 또는 CVAT로 수정작업
### 2. Train
yolov5모델을 선택하여 training
### 3. Inference
### 4. GUI
pyqt5 - GUI 구현
### 5. Tracking - DeepSort
사람 객체 인식 후 ID 부여
사람 객체 인식 후, 장구류 착용 여부 확인
### 6. Timer
장구류 미착용 사람 발견 시, 타이머 작동
### 7. Alarm
일정시간 미착용 시 이메일 발송 


<hr>

## 1. Dataset
### Label 1
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

### Label 2
```
label_dict = {
    0: 'belt',
    1: 'shoes',
    2: 'helmet',
    3: 'person'
}
```

### Dataset
AI 허브(공사 현장 안전장비 인식 이미지)

https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=163

roboflow(Personal Protective Equipment - Combined Model)

https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/browse?queryText=class%3A%22NO-Safety+Vest%22&pageSize=50&startingIndex=500&browseQuery=true

roboflow(Hard Hat Workers Dataset) - 안전모 미착용

https://public.roboflow.com/object-detection/hard-hat-workers/2


roboflow(clothes detect) - 안전조끼 미착용

https://universe.roboflow.com/zhang-ya-ying/clothes-detect-fevqm/dataset/5

roboflow(site2)

https://app.roboflow.com/changwoo-kim-vvfty/site2/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

roboflow(whole_dataset) - 안전화 미착용

https://universe.roboflow.com/business-qcddc/whole_dataset/dataset/4


### Person Label
```
Yolov5 - detect.py - pretrained model (yolo5s, default) - label:0
```

<hr>

## 2. Train
### [Yolo v5](https://github.com/ultralytics/yolov5)
### Pytorch version
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<hr>

## 4. GUI (PyQt5)

```
pip install pyqt5
```
### 동영상 플레이어 참고 https://oceancoding.blogspot.com/2020/07/blog-post_22.html

<hr>

## 5. Reference

https://koreascience.kr/article/JAKO201915555313326.pdf
https://ysyblog.tistory.com/m/294 (Python email 이미지 첨부)


