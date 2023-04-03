# 2023년 1월 28일(토)
import torch.cuda
import os
import glob
import cv2

# device setting
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model call
model = torch.hub.load('ultralytics/yolov5', 'custom', path="../runs/train/exp8/weights/best.pt")
# model = custom(path_or_model='.\\runs\\train\\exp3\\weights\\best.pt')
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.to(DEVICE)

# image path list
# jpg파일을 가져옴
image_path_list = glob.glob(os.path.join("A:\\week1\\images", "*0510*.jpg"))

label_dict = {
    0: 'belt',
    1: 'no_belt',
    2: 'hoes',
    3: 'no_shoes',
    4: 'helmet',
    5: 'no_helmet',
    6: 'person'
}

# cvt_label_dict = {v: k for k, v in label_dict.items()}



def safe_check(person, no_person) :
    result_list = []
    for p in person :
        result = '-=*'
        # print('='*10, p[0], p[1], p[2], p[3])
        for np in no_person :
            np_x, np_y = (np[0] + np[2]) / 2 , (np[1] + np[3]) / 2
            # print('-' * 10, np[0], np[1], np[2], np[3], np_x, np_y)

            if p[0] <= np_x and p[2] >= np_x and p[1] <= np_y and p[3] >= np_y :
                # print('*', np[4])
                if np[4] == 0 :
                    result = result.replace('-', 'B')
                elif np[4] == 2 :
                    result = result.replace('=', 'S')
                elif np[4] == 4 :
                    result = result.replace('*', 'H')
        result_list.append([p[0], p[1], p[2], p[3], p[4], result])

    return result_list


# 하나하나의 이미지 추출
for i in image_path_list:
    image_path = i

    # cv2 image read
    image = cv2.imread(image_path)

    # model input
    output = model(image, size=640)

    bbox_info = output.xyxy[0]  # bounding box의 결과를 추출
    # for문을 들어가서 우리가 원하는 결과를 뽑는다.

    person = []
    no_person = []

    for bbox in bbox_info:
        # bbox에서 x1, y1, x2, y2, score, label_number의 결과를 가지고 온다.
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())

        score = bbox[4].item()

        label_number = int(bbox[5].item())

        if label_number == 6 :
            person.append([x1, y1, x2, y2, score])
        else :
            no_person.append([x1, y1, x2, y2, label_number])
        # try:
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #     cv2.putText(image, label_dict[label_number], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (0, 255, 255), 2)
        #     cv2.putText(image, str(round(score, 4)), (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (0, 255, 255), 2)
        # except Exception as e:
        #     print(e)

    result_list = safe_check(person, no_person)

    for r in result_list :
        x1, y1, x2, y2, score, result = r[0], r[1], r[2], r[3], r[4], r[5]
        try:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(image, r[5], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
            cv2.putText(image, str(round(r[4], 4)), (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        except Exception as e:
            print(e)

    cv2.imshow("test", cv2.resize(image, (1440, 810)))
    cv2.waitKey(0)
