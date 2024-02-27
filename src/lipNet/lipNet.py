"""
@author:liazylee
@license: Apache Licence
@time: 06/12/2023 13:18
@contact: li233111@gmail.com
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio


import cv2
import mediapipe as mp

# 初始化Mediapipe的Face模型
mp_face = mp.solutions.face_detection

# 初始化Face检测器
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# 加载视频
video_path = "data/s1/bbbmzn.mpg"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行Face检测
    results = face_detector.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            print(f'Image Height: {ih}, Image Width: {iw}')
            print(f"Bounding Box: {bboxC.xmin * iw}, {bboxC.ymin * ih}, {bboxC.width * iw}, {bboxC.height * ih}")
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # 输出嘴唇区域的长宽
            lip_width = w
            lip_height = h
            # print(f"Lip Width: {lip_width}, Lip Height: {lip_height}")
            print(f'x: {x}, y: {y}, w: {w}, h: {h}')
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(frame, (x, y), (x + 100, y + 50), (255, 0, 0), 2)


    # 显示结果
    cv2.imshow("Lip Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



