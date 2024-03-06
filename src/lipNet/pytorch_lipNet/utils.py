"""
@author:liazylee
@license: Apache Licence
@time: 26/02/2024 12:24
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
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def mouth_extractor(file_path: str, scale_factor=1.3, min_neighbors=5, mouth_size=(140, 70)) -> None:
    """
    Extract the mouth from the video save as npy file
    :param file_path:
    :return: npy file
    """
    base_path = file_path.split('.')[0]
    if not os.path.exists(base_path + '.npy'):
        if file_path.endswith('.mpg'):
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise Exception("Error: Could not open video.")
            frames = []
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Error: Could not read frame.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
                for (x, y, w, h) in faces:
                    mouth_roi = frame[y + int(h / 2):y + h, x:x + w]
                    mouth_roi = cv2.resize(mouth_roi, mouth_size)
                    mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                    frames.append(mouth_roi)

            cap.release()
            mean = np.mean(frames)
            std = np.std(frames)
            frames = [(frame - mean) / std for frame in frames]
            frames_tensor = np.array(frames)
            # save as npy file
            np.save(base_path + '.npy', frames_tensor) # change the filepath
        else:
            raise Exception("Error: File format not supported.")




# write a wrapper function to caculate the time of the function and not loss any information
def timmer(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} took {end_time - start_time} seconds')
        return result
    return wrapper