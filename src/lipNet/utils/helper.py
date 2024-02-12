import os
from typing import List
import tensorflow as tf
import cv2

lip_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def generate_mounth_video_dict(path:str) -> dict[str:List[float]]:
    video_dict= {}
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist')
    for file in os.listdir(path):
        if file.endswith('.mpg'):
            cap = cv2.VideoCapture(os.path.join(path, file))
            frames = []
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                # frame = tf.image.rgb_to_grayscale(frame)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lips = lip_cascade.detectMultiScale(gray_frame, 1.1, 4)
                for (x, y, w, h) in lips:
                    frame = gray_frame[y:y + h, x:x + w]
                    frame = tf.image.resize(frame, [46, 140])
                frames.append(frame)

                frames.append(frame)
            cap.release()
            mean = tf.math.reduce_mean(frames)
            std = tf.math.reduce_std(tf.cast(frames, tf.float16))
            video_dict[file] = tf.cast((frames - mean), tf.float16) / std
    return video_dict


    # video_dict= {}
    # if not os.path.exists(path):
    #     raise FileNotFoundError(f'{path} does not exist')
    # for file in os.listdir(path):
    #     if file.endswith('.mpg'):
    #         cap = cv2.VideoCapture(os.path.join(path, file))
    #         frames = []
    #         for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    #             ret, frame = cap.read()
    #             frame = tf.image.rgb_to_grayscale(frame)
    #             frames.append(frame[190:236,80:220,:])
    #         cap.release()
    #         mean = tf.math.reduce_mean(frames)
    #         std = tf.math.reduce_std(tf.cast(frames, tf.float16))
    #         video_dict[file] = tf.cast((frames - mean), tf.float16) / std
    # return video_dict