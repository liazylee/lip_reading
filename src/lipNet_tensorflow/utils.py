import glob
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

MOUTH_W, MOUTH_H = 140, 70


def pre_process_mouth_extractor(path: str, mouth_size=(MOUTH_W, MOUTH_H)) -> None:
    base_path = path.split('.')[0]
    # if not os.path.exists(base_path + '.npy'):
    if path.endswith('.mpg'):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")
        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error: Could not read frame.")
            # Convert to grayscale

            # faces = lip_cascade.detectMultiScale(frame, scale_factor, min_neighbors)
            # if len(faces) == 0:
            #     print(f'Error: No face detected in {file_path}')
            #     continue
            frame = frame[180:250, 100:220, :]
            frame = cv2.resize(frame, mouth_size)
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame)
            # for (x, y, w, h) in faces:
            #     mouth_roi = frame[y + int(h / 2):y + h, x:x + w, :]

            #     frame = tf.image.rgb_to_grayscale(frame)
            #     frames.append(frame)

        cap.release()
        # if len(frames) < 62:
        #     print(f'Error: {file_path} has less than 62 frames,rather drop out', len(frames))
        #     return
        # Normalize frames
        mean = tf.math.reduce_mean(frames)  # compute the mean
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        frames_tensor = tf.cast((frames - mean), tf.float32) / std
        # Save as npy file
        np.save(base_path + '.npy', frames_tensor)
        return frames_tensor
    else:
        raise Exception("Error: File format not supported.")


DIR = '/home/liazylee/jobs/python/AI/lip_reading/src/lipNet/data/'  # absolute path

video_list = glob.glob(DIR + '/**/*.mpg', recursive=True)
# for video in tqdm.tqdm(video_list, total=len(video_list)):
#     pre_process_mouth_extractor(video)


if __name__ == '__main__':
    time1 = time.time()
    for video in tqdm(video_list, total=len(video_list)):
        pre_process_mouth_extractor(video)
    print(f'Pretain took {time.time() - time1} seconds')
    print('Pretain finished')
