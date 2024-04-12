import os
from typing import List

import cv2
import tensorflow as tf

# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str) -> List[float]:
    # print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = frame[180:250, 100:220, :]
        frame = cv2.resize(frame, [140, 70])
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame)
        # frame = tf.image.rgb_to_grayscale(frame)
        # frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float16))
    return tf.cast((frames - mean), tf.float16) / std


def load_alignments(path: str) -> List[str]:
    # print(path)
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_alignments_text(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return tokens


def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..', 'data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('..', 'data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


def cosine_similarity(word1: str = None, words2: str = None) -> float:
    if word1 is None or words2 is None:
        return 0
    v1, v2 = [ord(x) for x in ''.join(word1.split())], [ord(x) for x in ''.join(words2.split())]
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / ((sumxx ** .5) * (sumyy ** .5))
