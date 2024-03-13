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

from functools import reduce
from typing import Tuple

import cv2
import numpy as np
import torch

from config import NUMBER_DICT, DIR
from dataset import LRNetDataset

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def mouth_extractor(file_path: str, scale_factor=1.3, min_neighbors=5, mouth_size=(140, 70)) -> None:
    """
    Extract the mouth from the video and save as an npy file
    :param file_path: Path to the video file
    :param scale_factor: Parameter specifying how much the image size is reduced at each image scale
    :param min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
    :param mouth_size: Size of the extracted mouth region
    :return: None
    """
    base_path = file_path.split('.')[0]
    # if not os.path.exists(base_path + '.npy'):
    if file_path.endswith('.mpg'):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")

        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error: Could not read frame.")
            # 参数分别为低阈值和高阈值
            faces = face_cascade.detectMultiScale(frame, scale_factor, min_neighbors)
            for (x, y, w, h) in faces:
                mouth_roi = frame[y + int(h / 2):y + h, x:x + w, :]

                mouth_roi = cv2.resize(mouth_roi, mouth_size)
                frames.append(mouth_roi)

        cap.release()

        # Normalize frames
        frames_tensor = np.array(frames)
        mean = np.mean(frames_tensor)
        std = np.std(frames)
        frames_tensor = (frames - mean) / std
        # frames_tensor = np.expand_dims(frames_tensor, axis=-1)
        # Save as npy file

        np.save(base_path + '.npy', frames_tensor)
        return frames_tensor
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


# evaluate the model with the test data
def validate(model: torch.nn.Module, criterion: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """

    :param model:
    :param criterion:
    :param val_loader:
    :param device:
    :return:
    """
    model.eval()  # Set the model to evaluation mode
    val_loss, val_wer, val_cer = 0, 0, 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.permute(1, 0, 2)  # (time, batch, n_class)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            target_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1), dtype=torch.long)
            text_outputs: str = ctc_decode_tensor(outputs, input_lengths)
            text_targets: str = decode_tensor(targets)
            val_wer = calculate_wer(text_outputs, text_targets)
            val_cer = calculate_cer(text_outputs, text_targets)
            loss = criterion(outputs, targets, input_lengths, target_lengths)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_wer = val_wer / len(val_loader.dataset)
    val_cer = val_cer / len(val_loader.dataset)
    return avg_val_loss, val_wer, val_cer


# decode the tensor to string

def decode_tensor(tensor: torch.Tensor, number_dict: dict = NUMBER_DICT) -> str:
    """
  Decodes a tensor into a string using a mapping dictionary.
    Args:
        tensor (torch.Tensor): Input tensor of shape (4, 28) containing numerical indices.
        NUMBER_DICT (dict): Dictionary mapping numerical indices to characters.
    Returns:
        str: The decoded string.
    """
    tensors = tensor.tolist()
    result = ''
    for tensor in tensors:
        result += reduce(lambda x, y: x + y, [number_dict[i] if i != 0 else '' for i in tensor])
    return result


def ctc_decode_tensor(logits, input_lengths, number_dict: dict = NUMBER_DICT, blank_label=0, collapse_repeated=True) \
        -> str:
    """
    Decodes the output tensor from a CTC-trained model into human-readable strings.
    Assumes logits are in 'log softmax' form.

    Args:
        logits (Tensor): The model's raw output of shape (T, N, C).
        input_lengths (Tensor): The actual lengths of each sequence in the batch.
        number_dict (dict): Mapping from numerical indices to characters.
        blank_label (int): The index of the blank label in logits.
        collapse_repeated (bool): If True, collapse repeated characters.

    Returns:
        List[str]: Decoded strings for each sequence in the batch.
    """
    decoded_strings = []

    # Assuming logits are already in log softmax form
    max_probs, max_indices = torch.max(logits, 2)  # (T, N)

    for i, length in enumerate(input_lengths):
        raw_sequence = max_indices[:, i][:length].tolist()  # Truncate to actual length
        decoded_sequence = []

        last_label = None
        for label in raw_sequence:
            if label != last_label or not collapse_repeated:
                if label != blank_label:
                    decoded_sequence.append(number_dict.get(label, ''))
            last_label = label

        decoded_string = ''.join(decoded_sequence)
        decoded_strings.append(decoded_string)

    return ''.join(decoded_strings)


def load_train_test_data(split_ratio: float = 0.8) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    video_dataset = LRNetDataset(DIR)
    train_size = int(split_ratio * len(video_dataset))
    test_size = len(video_dataset) - train_size
    return torch.utils.data.random_split(video_dataset, [train_size, test_size])


def calculate_wer(predicted: str, true: str) -> float:
    """

    :param predicted: str
    :param true: str
    :return: float
    """
    d = np.zeros((len(predicted) + 1, len(true) + 1), dtype=np.uint8)
    for i in range(len(predicted) + 1):
        d[i][0] = i
    for j in range(len(true) + 1):
        d[0][j] = j
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(true) + 1):
            if predicted[i - 1] == true[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j]) + 1
    return d[len(predicted)][len(true)] / len(true)


# caculate the cer
def calculate_cer(predicted: str, true: str) -> float:
    """

    :param predicted: str
    :param true: str
    :return: float
    """
    d = np.zeros((len(predicted) + 1, len(true) + 1), dtype=np.uint8)
    for i in range(len(predicted) + 1):
        d[i][0] = i
    for j in range(len(true) + 1):
        d[0][j] = j
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(true) + 1):
            if predicted[i - 1] == true[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j]) + 1
    return d[len(predicted)][len(true)] / max(len(true), 1)
