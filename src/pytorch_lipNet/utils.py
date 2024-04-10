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
from typing import Tuple, List

import cv2
import numpy as np
import torch
# from ctcdecode import CTCBeamDecoder  # noqa
from jiwer import wer, cer
from torch import Tensor

from config import NUMBER_DICT, DIR, LETTER, MOUTH_W, MOUTH_H
from dataset import LRNetDataset

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def mouth_extractor(file_path: str, scale_factor=1.3, min_neighbors=5, mouth_size=(MOUTH_W, MOUTH_H)) -> None:
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
            # Convert to grayscale
            faces = face_cascade.detectMultiScale(frame, scale_factor, min_neighbors)
            if len(faces) == 0:
                print(f'Error: No face detected in {file_path}')
                continue
            for (x, y, w, h) in faces:
                mouth_roi = frame[y + int(h / 2):y + h, x:x + w]
                mouth_roi = cv2.resize(mouth_roi, mouth_size)
                mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                frames.append(mouth_roi)
        cap.release()
        if len(frames) < 62:
            print(f'Error: {file_path} has less than 62 frames,rather drop out', len(frames))
            return
        # Normalize frames
        frames_tensor = np.array(frames, dtype=np.float32)
        mean = np.mean(frames_tensor)
        std = np.std(frames_tensor)
        # Normalize frames
        frames_tensor = (frames_tensor - mean) / std  # Broadcasting subtraction and division
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
             val_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[
    np.float64, np.float64, np.float64]:
    """

    :param model:
    :param criterion:
    :param val_loader:
    :param device:
    :return:
    """
    # model.eval()  # Set the model to evaluation mode
    val_loss, val_wer, val_cer = [], [], []
    with torch.no_grad():  # Disable gradient calculation during validation
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            model.eval()
            outputs = model(inputs)
            text_outputs: List[str] = ctc_decode_tensor(outputs)
            text_targets: List[str] = decode_tensor(targets)
            val_wer.append(calculate_wer(text_outputs, text_targets))
            val_cer.append(calculate_cer(text_outputs, text_targets))
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            # outputs_detached = outputs.detach()
            val_loss.append(loss.item())

    avg_val_loss, val_wer, val_cer = np.mean(val_loss), np.mean(val_wer), np.mean(val_cer)
    print(f'Validation Loss: {avg_val_loss}, WER: {val_wer}, CER: {val_cer}')
    print(f'text_outputs: {text_outputs}, text_targets: {text_targets}')
    return avg_val_loss, val_wer, val_cer


# decode the tensor to string

def decode_tensor(tensor: torch.Tensor, ) -> List[str]:
    """
  Decodes a tensor into a string using a mapping dictionary.
    Args:
        tensor (torch.Tensor): Input tensor of shape (4, 28) containing numerical indices.
        NUMBER_DICT (dict): Dictionary mapping numerical indices to characters.
    Returns:
        str: The decoded string.
    """
    tensors = Tensor.cpu(tensor).detach().numpy()
    result = []
    for tensor in tensors:
        result.append(reduce(lambda x, y: x + y,
                             [NUMBER_DICT.get(i, '') for i in tensor]))

    return result


def ctc_decode_tensor(tensor: torch.Tensor, greedy: bool = True, beam_width: int = 100) -> List[str]:
    """
    Decodes a tensor into a string using a mapping dictionary.
    Args:
        tensor: Input tensor of shape (time, batch, n_class) containing
        greedy:
        beam_width:
    Returns: str
    """
    blank_label = 29
    decoded_sequences = []
    if greedy:
        for i in range(tensor.shape[1]):
            sequence = tensor[:, i, :].argmax(-1)  # (time, n_class) -> (time,)
            indices = torch.unique_consecutive(sequence, dim=-1)
            indices = [i for i in indices if i != blank_label]
            joined = "".join([NUMBER_DICT.get(i.item(), '') for i in indices])
            decoded_sequences.append(joined)
        return decoded_sequences
    else:
        tensor = tensor.permute(1, 0, 2)
        # batch x num_timesteps x num_labels.
        decoder = CTCBeamDecoder(LETTER, log_probs_input=True, beam_width=beam_width, )
        outputs, *args = decoder.decode(tensor)
        for output in outputs:
            # print(output)
            for i in range(len(output[:3])):
                indices = torch.unique_consecutive(output[i], dim=-1)
                indices = [i for i in indices if i != blank_label]
                joined = "".join([NUMBER_DICT.get(i.item(), '') for i in indices])
                print(joined)
            decoded_sequences.append(joined)
        return decoded_sequences


def load_train_test_data(split_ratio: float = 0.8) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    video_dataset = LRNetDataset(DIR)
    train_size = int(split_ratio * len(video_dataset))
    test_size = len(video_dataset) - train_size
    return torch.utils.data.random_split(video_dataset, [train_size, test_size])


def calculate_wer(predicted: List[str], true: List[str]) -> float:
    """

    :param predicted: str
    :param true: str
    :return: float
    """
    # Create a matrix of zeros
    return wer(true, predicted)


# caculate the cer
def calculate_cer(predicted: List[str], true: List[str]) -> float:
    """

    :param predicted: str
    :param true: str
    :return: float
    """
    return cer(true, predicted)
