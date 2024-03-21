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
from pyctcdecode import build_ctcdecoder  # noqa
from word_beam_search import WordBeamSearch  # noqa

from config import DIR, LETTER_CORPUS, CORPUS_size
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
        frames_tensor = ((frames - mean) / std).astype(np.float16)
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
            outputs = model(inputs)
            outputs = outputs.permute(1, 0, 2)  # (time, batch, n_class)
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
        LETTER_CORPUS (dict): Dictionary mapping numerical indices to characters.
    Returns:
        str: The decoded string.
    """
    tensors = tensor.tolist()
    result = []
    for tensor in tensors:
        result.append(reduce(lambda x, y: x + ' ' + y,
                             [LETTER_CORPUS.get(i, '') if i < (CORPUS_size + 1)
                              else '' for i in tensor]))
    return result


# def ctc_decode_tensor(tensor: torch.Tensor,
#                       greedy: bool = True, beam_width: int = 10) -> List[str]:
#     """
#     Decodes a tensor into a string using a mapping dictionary.
#     Args:
#         tensor:
#         input_lengths:
#         number_dict:
#         greedy:
#
#     Returns: str
#
#     """
#     blank_label = len(NUMBER_DICT) + 1
#     decoded_sequences = []
#     if greedy:
#         # probabilities = torch.log_softmax(tensor, dim=-1)  # (time, batch, n_class)
#
#         for i in range(tensor.shape[1]):
#             sequence = tensor[:, i, :].argmax(-1)  # (time, n_class) -> (time,)
#             decoded_sequence = ""
#             prev_label = None
#             for label in sequence:
#                 # print(label, type(label), label.item())
#                 if label != prev_label and label != blank_label:
#                     decoded_sequence += NUMBER_DICT.get(label.item(), '')
#                 prev_label = label
#             decoded_sequences.append(decoded_sequence)
#         return decoded_sequences
#     else:
#         # beam search
#         probabilities = tensor.log_softmax(dim=-1)
#         input_lengths = torch.full(size=(tensor.size(1),), fill_value=tensor.size(0), dtype=torch.long)
#         decoder = build_ctcdecoder(NUMBER_DICT, alpha=0.5, beta=0.5,
#                                    )
#         decoded, _ = decoder.decode(probabilities)
def apply_word_beam_search(mat, corpus, chars, word_chars) -> Tuple[List[str], List[str]]:
    """Decode using word beam search. Result is tuple, first entry is label string, second entry is char string."""
    T, B, C = mat.shape

    # decode using the "Words" mode of word beam search with beam width set to 25 and add-k smoothing to 0.0
    assert len(chars) + 1 == C

    wbs = WordBeamSearch(40, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), word_chars.encode('utf8'))
    # convert to numpy array
    mat_numpy = mat.cpu().detach().numpy()
    label_str = wbs.compute(mat_numpy)

    # result is string of labels terminated by blank
    char_str = []
    for curr_label_str in label_str:
        s = ''
        for label in curr_label_str:
            s += chars[label]  # map label to char
        char_str.append(s)

    return label_str, char_str


def ctc_decode_tensor(tensor: torch.Tensor, greedy: bool = True, ) -> List[str]:
    """
    Decodes a tensor into a string using a mapping dictionary.
    Args:
        tensor: (time, batch, n_class)
        greedy:
        beam_width:
    Returns: str
    """
    sequences = []
    if greedy:
        for i in range(tensor.shape[1]):
            sequences.append(greedy_decoder(tensor[:, i, :]))
        return sequences

    else:
        # beam search
        pass


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels: dict, blank: int = 0) -> None:
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = " ".join([self.labels.get(i.item(), '') for i in indices])
        joined = ' '.join(joined.split())
        return joined


greedy_decoder = GreedyCTCDecoder(LETTER_CORPUS)


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
