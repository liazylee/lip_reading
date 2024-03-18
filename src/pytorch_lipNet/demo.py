import numpy as np
import torch

from model import LRModel
from utils import ctc_decode_tensor


def test():
    weights_point = torch.load('models/model_epoch_100_0.99_-0.29.pth')
    model = LRModel()
    model_dict = model.state_dict()
    pretained_dict = {k: v for k, v in weights_point.items() if k in model.state_dict()}
    missed_params = [k for k, v in model_dict.items() if k not in weights_point]
    print(f'missed_params: {missed_params}')
    model_dict.update(pretained_dict)
    model.load_state_dict(model_dict)

    video_npy = np.load('../lipNet/data/s1/bbaszn.npy')
    video_torch = torch.from_numpy(video_npy).unsqueeze(0).permute(0, 4, 1, 2, 3).float()
    print(video_torch.shape)
    model.eval()
    outputs = model(video_torch)
    print(outputs.shape)
    outputs.log_softmax(dim=-1)
    print(outputs.shape)
    txt = ctc_decode_tensor(outputs)
    text = ''.join(txt).strip()
    print(text)


test()
