"""
@author:liazylee
@license: Apache Licence
@time: 05/03/2024 20:07
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
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import LRNetDataset


class LRNetDataLoader(DataLoader):
    def __init__(self, dataset: LRNetDataset, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True):
        super(LRNetDataLoader, self).__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                              collate_fn=collate_fn)


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor,
torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    inputs_lengths = torch.tensor([len(seq) for seq in inputs])
    targets_lengths = torch.tensor([len(seq) for seq in targets])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    # (batch, channel, time, height, width) -> (batch, time, channel, height, width)
    # (batch,time,height,width)->add 1 channel dimension (batch, time, channel, height, width)
    padded_inputs = padded_inputs.permute(0, 4, 1, 2, 3)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets, inputs_lengths, targets_lengths
