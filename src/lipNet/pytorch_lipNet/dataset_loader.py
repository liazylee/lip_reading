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
from torch.nn.utils.rnn import pad_sequence

from src.lipNet.pytorch_lipNet.dataset import LRNetDataset
from torch.utils.data import DataLoader
import torch


class LRNetDataLoader(DataLoader):
    def __init__(self, dataset: LRNetDataset, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True):
        super(LRNetDataLoader, self).__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                              collate_fn=collate_fn)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets,
