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
from torch.utils.data import DataLoader

from dataset import LRNetDataset


class LRNetDataLoader(DataLoader):
    def __init__(self, dataset: LRNetDataset, batch_size: int = 4, num_workers: int = 4, shuffle: bool = True):
        super(LRNetDataLoader, self).__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                              collate_fn=collate_fn)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    # (batch, channel, time, height, width) -> (batch, time, channel, height, width)
    padded_inputs = padded_inputs.permute(0, 4, 1, 2, 3)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets,
