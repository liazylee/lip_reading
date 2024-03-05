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
from src.lipNet.pytorch_lipNet.dataset import LRNetDataset
from torch.utils.data import DataLoader
import torch

class LRNetDataLoader:
    def __init__(self, dir: str, batch_size: int=200, num_workers: int=4, shuffle: bool=True):
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def get_data_loader(self) -> DataLoader:
        dataset = LRNetDataset(self.dir)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

