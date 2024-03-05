"""
@author:liazylee
@license: Apache Licence
@time: 05/03/2024 17:50
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
import glob
import logging
import os
from typing import List, Tuple

from torch.utils.data import Dataset
import numpy as np

from src.lipNet.pytorch_lipNet.config import LETTER_DICT


class LRNetDataset(Dataset):
    def __init__(self, dir:str)->None:
        self.dir = dir
        self.data = self.load_data()

    def load_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        data = []
        # load alignments
        alignments = glob.glob(self.dir + '/**/*.align', recursive=True)
        alignments_dict = {}
        for align in alignments:
            alignments_dict[align.split('/')[-1].split('.')[0]] = self.load_alignments(align)
        for root, dir, files in os.walk(self.dir):
            for file in files:
                if file.endswith('.npy'):
                    video_frames = np.load(os.path.join(root, file))
                    filename = file.split('.')[0]
                    if filename in alignments_dict:
                        alignments = alignments_dict[filename]
                    else:
                        logging.warning(f'No alignment found for {filename}')
                        alignments = ''
                    data.append((video_frames, alignments))
        return data
    def load_alignments(self,path: str) -> np.array:
        with open(path, 'r') as f:
            lines = f.readlines()
            tokens = ''
            for line in lines:
                line = line.split()
                if line[2] != 'sil':
                    tokens = tokens + ' ' + line[2]
            tokens_np = np.array([LETTER_DICT[c] for c in tokens])
            return tokens_np

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (video_frames,alignments) = self.data[idx]
        return video_frames,alignments

