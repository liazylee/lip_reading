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

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CORPUS_LETTER


class LRNetDataset(Dataset):
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.video_list = self.load_video()
        self.alignments = self.load_alignments_dict()

    def load_video(self) -> List[str]:
        video_list = []
        for root, dir, files in os.walk(self.dir):
            for file in files:
                if file.endswith('.npy'):
                    video_list.append(os.path.join(root, file))
        return video_list

    def load_alignments_dict(self) -> dict:
        # load alignments
        alignments = glob.glob(self.dir + '/**/*.align', recursive=True)
        print(f'the total alignments files is {len(alignments)}')
        logging.info(f'the total alignments files is {len(alignments)}')
        alignments_dict = {}
        for align in alignments:
            alignments_dict[align.split('/')[-1].split('.')[0]] = self.load_alignments(align)
        return alignments_dict

    def load_alignments(self, path: str) -> np.array:
        with open(path, 'r') as f:
            lines = f.readlines()
            tokens = []
            for line in lines:
                line = line.split()
                if line[2] != 'sil':
                    tokens.append(line[2])
            tokens_np = np.array([CORPUS_LETTER.get(token, 0) for token in tokens])
            return tokens_np

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        video_frames = np.load(self.video_list[idx])
        alignments = self.alignments.get(self.video_list[idx].split('/')[-1].split('.')[0])

        return torch.from_numpy(video_frames).float(), torch.from_numpy(alignments).int()
