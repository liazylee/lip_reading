"""
@author:liazylee
@license: Apache Licence
@time: 05/03/2024 19:52
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
import torch
import torch.nn as nn

from src.lipNet.pytorch_lipNet.config import LETTER_SIZE
from  torch.nn import functional as F

class LRModel(nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 75, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.flatten = nn.Flatten()

        self.lstm1 = nn.LSTM(input_size=75 * 2 * 5, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(input_size=128 * 2, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(128 * 2, LETTER_SIZE + 1)  # Output layer

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        return x





def CTCLoss(y_true, y_pred):
    batch_len = y_true.size(0)
    input_length = y_pred.size(1)
    label_length = y_true.size(1)

    input_length = input_length * torch.ones(batch_len, dtype=torch.int64)
    label_length = label_length * torch.ones(batch_len, dtype=torch.int64)

    loss = nn.CTCLoss(blank=0, zero_infinity=True)(y_pred, y_true, input_length, label_length)
    return loss

