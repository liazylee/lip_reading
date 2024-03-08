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
# summary of the model
from torchsummary import summary

from config import LETTER_SIZE


class LRModel(nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.flatten = nn.Flatten(2, 2)
        self.lstm1 = nn.LSTM(input_size=17408, hidden_size=128, num_layers=2, batch_first=True,
                             bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(input_size=128 * 2, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(128 * 2, LETTER_SIZE + 1)  # Output layer

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        # torch.Size([2, 1, 75, 70, 140])
        # torch.Size([2, 128, 75, 35, 70])
        x = self.pool2(self.relu2(self.conv2(x)))
        # torch.Size([2, 256, 75, 17, 35])
        x = self.pool3(self.relu3(self.conv3(x)))
        # torch.Size([2, 128, 75, 8, 17])
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        torch.Size([75, 2, 128, 8, 17])
        x = x.view(x.size(0), x.size(1), -1)
        x = self.flatten(x)
        # torch.Size([75, 2, 17408])
        print(x.size(), 'flatten')
        print(x.shape, 'flatten')
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    model = LRModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    print(summary(model, (3, 75, 70, 140)))  # Fix this code in LSTM
