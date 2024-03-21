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
from torchinfo import summary

from config import CORPUS_size


# summary of the model


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

        # New convolutional layer
        self.conv4 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.flatten = nn.Flatten(2, 4)
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=128, batch_first=True, num_layers=2,
                             bidirectional=True)
        # self.initialize_lstm_forget_gate_bias(self.lstm1)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=128 * 2, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, CORPUS_size + 1, )

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        # print(x.shape, 'conv3')
        x = self.pool4(self.relu4(self.conv4(x)))
        # print(x.shape, 'conv4')
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = self.flatten(x)
        # print(x.size(), 'flatten')
        x = x.permute(1, 0, 2)
        # print(x.size(), 'flatten')
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        return x

    # def initialize_lstm_forget_gate_bias(self, lstm):
    #     """
    #     Initialize the bias for the forget gate in the LSTM cell
    #     """
    #     for names in lstm._all_weights:
    #         for name in filter(lambda n: "bias" in n, names):
    #             bias = getattr(lstm, name)
    #             n = bias.size(0)
    #             start, end = n // 4, n // 2
    #             bias.data[start:end].fill_(1.0)  # Initialize the forget gate bias to 1.0


if __name__ == '__main__':
    model = LRModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    # print(model.lstm1)
    print(summary(model, input_size=(2, 3, 75, 70, 140)))
