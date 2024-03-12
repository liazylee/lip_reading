"""
@author:liazylee
@license: Apache Licence
@time: 05/03/2024 20:09
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
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from config import DIR
from dataset import LRNetDataset
from dataset_loader import LRNetDataLoader
from model import LRModel
from utils import validate, decode_tensor, calculate_wer, calculate_cer, ctc_decode_tensor


def load_train_test_data() -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    video_dataset = LRNetDataset(DIR)
    train_size = int(0.8 * len(video_dataset))
    test_size = len(video_dataset) - train_size
    return torch.utils.data.random_split(video_dataset, [train_size, test_size])


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = load_train_test_data()
    train_loader = LRNetDataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
    val_loader = LRNetDataLoader(val_dataset, batch_size=4, num_workers=4, shuffle=True)
    writer = SummaryWriter()
    # Load model
    model = LRModel().to(device)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        (train_loss_curve, val_loss_curve, train_wer_curve, val_wer_curve,
         train_cer_curve, val_cer_curve) = [], [], [], [], [], []

        for i, (inputs, targets) in enumerate(train_loader):
            train_loss, train_wer, train_cer = 0, 0, 0
            inputs, targets = inputs.to(device), targets.to(device)
            # torch.Size([4, 3, 75, 70, 140]), torch.Size([4, 28])
            optimizer.zero_grad()
            outputs = model(inputs)  # (batch, time, n_class) # torch.Size([4, 75, 28])

            inputs_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1),
                                        dtype=torch.long)  # tensor([75, 75, 75, 75])
            targets_lengths = torch.full(size=(targets.size(1),), fill_value=targets.size(0),
                                         dtype=torch.long)  # torch.Size([29])
            loss = criterion(outputs, targets, inputs_lengths, targets_lengths)
            # input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
            # target_lengths = torch.full(size=(targets.size(1),), fill_value=targets.size(0), dtype=torch.long)
            # loss = criterion(outputs, targets, input_lengths, target_lengths)
            text_outputs = ctc_decode_tensor(outputs, inputs_lengths, greedy=True)
            text_targets = decode_tensor(targets)
            train_wer = calculate_wer(text_outputs, text_targets)
            train_cer = calculate_cer(text_outputs, text_targets)
            train_wer_curve.append(train_wer)
            train_cer_curve.append(train_cer)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 10 == 0:
                writer.add_scalar('train_loss', loss.item(), i)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, i)
                    writer.add_histogram(f'{name}.grad', param.grad, i)
                # print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')
        train_loss_curve.append(train_loss / len(train_dataset))
        writer.add_scalar('train_loss_curve', train_loss / len(train_dataset), epoch)

        # calculate the wer

        # validation
        val_loss = validate(model, criterion, val_loader, device)
        val_loss_curve.append(val_loss)
        writer.add_scalar('val_loss_curve', val_loss, epoch)


if __name__ == '__main__':
    main()
