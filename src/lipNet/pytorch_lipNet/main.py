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
import os
from typing import Tuple

from config import DIR
from dataset_loader import LRNetDataLoader
from dataset import LRNetDataset
from src.lipNet.pytorch_lipNet.model import LRModel
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

def load_train_test_data()->Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    video_dataset = LRNetDataset(DIR)
    train_size = int(0.8 * len(video_dataset))
    test_size = len(video_dataset) - train_size
    return torch.utils.data.random_split(video_dataset, [train_size, test_size])

def main():
    # use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = load_train_test_data()
    train_loader = LRNetDataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    val_loader = LRNetDataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True)
    writer = SummaryWriter()
    # Load model
    model = LRModel().to(device)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_curve,val_loss_curve,train_wer_curve,val_wer_curve = [],[],[],[]
    for epoch in range(10):
        train_loss = 0
        val_loss = 0
        train_wer = 0
        val_wer = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.permute(1, 0, 2) # (time, batch, n_class)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            target_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1), dtype=torch.long)
            loss = criterion(outputs, targets, input_lengths, target_lengths)

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
        # validation

    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./log'):
        os.mkdir('./log')
    params = list(model.parameters())
    torch.save(model.state_dict(), './model/lRnet.pth')
    writer.export_scalars_to_json("./log/all_scalars.json")
    writer.close()

if __name__ == '__main__':
    main()