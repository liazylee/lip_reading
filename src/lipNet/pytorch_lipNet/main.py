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

from src.lipNet.pytorch_lipNet.config import DIR
from src.lipNet.pytorch_lipNet.dataset_loader import LRNetDataLoader
from src.lipNet.pytorch_lipNet.model import LRModel
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
def main():

    data_loader = LRNetDataLoader(DIR).get_data_loader()
    writer = SummaryWriter()
    # Load model
    model = LRModel()
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_curve,val_loss_curve,train_wer_curve,val_wer_curve = [],[],[],[]
    for epoch in range(10):
        train_loss = 0
        val_loss = 0
        train_wer = 0
        val_wer = 0
        for i, (inputs, targets) in enumerate(data_loader):
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
        train_loss_curve.append(train_loss / len(data_loader))
        val_loss_curve.append(val_loss / len(data_loader))
        train_wer_curve.append(train_wer / len(data_loader))
        val_wer_curve.append(val_wer / len(data_loader))
        # add log
        writer.add_scalar('train_loss', train_loss / len(data_loader), epoch)
        writer.add_scalar('val_loss', val_loss / len(data_loader), epoch)
        writer.add_scalar('train_wer', train_wer / len(data_loader), epoch)
        writer.add_scalar('val_wer', val_wer / len(data_loader), epoch)
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./log'):
        os.mkdir('./log')
    params = list(model.parameters())
    torch.save(model.state_dict(), './model/lRnet.pth')
    writer.export_scalars_to_json("./log/all_scalars.json")
    writer.close()