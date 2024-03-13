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

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED
from dataset_loader import LRNetDataLoader
from model import LRModel
from utils import validate, decode_tensor, calculate_wer, calculate_cer, ctc_decode_tensor, load_train_test_data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)  # Set for testing
    torch.cuda.manual_seed_all(RANDOM_SEED)
    train_dataset, val_dataset = load_train_test_data()
    train_loader = LRNetDataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader = LRNetDataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    writer = SummaryWriter()
    # Load model
    model = LRModel().to(device)
    criterion = nn.CTCLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        (train_loss_curve, val_loss_curve, train_wer_curve, val_wer_curve,
         train_cer_curve, val_cer_curve) = [], [], [], [], [], []
        print(f'Epoch {epoch}')
        train_loss, train_wer, train_cer = 0, 0, 0
        for i, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            # torch.Size([4, 3, 75, 70, 140]), torch.Size([4, 28])
            optimizer.zero_grad()
            outputs = model(inputs)  # (batch, time, n_class) # torch.Size([4, 75, 28])
            outputs = outputs.permute(1, 0, 2)  # (time, batch, n_class)
            outputs = F.log_softmax(outputs, dim=2)  # (time, batch, n_class)
            inputs_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0),
                                        dtype=torch.long).to(device)  # tensor([75, 75, 75, 75])
            targets_lengths = torch.full(size=(targets.size(0),), fill_value=targets.size(1),
                                         dtype=torch.long).to(device)  # tensor([33, 33, 33, 33])
            loss = criterion(outputs, targets, inputs_lengths, targets_lengths)
            text_outputs: str = ctc_decode_tensor(outputs, )
            print(f'text_outputs: {text_outputs}')
            text_targets: str = decode_tensor(targets)
            print(f'text_targets: {text_targets}')
            train_wer += calculate_wer(text_outputs, text_targets)
            train_cer += calculate_cer(text_outputs, text_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 10 == 0:
                writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + i)
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param, epoch * len(train_loader) + i)
                #     writer.add_histogram(f'{name}.grad', param.grad, epoch * len(train_loader) + i)
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')
        num_batches = len(train_loader)
        writer.add_scalar('epoch_train_loss', train_loss / num_batches, epoch)
        writer.add_scalar('epoch_train_wer', train_wer / num_batches, epoch)
        writer.add_scalar('epoch_train_cer', train_cer / num_batches, epoch)

        val_loss, val_wer, val_cer = validate(model, criterion, val_loader,
                                              device)  # This function should return validation loss, WER, and CER
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_wer', val_wer, epoch)
        writer.add_scalar('val_cer', val_cer, epoch)
        # train_loss_curve.append(train_loss / len(train_dataset))
        # writer.add_scalar('train_loss_curve', train_loss / len(train_dataset), epoch)
        # # calculate the wer
        # # validation
        # val_loss = validate(model, criterion, val_loader, device)
        # val_loss_curve.append(val_loss)
        # writer.add_scalar('val_loss_curve', val_loss, epoch)
        writer.close()


if __name__ == '__main__':
    main()
