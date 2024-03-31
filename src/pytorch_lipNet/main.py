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
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import functional as F  # noqa
from tqdm import tqdm

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED, MODEL_PATH
from dataset_loader import LRNetDataLoader
from model import LRModel
from utils import validate, decode_tensor, load_train_test_data, calculate_wer, calculate_cer, ctc_decode_tensor


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
    torch.backends.cudnn.deterministic = True  # Set for testing
    iter_num = 0
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        iter_num = MODEL_PATH.split('/')[-1].split('_')[0]
        if not iter_num.isdigit():
            iter_num = 0
        else:
            iter_num = int(iter_num) + 1
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    criterion = nn.CTCLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS + 1):
        (train_loss_curve, val_loss_curve, train_wer_curve, val_wer_curve,
         train_cer_curve, val_cer_curve) = [], [], [], [], [], []
        print(f'Epoch {epoch}')
        train_loss, train_wer, train_cer = 0, 0, 0
        for i, (inputs, targets, inputs_lengths, targets_lengths) in enumerate(tqdm(train_loader, leave=False,
                                                                                    desc="Train", ncols=75)):

            inputs, targets = (inputs.float()).to(device), (targets.int()).to(device)
            inputs_lengths, targets_lengths = (inputs_lengths.int()).to(device), (targets_lengths.int()).to(device)
            optimizer.zero_grad()
            model.train()
            outputs = model(inputs)
            with torch.backends.cudnn.flags():
                loss = criterion(outputs, targets, inputs_lengths, targets_lengths)
            loss.backward()
            optimizer.step()
            text_outputs: List[str] = ctc_decode_tensor(outputs)
            text_targets: List[str] = decode_tensor(targets)
            train_wer_curve.append(calculate_wer(text_outputs, text_targets))
            train_cer_curve.append(calculate_cer(text_outputs, text_targets))
            train_loss += loss.item()
            if i % 20 == 0:
                writer.add_scalar('train_loss', train_loss, epoch * len(train_loader) + i)
                print(f'Epoch {epoch}, Batch {i}, loss: {loss.item()}')
                print(f'text_outputs: {text_outputs}, \n '
                      f'text_targets: {text_targets}')
                print(
                    f'WER: {calculate_wer(text_outputs, text_targets)},\n'
                    f' CER: {calculate_cer(text_outputs, text_targets)}')
        train_loss_curve.append(train_loss)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_wer', np.mean(train_wer_curve), epoch)
        writer.add_scalar('train_cer', np.mean(train_cer_curve), epoch)

        # save model
        if epoch % 20 == 0:
            if not os.path.exists('models'):
                os.makedirs('models')
            print(f'saving model at epoch {epoch}')

            torch.save(model.state_dict(), f'./models/{iter_num}_model_epoch_{epoch}_'
                                           f'{round(np.mean(train_wer_curve), 2)}_'
                                           f'{round(np.mean(train_cer_curve), 2)}.pth')
        print(f'begin validation')
        val_loss, val_wer, val_cer = validate(model, criterion, val_loader,
                                              device)  # This function should return validation loss, WER, and CER
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_wer', val_wer, epoch)
        writer.add_scalar('val_cer', val_cer, epoch)
        writer.close()


if __name__ == '__main__':
    main()
