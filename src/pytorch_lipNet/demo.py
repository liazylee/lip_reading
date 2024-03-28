import torch

from dataset_loader import LRNetDataLoader
from model import LRModel
from utils import load_train_test_data, ctc_decode_tensor, decode_tensor


def demo():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_point = torch.load('models/1_model_epoch_20_0.99_0.56.pth')
    model = LRModel()
    model_dict = model.state_dict()
    pretained_dict = {k: v for k, v in weights_point.items() if k in model.state_dict()}
    missed_params = [k for k, v in model_dict.items() if k not in weights_point]
    print(f'missed_params: {missed_params}')
    model_dict.update(pretained_dict)
    model.load_state_dict(model_dict)
    _, val_dataset = load_train_test_data()
    val_loader = LRNetDataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=True)
    val_data = iter(val_loader)
    for i in range(10):
        inputs, targets, inputs_lengths, targets_lengths = next(val_data)
        model.eval()
        outputs = model(inputs)
        txt = ctc_decode_tensor(outputs, greedy=False)
        real_text = decode_tensor(targets)
        text = ''.join(txt).strip()
        print(f'real_text: {real_text}, text: {text}')

    # video_npy = np.load('../lipNet/data/s1/bbaszn.npy')
    # video_torch = torch.from_numpy(video_npy).unsqueeze(0).permute(0, 4, 1, 2, 3).float()
    # print(video_torch.shape)
    # model.eval()
    # outputs = model(video_torch)
    # print(outputs.shape)
    # outputs.log_softmax(dim=-1)
    # print(outputs.shape)
    # txt = ctc_decode_tensor(outputs, greedy=False)  # lat wrie it l sire eon
    # real_text = aligment
    # text = ''.join(txt).strip()
    # print(text)


demo()
