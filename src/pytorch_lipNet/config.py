"""
@author:liazylee
@license: Apache Licence
@time: 05/03/2024 19:19
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

DIR = '/home/liazylee/jobs/python/AI/lip_reading/src/lipNet/data/'  # absolute path to the data directory
# DIR = '/Users/zhenyili/research project/src/lipNet/data'  # absolute path to the data directory
LETTER_DICT = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6,
               'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
               'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18,
               's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, ' ': 27, }

NUMBER_DICT = {27: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f',
               7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
               13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r',
               19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

LETTER = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
LETTER_STR = ' abcdefghijklmnopqrstuvwxyz '
MODEL_PATH = '/home/liazylee/jobs/python/AI/lip_reading/src/pytorch_lipNet/models/1_model_epoch_20_0.99_0.56.pth'
LETTER_SIZE = len(LETTER_DICT) + 1
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.0001
RANDOM_SEED = 12
