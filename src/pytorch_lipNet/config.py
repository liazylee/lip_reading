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
LETTER_DICT = {'a': 97, 'b': 98, 'c': 99, 'd': 100,
               'e': 101, 'f': 102, 'g': 103,
               'h': 104, 'i': 105, 'j': 106,
               'k': 107, 'l': 108, 'm': 109,
               'n': 110, 'o': 111, 'p': 112,
               'q': 113, 'r': 114, 's': 115,
               't': 116, 'u': 117, 'v': 118,
               'w': 119, 'x': 120, 'y': 121,
               'z': 122, ' ': 32}

NUMBER_DICT = {97: 'a', 98: 'b', 99: 'c', 100: 'd',
               101: 'e', 102: 'f', 103: 'g',
               104: 'h', 105: 'i', 106: 'j',
               107: 'k', 108: 'l', 109: 'm',
               110: 'n', 111: 'o', 112: 'p',
               113: 'q', 114: 'r', 115: 's',
               116: 't', 117: 'u', 118: 'v',
               119: 'w', 120: 'x', 121: 'y',
               122: 'z', 32: ' ', 0: ''}
LETTER_SIZE = len(LETTER_DICT)
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001
