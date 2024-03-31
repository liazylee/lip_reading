from typing import Tuple, Dict

DIR = '/home/liazylee/jobs/python/AI/lip_reading/src/lipNet/data/'  # absolute path to the data directory
# DIR = '/Users/zhenyili/research project/src/lipNet/data'  # absolute path to the data directory
MODEL_PATH = ''
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def get_corpus() -> Tuple[Dict[str, int], Dict[int, str]]:
    """

    """
    CORPUS_LETTER = {}
    LETTER_CORPUS = {}
    with open('text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            for i, word in enumerate(line):
                CORPUS_LETTER[word] = i + 1
                LETTER_CORPUS[i + 1] = word

    return CORPUS_LETTER, LETTER_CORPUS


CORPUS_LETTER = {'p': 6, 'q': 2, 'f': 3, 'x': 4, 's': 11, 'h': 7, 'a': 8,
                 'now': 9, 'six': 10, 'u': 12, 'at': 13, 'set': 14, 'n': 15,
                 'with': 16, 'zero': 17, 'three': 18, 'five': 19, 'nine': 20,
                 'lay': 21, 'in': 22, 'soon': 23, 'green': 24, 'please': 25,
                 'r': 26, 'seven': 27, 'z': 28, 't': 29, 'white': 30, 'g': 31,
                 'eight': 32, 'b': 33, 'four': 34, 'one': 35, 'blue': 36,
                 'c': 37, 'e': 38, 'j': 39, 'm': 40, 'place': 41, 'two':
                     42, 'k': 43, 'v': 44, 'o': 45, 'l': 46, 'd': 47,
                 'red': 48, 'i': 49, 'again': 50, 'y': 51, 'by': 52, 'bin': 53, }

LETTER_CORPUS = {1: 'p', 2: 'q', 3: 'f', 4: 'x', 5: 's', 6: 'p', 7: 'h', 8: 'a',
                 9: 'now', 10: 'six', 11: 's', 12: 'u', 13: 'at', 14: 'set', 15:
                     'n', 16: 'with', 17: 'zero', 18: 'three', 19: 'five', 20:
                     'nine', 21: 'lay', 22: 'in', 23: 'soon', 24: 'green',
                 25: 'please', 26: 'r', 27: 'seven', 28: 'z', 29: 't',
                 30: 'white', 31: 'g', 32: 'eight', 33: 'b', 34: 'four',
                 35: 'one', 36: 'blue', 37: 'c', 38: 'e', 39: 'j', 40: 'm',
                 41: 'place', 42: 'two', 43: 'k', 44: 'v', 45: 'o', 46: 'l',
                 47: 'd', 48: 'red', 49: 'i', 50: 'again', 51: 'y', 52: 'by', 53: 'bin', }

LETTER = ['p', 'q', 'f', 'x', 's', 'h', 'a',
          'now', 'six', 'u', 'at', 'set', 'n',
          'with', 'zero', 'three', 'five', 'nine',
          'lay', 'in', 'soon', 'green', 'please',
          'r', 'seven', 'z', 't', 'white', 'g',
          'eight', 'b', 'four', 'one', 'blue',
          'c', 'e', 'j', 'm', 'place', 'two', 'k',
          'v', 'o', 'l', 'd', 'red', 'i', 'again',
          'y', 'by', 'bin']
CORPUS_size = len(LETTER)
MOUTH_H = 35
MOUTH_W = 70
