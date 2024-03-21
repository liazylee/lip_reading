from typing import Tuple, Dict

DIR = '/home/liazylee/jobs/python/AI/lip_reading/src/lipNet/data/'  # absolute path to the data directory
# DIR = '/Users/zhenyili/research project/src/lipNet/data'  # absolute path to the data directory
MODEL_PATH = ''
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.0001
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


CORPUS_LETTER, LETTER_CORPUS = get_corpus()
CORPUS_size = len(CORPUS_LETTER)
