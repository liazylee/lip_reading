# this script is used to extract the mouth region from the video
import logging
import sys
import glob

from tqdm import tqdm

from src.lipNet.pytorch_lipNet.config import DIR
from src.lipNet.pytorch_lipNet.utils import mouth_extractor, timmer


@timmer
def pretain(dir:str)->None:
    """
    find all the video and extra the mouth region
    :param dir:
    :return:
    """

    video_list=glob.glob(dir+'/**/*.mpg',recursive=True)
    for i in tqdm(range(len(video_list)),desc='Extracting mouth region from video',ncols=100):
        mouth_extractor(video_list[i])
        


if __name__ == '__main__':
    if len(sys.argv) != 2:
        dir=DIR
    else:
        dir=sys.argv[1]
    pretain(dir)
    logging.info('Pretain finished')