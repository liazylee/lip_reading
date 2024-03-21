# this script is used to extract the mouth region from the video
import logging
import sys
import glob
import time

from tqdm import tqdm

from config import DIR
from utils import mouth_extractor, timmer
import multiprocessing as mp

# @timmer
def pretain(dir:str)->None:
    """
    find all the video and extra the mouth region
    :param dir:
    :return:
    """
    time1=time.time()
    video_list=glob.glob(dir+'/**/*.mpg',recursive=True)
    # use multi-thread to extract the mouth region
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(mouth_extractor,video_list)
    print(f'Pretain took {time.time()-time1} seconds')
    # for video in tqdm(video_list):
    #     mouth_extractor(video)




if __name__ == '__main__':
    if len(sys.argv) != 2:
        dir=DIR
    else:
        dir=sys.argv[1]
    pretain(dir)
    logging.info('Pretain finished')