import glob
import time

import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

MOUTH_W, MOUTH_H = 140, 70
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
FACEMESH_LIPS = mp_face_mesh.FACEMESH_LIPS


def pre_process_mouth_extractor(path: str, mouth_size=(MOUTH_W, MOUTH_H)) -> None:
    base_path = path.split('.')[0]
    # if not os.path.exists(base_path + '.npy'):
    if path.endswith('.mpg'):
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
                if not results.multi_face_landmarks:
                    continue
                for face_landmarks in results.multi_face_landmarks:

            cap.release()

        # show the 35th frame
        plt.imshow(frames[0])
        plt.show()
        return None


DIR = '/home/liazylee/jobs/python/AI/lip_reading/src/lipNet/data/'  # absolute path

video_list = glob.glob(DIR + '/**/*.mpg', recursive=True)
# for video in tqdm.tqdm(video_list, total=len(video_list)):
#     pre_process_mouth_extractor(video)


if __name__ == '__main__':
    time1 = time.time()
    floder_set = set()
    for video in tqdm(video_list, total=len(video_list)):
        #     pre_process_mouth_extractor(video)
        # print(f'Pretain took {time.time() - time1} seconds')
        # print('Pretain finished')
        filePath = video.split('/')[-2]
        if filePath not in floder_set:
            floder_set.add(filePath)
            pre_process_mouth_extractor(video)
            # file_npy = video.split('/')[-1].split('.')[0] + '.npy'
            # np.load(DIR + filePath + '/' + file_npy)
            # # show the 35th frame
            # plt.imshow(np.load(DIR + filePath + '/' + file_npy)[35][:, :, 0])
            # plt.show()
