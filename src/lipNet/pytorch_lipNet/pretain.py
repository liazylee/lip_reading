# this script is used to extract the mouth region from the video
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def mouth_extractor(file_path:str):
    """
    Extract the mouth from the video save as npy file
    :param file_path:
    :return:
    """
    if file_path.endswith('.mpg') or file_path.endswith('.avi') or file_path.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()
        frames=[]
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                sys.exit()
            # get the face region

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                mouth_roi = frame[y + int(h/2):y + h, x:x + w]
                # resize the mouth region to 70x140
                mouth_roi = cv2.resize(mouth_roi, (140, 70))
                mouth_roi=rgb_to_grayscale(mouth_roi)
                frames.append(mouth_roi)  #

        cap.release()
        mean_frame = torch.stack(frames).mean(dim=0)
        std= torch.stack(frames).std(dim=0)
        frames = torch.stack(frames)
        frames = (frames - mean_frame) / std

        # show the mouth region
        plt.imshow(make_grid(frames, nrow=10).permute(1, 2, 0))
        # Save the mouth region as a numpy file
        # np.save(file_path.replace('.mpg', '_mouth.npy'), mouth_roi)
        # # Release the video capture object

        # # Close all the windows
        cv2.destroyAllWindows()



def rgb_to_grayscale(frame:np.array)->torch.Tensor:

    # Convert frame to PyTorch tensor
    frame_tensor = torch.from_numpy(frame).float()
    # Convert RGB to grayscale
    grayscale_frame_tensor = frame_tensor.mean(dim=2).unsqueeze(0)
    return grayscale_frame_tensor
if __name__ == '__main__':
    mouth_extractor('../data/s10/bbaczp.mpg')