import os
import cv2
import numpy as np
from skimage import io

from ultralytics import YOLO

def show_video(seq_name,fps):

    path = os.path.join('data/train',seq_name,'img1')

    image_files = [f for f in os.listdir(path)]
    image_files.sort()

    for image in image_files:
        image = io.imread(os.path.join(path,image))
        cv2.imshow(seq_name,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(int(1000/fps))

    cv2.destroyAllWindows()