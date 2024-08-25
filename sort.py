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

# Carico YOLOv8 preaddestrato
model = YOLO('yolov8n.pt')

path = os.path.dirname('data/train/')

for seq in os.listdir(path):

    seq_path = os.path.join(path,seq,'img1')

    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()

    for image_file in image_files:

        image = io.imread(os.path.join(seq_path,image_file))

        results = model(image)

        detections = results[0].boxes.data.numpy()

        detections = detections[np.where(detections[:,5] == 0)]

        detections = detections[:,:5]