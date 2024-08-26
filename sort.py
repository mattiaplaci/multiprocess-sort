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

# Pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

path = os.path.dirname('data/train/')

# Cicle through the train sequences
for seq in os.listdir(path):

    seq_path = os.path.join(path,seq,'img1')

    # Frame list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()

    # Cicle through sequence's frames
    for image_file in image_files:

        image = io.imread(os.path.join(seq_path,image_file))

        # Detect on a single frame
        results = model(image)

        detections = results[0].boxes.data.numpy()

        # Discard detections different from pedestrians
        detections = detections[np.where(detections[:,5] == 0)]

        # Delete class_id column
        detections = detections[:,:5]