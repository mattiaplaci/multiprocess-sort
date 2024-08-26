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


# Converts format [x1,y1,x2,y2] to [x,y,s,r]
def convert_xyxy_to_xysr(boxes):
    
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]

    # Ratio
    r = w / h
    # Scale (area)
    s = w * h
    # Center x coordinate
    x = boxes[:,0] + w/2
    # Center y coordinate
    y = boxes[:,1] + h/2

    boxes[:,0] = x
    boxes[:,1] = y
    boxes[:,2] = s
    boxes[:,3] = r

    return boxes

# Converts format [x,y,s,r] to [x1,y1,x2,y2]
def convert_xysr_to_xyxy(boxes):

    w = np.sqrt(boxes[:,2] * boxes[:,3])
    h = w / boxes[:,3]

    # Top-left x coordinate
    x1 = boxes[:,0] - w/2
    # Top-left y coordinate
    y1 = boxes[:,1] - h/2
    # Bottom-right x coordinate
    x2 = boxes[:,0] + w/2
    # Bottom-right y coordinate
    y2 = boxes[:,1] + h/2

    boxes[:,0] = x1
    boxes[:,1] = y1
    boxes[:,2] = x2
    boxes[:,3] = y2

    return boxes


# Pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

path = os.path.dirname('data/train/')

# Cicle through the train sequences
for seq in os.listdir(path):

    seq_path = os.path.join(path,seq,'img1')

    # Frames list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()

    # Cicle through sequence's frames
    for image_file in image_files:

        image = io.imread(os.path.join(seq_path,image_file))

        # Detect on a single frame
        results = model(image)

        # Detections in [x1,y1,x2,y2] format
        detections = results[0].boxes.data.numpy()

        # Discard detections different from pedestrians
        detections = detections[np.where(detections[:,5] == 0)]

        # Delete class_id column
        detections = detections[:,:5]