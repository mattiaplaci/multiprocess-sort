import os
import cv2
import numpy as np
from skimage import io

from ultralytics import YOLO

from filterpy.kalman import KalmanFilter

import lap


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


def iou_matrix(boxes1,boxes2):
    
    # For broadcasting
    boxes1 = np.expand_dims(boxes1,1)
    boxes2 = np.expand_dims(boxes2,0)

    # Intersection bounding boxes
    inter_x1 = np.maximum(boxes1[...,0],boxes2[...,0])
    inter_y1 = np.maximum(boxes1[...,1],boxes2[...,1])
    inter_x2 = np.minimum(boxes1[...,2],boxes2[...,2])
    inter_y2 = np.minimum(boxes1[...,3],boxes2[...,3])

    # Intersection areas
    intersection = np.maximum(0,inter_x2-inter_x1) * np.maximum(0,inter_y2-inter_y1)

    # Bounding boxes areas
    area1 = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    area2 = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])

    # Union areas
    union = area1 + area2 - intersection

    # IOU matrix
    return intersection / union


class KalmanBoxTracker:

    count = 0

    def __init__(self,box):

        self.kf = KalmanFilter(dim_x=7,dim_z=4)

        # Costant velocity model:
        # state transiction matrix F
        self.kf.F = np.array([ [1,0,0,0,1,0,0],
                               [0,1,0,0,0,1,0],
                               [0,0,1,0,0,0,1],
                               [0,0,0,1,0,0,0],
                               [0,0,0,0,1,0,0],
                               [0,0,0,0,0,1,0],
                               [0,0,0,0,0,0,1] ])
        # measurement matrix H
        self.kf.H = np.array([ [1,0,0,0,0,0,0],
                               [0,1,0,0,0,0,0],
                               [0,0,1,0,0,0,0],
                               [0,0,0,1,0,0,0] ])
        # initial state
        self.kf.x = box
        # state covariance matrix P
        self.kf.P *= 10
        self.kf.P[4:,4:] *= 1000
        # process noise matrix Q
        self.kf.Q[4:,4:] *= 0.01
        self.kf.Q[-1,-1] *= 0.01
        # measurement noise matrix R
        self.kf.R[2:,2:] *= 10

        self.id = KalmanBoxTracker.count

        KalmanBoxTracker.count += 1


    def predict(self):
        pass

    def update(self):
        pass

    def get_state(self):

        return self.kf.x
    

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

        # Discard detections with score less then 50%
        detections = detections[np.where(detections[:,4] >= 0.5)]

        # Delete class_id and score columns
        detections = detections[:,:4]
