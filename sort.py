import os
import cv2
import numpy as np
from skimage import io

from ultralytics import YOLO

from filterpy.kalman import KalmanFilter

import lap
from scipy.optimize import linear_sum_assignment


def show_video(seq_name,fps):

    path = os.path.join('data/train',seq_name,'img1')

    image_files = [f for f in os.listdir(path)]
    image_files.sort()

    for frame in image_files:
        frame = io.imread(os.path.join(path,frame))
        cv2.imshow(seq_name,cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(int(1000/fps))

    cv2.destroyAllWindows()


class YOLOv8Detector:

    def __init__(self,score_threshold=0.5):

        # Pretrained YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.score_threshold = score_threshold

    def get_detections(self,image):

        # Detect on a single frame
        results = self.model(image)

        # Detections in [x1,y1,x2,y2] format
        detections = results[0].boxes.data.numpy()

        # Discard detections different from pedestrians
        detections = detections[np.where(detections[:,5] == 0)]

        # Discard detections with score less then score_threshold
        detections = detections[np.where(detections[:,4] >= self.score_threshold)]

        # Delete class_id and score columns
        detections = detections[:,:4]

        return detections


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
    


def calculate_iou_matrix(boxes1,boxes2):
    
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


def associate_detections_to_trackers(detections,trackers,iou_threshold=0.3):
    
    # Empty tracker list
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), detections, np.empty((0,4))
    
    # IOU matrix
    iou_matrix = calculate_iou_matrix(detections,trackers)

    # Assignment problem solution
    if min(iou_matrix.shape) > 0:
        row, col = linear_sum_assignment(iou_matrix,maximize=True)
        matched_indices = np.array(list(zip(row,col)))
    else:
        matched_indices = np.empty((0,2),dtype=int)

    # Unmatched detections
    det_mask = []
    for i in range(len(detections)):
        if i in matched_indices[:,0]:
            det_mask.append(False)
        else:
            det_mask.append(True)
    unmatched_detections = detections[det_mask]
    
    # Unmatched trackers
    trk_mask = []
    for j in range(len(trackers)):
        if j in matched_indices[:,1]:
            trk_mask.append(False)
        else:
            trk_mask.append(True)
    unmatched_trackers = trackers[trk_mask]

    # Filter out matches with iou under threshold
    match_mask = []
    for indices in matched_indices:
        row = indices[0]
        col = indices[1]
        if iou_matrix[row,col] >= iou_threshold:
            match_mask.append(True)
        else:
            match_mask.append(False)
    not_match_mask = [not x for x in match_mask]
    unmatched_indices = matched_indices[not_match_mask]
    matched_indices = matched_indices[match_mask]

    # Add filtered out matches to unmatched lists
    unmatched_detections = np.concatenate((unmatched_detections,detections[unmatched_indices[:,0]]))
    unmatched_trackers = np.concatenate((unmatched_trackers,trackers[unmatched_indices[:,1]]))
    
    return matched_indices, unmatched_detections, unmatched_trackers


detector = YOLOv8Detector()

path = os.path.dirname('data/train/')

# Cicle through the train sequences
for seq in os.listdir(path):

    seq_path = os.path.join(path,seq,'img1')

    # Frames list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()

    # Cicle through sequence's frames
    for image in image_files:

        frame = io.imread(os.path.join(seq_path,image))

        detections = detector.get_detections(frame)
