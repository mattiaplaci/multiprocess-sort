import os
import cv2
import numpy as np
from skimage import io
import configparser
import tkinter

from ultralytics import YOLO

from filterpy.kalman import KalmanFilter

from scipy.optimize import linear_sum_assignment


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
def convert_xyxy_to_xysr(box):

    w = box[2] - box[0]
    h = box[3] - box[1]

    # Ratio
    r = w / h
    # Scale (area)
    s = w * h
    # Center x coordinate
    x = box[0] + w/2
    # Center y coordinate
    y = box[1] + h/2

    box[0] = x
    box[1] = y
    box[2] = s
    box[3] = r

    return box.reshape((4,1))

# Converts format [x,y,s,r] to [x1,y1,x2,y2]
def convert_xysr_to_xyxy(state_box):

    box = np.copy(state_box)
    box = np.reshape(box,(4,))

    w = np.sqrt(box[2] * box[3])
    h = w / box[3]

    # Top-left x coordinate
    x1 = box[0] - w/2
    # Top-left y coordinate
    y1 = box[1] - h/2
    # Bottom-right x coordinate
    x2 = box[0] + w/2
    # Bottom-right y coordinate
    y2 = box[1] + h/2

    box[0] = x1
    box[1] = y1
    box[2] = x2
    box[3] = y2

    return box


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
        self.kf.x[:4] = convert_xyxy_to_xysr(box)
        # state covariance matrix P
        self.kf.P *= 10
        self.kf.P[4:,4:] *= 1000
        # process noise matrix Q
        self.kf.Q[4:,4:] *= 0.01
        self.kf.Q[-1,-1] *= 0.01
        # measurement noise matrix R
        self.kf.R[2:,2:] *= 10

        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hit_streak = 0

        KalmanBoxTracker.count += 1

    def predict(self):

        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_box()

    def update(self,det):

        box = convert_xyxy_to_xysr(det)
        self.kf.update(box)

        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self):
        return self.kf.x
    
    def get_box(self):
        return convert_xysr_to_xyxy(self.kf.x[:4])
    
    def get_id(self):
        return self.id
    
    def get_hit_streak(self):
        return self.hit_streak
    
    def get_time_since_update(self):
        return self.time_since_update
    

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


def associate_detections_to_trackers(detections,trackers,iou_threshold):
    
    # Empty tracker list
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,1),dtype=int)
    
    # IOU matrix
    iou_matrix = calculate_iou_matrix(detections,trackers)

    # Assignment problem solution
    if min(iou_matrix.shape) > 0:
        row, col = linear_sum_assignment(iou_matrix,maximize=True)
        matched = np.array(list(zip(row,col)),dtype=int)
    else:
        matched = np.empty((0,2),dtype=int)
    
    # Unmatched detections
    unmatched_detections = []
    for i in range(len(detections)):
        if i not in matched[:,0]:
            unmatched_detections.append(i)
    unmatched_detections = np.array(unmatched_detections,dtype=int)
    
    # Unmatched trackers
    unmatched_trackers = []
    for j in range(len(trackers)):
        if j not in matched[:,1]:
            unmatched_trackers.append(j)
    unmatched_trackers = np.array(unmatched_trackers,dtype=int)

    # Filter out matches with iou under threshold
    match_mask = []
    for indices in matched:
        row = indices[0]
        col = indices[1]
        if iou_matrix[row,col] >= iou_threshold:
            match_mask.append(True)
        else:
            match_mask.append(False)
    not_match_mask = [not x for x in match_mask]
    unmatched = matched[not_match_mask]
    matched = matched[match_mask]
    
    # Add filtered out matches to unmatched lists
    unmatched_detections = np.concatenate((unmatched_detections,unmatched[:,0]),dtype=int)
    unmatched_trackers = np.concatenate((unmatched_trackers,unmatched[:,1]),dtype=int)
    
    return matched, unmatched_detections, unmatched_trackers


class SORT:

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):

        self.trackers = []
        
        # Tracker parameters
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.frame_count = 0

    def update(self,detections):

        self.frame_count += 1
        
        # Predict new positions through trackers from previous frame
        predicted_boxes = []
        if len(self.trackers) > 0:
            for trk in self.trackers:
                predicted_box = trk.predict()
                predicted_boxes.append(predicted_box)
            predicted_boxes = np.array(predicted_boxes)
        else:
            predicted_boxes = np.empty((0,4))

        # Detections associations to trackers
        matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(
            detections,predicted_boxes,self.iou_threshold)

        # Update matched trackers
        for t,trk in enumerate(self.trackers):
            if t not in unmatched_trackers:
                det_index = matched[np.where(matched[:,1] == t)[0][0],0]
                trk.update(detections[det_index])

        # New trackers for unmatched detections
        for d in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[d])
            self.trackers.append(new_tracker)

        # Filter out old trackers
        self.trackers = [trk for trk in self.trackers if trk.get_time_since_update() < self.max_age]

        # Build outputs
        output = []
        for trk in self.trackers:
            if trk.get_hit_streak() >= self.min_hits or self.frame_count < self.min_hits:
                box = trk.get_box()
                row = np.concatenate(([trk.get_id()],box)).reshape((1,5))
                output.append(row)

        if len(output) > 0:
            return np.concatenate(output)
        else:
            return np.empty((0,5))
               

    # Reset tracker status for new sequences
    def reset(self):
        self.trackers = []
        self.frame_count = 0


display = True

tk = tkinter.Tk()
screen_width, screen_height = tk.winfo_screenwidth(), tk.winfo_screenheight()
screen_ratio = screen_width/screen_height

detector = YOLOv8Detector()

mot_tracker = SORT()

path = os.path.dirname('data/train/')

# Cicle through the train sequences
for seq in os.listdir(path):

    seq_path = os.path.join(path,seq,'img1')

    # Get sequence info
    config = configparser.ConfigParser()
    config.read(os.path.join(path,seq,'seqinfo.ini'))
    seq_name = config.get('Sequence','name')
    framerate = config.getint('Sequence', 'frameRate')
    width = config.getint('Sequence','imWidth')
    height = config.getint('Sequence','imHeight')
    ratio = width/height

    # Frames list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()

    # Outputs list
    outputs = np.empty((0,5))

    # Cicle through sequence's frames
    for image in image_files:

        frame = io.imread(os.path.join(seq_path,image))

        detections = detector.get_detections(frame)

        output = mot_tracker.update(detections)

        if display:
            for o in output:
                id, x1, y1, x2, y2 = int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(frame, f'ID: {id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if width > screen_width or height > screen_height:
                if ratio > screen_ratio:
                    frame = cv2.resize(frame,(screen_width,int(screen_width/ratio)))
                elif ratio < screen_ratio:
                    frame = cv2.resize(frame,(int(screen_height*ratio),screen_height))
                else:
                    frame = cv2.resize(frame,(screen_width,screen_height))

            cv2.imshow(seq,cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(int(1000/framerate))

        outputs = np.concatenate((outputs,output))

    mot_tracker.reset()
    KalmanBoxTracker.count = 0

    if display:
        cv2.destroyAllWindows()
