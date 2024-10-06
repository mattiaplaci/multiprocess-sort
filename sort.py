import numpy as np

from ultralytics import YOLO

from filterpy.kalman import KalmanFilter

from scipy.optimize import linear_sum_assignment


class YOLOv8Detector:

    def __init__(self,score_threshold=0.5):

        # Pretrained YOLOv8 model
        self.model = YOLO('yolov8x.pt')

        # Detector parameters
        self.score_threshold = score_threshold


    def get_detections(self,image_path):

        # Detect on a single frame
        results = self.model(image_path, verbose=False)

        # Detections in [x1,y1,x2,y2] format
        detections = results[0].boxes.data.cpu().numpy()

        # Discard detections different from pedestrians
        detections = detections[np.where(detections[:,5] == 0)]

        # Discard detections with score less then score_threshold
        detections = detections[np.where(detections[:,4] >= self.score_threshold)]

        # Delete class_id and score columns
        detections = detections[:,:4]

        return detections


# Converts format [x1,y1,x2,y2] to [u,v,s,r]
def convert_xyxy_to_uvsr(box):

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

# Converts format [u,v,s,r] to [x1,y1,x2,y2]
def convert_uvsr_to_xyxy(state_box):

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

        KalmanBoxTracker.count += 1

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
        
        # state covariance matrix P
        self.kf.P *= 10
        self.kf.P[4:,4:] *= 1000
        # process noise matrix Q
        self.kf.Q[4:,4:] *= 0.01
        self.kf.Q[-1,-1] *= 0.01
        # measurement noise matrix R
        self.kf.R[2:,2:] *= 10

        # initial state
        self.kf.x[:4] = convert_xyxy_to_uvsr(box)

        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hit_streak = 0

        # Assign random color to tracker
        self.color = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))

    def predict(self):

        # Handle scale (area) negative values
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        # Predict next frame
        self.kf.predict()

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_box()


    def update(self,det):
        
        # Update tracker with new detection
        box = convert_xyxy_to_uvsr(det)
        self.kf.update(box)

        self.time_since_update = 0
        self.hit_streak += 1


    # Getter methods

    def get_state(self):
        return self.kf.x
    
    def get_box(self):
        return convert_uvsr_to_xyxy(self.kf.x[:4])
    
    def get_id(self):
        return self.id
    
    def get_hit_streak(self):
        return self.hit_streak
    
    def get_time_since_update(self):
        return self.time_since_update
    
    def get_color(self):
        return self.color
    

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
        matched_indices = np.array(list(zip(row,col)),dtype=int)
    else:
        matched_indices = np.empty((0,2),dtype=int)
    
    # Unmatched detections
    unmatched_detections = []
    for i in range(len(detections)):
        if i not in matched_indices[:,0]:
            unmatched_detections.append(i)
    unmatched_detections = np.array(unmatched_detections,dtype=int)
    
    # Unmatched trackers
    unmatched_trackers = []
    for j in range(len(trackers)):
        if j not in matched_indices[:,1]:
            unmatched_trackers.append(j)
    unmatched_trackers = np.array(unmatched_trackers,dtype=int)

    # Filter out matches with iou under threshold
    matched = np.empty((0,2),dtype=int)
    unmatched = np.empty((0,2),dtype=int)
    for indices in matched_indices:
        row = indices[0]
        col = indices[1]
        if iou_matrix[row,col] >= iou_threshold:
            matched = np.concatenate((matched,indices.reshape((1,2))))
        else:
            unmatched = np.concatenate((unmatched,indices.reshape((1,2))))
    
    # Add filtered out matches to unmatched lists
    unmatched_detections = np.concatenate((unmatched_detections,unmatched[:,0]))
    unmatched_trackers = np.concatenate((unmatched_trackers,unmatched[:,1]))
    
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
        matched_indices, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(
            detections,predicted_boxes,self.iou_threshold)

        # Update matched_indices trackers
        for t,trk in enumerate(self.trackers):
            if t not in unmatched_trackers:
                det_index = matched_indices[np.where(matched_indices[:,1] == t)[0][0],0]
                trk.update(detections[det_index])

        # New trackers for unmatched detections
        for d in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[d])
            self.trackers.append(new_tracker)

        # Filter out old trackers
        new_trackers = []
        for trk in self.trackers:
            if trk.get_time_since_update() < self.max_age:
                new_trackers.append(trk)
        self.trackers = new_trackers

        # Build output
        output = []
        for trk in self.trackers:
            if trk.get_hit_streak() >= self.min_hits or self.frame_count <= self.min_hits:
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
