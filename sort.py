import os
import cv2
import numpy as np
import configparser
import tkinter
import torch
import argparse
import time

from ultralytics import YOLO

from filterpy.kalman import KalmanFilter

from scipy.optimize import linear_sum_assignment

import cProfile
import pstats


class YOLOv8Detector:

    def __init__(self,score_threshold=0.5):

        if use_gpu:
            # Use gpu if available
            self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # Pretrained YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)

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

        if display:
            # Assign random color to tracker
            trackers_color[self.id] = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))

        KalmanBoxTracker.count += 1


    def predict(self):

        # Handle scale (area) negative values
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0

        # Predict next frame
        self.kf.predict()

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_box()


    def update(self,det):
        
        # Update tracker with new detection
        box = convert_xyxy_to_xysr(det)
        self.kf.update(box)

        self.time_since_update = 0
        self.hit_streak += 1


    # Getter methods

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

    def update(self,detections):

        global frame_count
        
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
        self.trackers = [trk for trk in self.trackers if trk.get_time_since_update() < self.max_age]

        # Build output
        output = []
        for trk in self.trackers:
            if trk.get_hit_streak() >= self.min_hits or frame_count <= self.min_hits:
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


def process_frame(image_path,frame):

    # Get detections from YOLOv8
    detections = detector.get_detections(image_path)

    # Update trackers state
    output = mot_tracker.update(detections)

    # Display results frame by frame
    if display:

        # Drawing bounding boxes
        for o in output:
            id, x1, y1, x2, y2 = int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4])
            color = trackers_color[id]
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f'ID: {id}', (x1, y1-5), cv2.FONT_ITALIC, 0.6, color, 2)

        # Adjust visualization in case of screen not big enough
        if width > screen_width or height > screen_height:
            if ratio > screen_ratio:
                frame = cv2.resize(frame,(screen_width,int(screen_width/ratio)))
            elif ratio < screen_ratio:
                frame = cv2.resize(frame,(int(screen_height*ratio),screen_height))
            else:
                frame = cv2.resize(frame,(screen_width,screen_height))

        cv2.imshow(seq,frame)
        cv2.waitKey(int(1000/framerate))

    if save_output:

        for o in output:
            id, x1, y1, x2, y2 = int(o[0]), o[1], o[2], o[3], o[4]
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(
                frame_count,id,x1,y1,x2-x1,y2-y1), file=output_file)


# Parse script arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='SORT by Mattia Corrado Placì')
    parser.add_argument('--display', dest='display', action='store_true', help='Display tracker output [False by default]')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='Decide whether to use gpu to compute detections [False by default]')
    parser.add_argument('--test', dest='test', action='store_true', help='Decide whether to use test set, metrics not available [False by default]')
    parser.add_argument('--save_output', dest='save_output', action='store_true', help='Decide whether to save the tracker output [False by default]')
    parser.add_argument('--profile', dest='profile', action='store_true', help='Enable profiling [False by default]')
    parser.add_argument('-max_age', default=1, type=int, help='Maximum number of frames to keep alive a track without associated detections [1 by default]')
    parser.add_argument('-min_hits', default=3, type=int, help='Minimum number of associated detections before track is initialised [3 by default]')
    parser.add_argument('-iou_threshold', default=0.3, type=float, help='Minimum IOU for match [0.3 by default]')
    parser.add_argument('-detection_score_threshold', default=0.5, type=float, help='Minimum score to consider detection [0.5 by default]')
    args = parser.parse_args()
    return args


# Script arguments
args = parse_arg()
display = args.display
use_gpu = args.use_gpu
test = args.test
save_output = args.save_output
profile = args.profile

# Profiling
if profile:
    profiler = cProfile.Profile()
    profiler.enable()

if display:
    # Screen info
    tk = tkinter.Tk()
    screen_width, screen_height = tk.winfo_screenwidth(), tk.winfo_screenheight()
    screen_ratio = screen_width/screen_height
    
    # Configuration files reader
    config = configparser.ConfigParser()

# Load YOLOv8 detector
detector = YOLOv8Detector(args.detection_score_threshold)

# Create SORT tracker object
mot_tracker = SORT(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)

# Train set path
if test:
    path = os.path.dirname('data/test/')
else:
    path = os.path.dirname('data/train/')

# Create output directory
if save_output and not os.path.exists('output'):
    os.makedirs('output')

# Cicle through the train sequences
for seq in os.listdir(path):

    seq_path = os.path.join(path,seq,'img1')

    if display:
        # Get sequence info
        config.read(os.path.join(path,seq,'seqinfo.ini'))
        framerate = config.getint('Sequence', 'frameRate')
        width = config.getint('Sequence','imWidth')
        height = config.getint('Sequence','imHeight')
        ratio = width/height

        # Keep track of bounding boxes colors
        trackers_color = {}

    # Frames list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()
    frame_count = 0

    # Read frames
    frame_list = [cv2.imread(os.path.join(seq_path,image_file)) for image_file in image_files]

    # Open output file
    if save_output:
        output_file = open(os.path.join('output','%s.txt'%(seq)),'w')

    print(seq+': ','Processing...')

    # Cicle through sequence's frames
    for x in range(len(image_files)):

        frame_count += 1

        image_path = os.path.join(seq_path,image_files[x])

        process_frame(image_path,frame_list[x])

    # Reset tracker for new sequence
    mot_tracker.reset()
    KalmanBoxTracker.count = 0

    if display:
        cv2.destroyAllWindows()
    if save_output:
        output_file.close()

if profile:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.dump_stats('profile.prof')
