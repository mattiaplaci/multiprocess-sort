import os
import configparser
import argparse

from sort import *
from utils import *

import multiprocessing as mp
import threading

import time
import psutil
from pynvml import *

import cProfile
import pstats


# Parse script arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='SORT by Mattia Corrado Placì')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display tracker output [False by default]')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true',
                        help='Use gpu to compute detections [False by default]')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Use test set [False by default]')
    parser.add_argument('--save_output', dest='save_output', action='store_true',
                        help='Save the tracker output [False by default]')
    parser.add_argument('--profile', dest='profile', action='store_true',
                        help='Enable profiling [False by default]')
    parser.add_argument('--performance', dest='performance', action='store_true',
                        help='Enable performance measurement [False by default]')
    parser.add_argument('-num_producers', default=4, type=int,
                        help='Number of processes computing detections in parallel')
    parser.add_argument('-max_age', default=1, type=int,
                        help='Maximum number of frames to keep alive a track without associated detections [1 by default]')
    parser.add_argument('-min_hits', default=3, type=int,
                        help='Minimum number of associated detections before track is initialised [3 by default]')
    parser.add_argument('-iou_threshold', default=0.3, type=float,
                        help='Minimum IOU for match [0.3 by default]')
    parser.add_argument('-detection_score_threshold', default=0.5, type=float,
                        help='Minimum score to consider detection [0.5 by default]')
    args = parser.parse_args()
    return args


def detection_producer(detection_score_threshold,use_gpu,input_queue,output_queue):

    # Load YOLOv8 detector
    detector = YOLOv8Detector(detection_score_threshold,use_gpu)

    # Send detection to parent process
    while True:

        frame = input_queue.get()
        frame_id = frame.get_id()
        if frame_id == -1:
            break
        image_path = frame.get_image()

        detections = detector.get_detections(image_path)
        frame.set_detections(detections)

        output_queue.put(frame)


def producers_coordinator(seq_path,image_files,framerate,num_producers,
                          detection_score_threshold,use_gpu,
                          input_queue,output_queue,
                          performance_manager):
    
    child_processes = []
    for _ in range(num_producers):

        p = mp.Process(target=detection_producer,
                       args=(detection_score_threshold,use_gpu,
                             input_queue,output_queue))
        
        p.start()
        child_processes.append(p)

        if performance_manager is not None:
            performance_manager.add_child(psutil.Process(p.pid))

    frame_count = 0

    for image_file in image_files:

        frame_count += 1

        image_path = os.path.join(seq_path,image_file)

        frame = Frame(image_path,frame_count)
        
        input_queue.put(frame)

        time.sleep(1/framerate)

    stop = Frame(None,-1)
    for _ in range(num_producers):
        input_queue.put(stop)
    
    for p in child_processes:
        p.join()

    output_queue.put(stop)


# Script arguments
args = parse_arg()
display = args.display
use_gpu = args.use_gpu
test = args.test
save_output = args.save_output
profile = args.profile
performance = args.performance
num_producers = args.num_producers

# Configuration files reader
config = configparser.ConfigParser()

# Create SORT tracker object
mot_tracker = SORT(max_age=args.max_age,
                   min_hits=args.min_hits,
                   iou_threshold=args.iou_threshold)

# Input and output queues
input_queue = mp.Queue()
output_queue = mp.Queue()

if display:
    displayer = Displayer()

# Dataset path
if test:
    path = os.path.dirname('data/test/')
else:
    path = os.path.dirname('data/train/')

# Create output directory
if save_output and not os.path.exists('output'):
    os.makedirs('output')

# Profiling
if profile:
    profiler = cProfile.Profile()
    profiler.enable()

# Performance metrics
if performance:
    performance_manager = PerformanceManager(use_gpu)
    performance_manager.resources_init()
    performance_manager.start_global_measurement()

# Cicle through the train sequences
for seq in os.listdir(path):

    print(seq+': ','Processing...')

    seq_path = os.path.join(path,seq,'img1')

    # Image names list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()
    frame_count = 0

    # Get sequence info
    config.read(os.path.join(path,seq,'seqinfo.ini'))

    # Get sequence framerate
    framerate = config.getint('Sequence','frameRate')

    if display:
        displayer.get_sequence_visualization_info(config)
    
    # Open output file
    if save_output:
        output_file = open(os.path.join('output','%s.txt'%(seq)),'w')

    # Start sequence measurement
    if performance:
        performance_manager.new_sequence_timer(config)
        performance_manager.start_sequence_timer()

    perf_arg = performance_manager if performance else None
    cooridinator_thread = threading.Thread(target=producers_coordinator,
                                           args=(seq_path,
                                                 image_files,
                                                 framerate,
                                                 num_producers,
                                                 args.detection_score_threshold,
                                                 use_gpu,
                                                 input_queue,
                                                 output_queue,
                                                 perf_arg))
    cooridinator_thread.start()
    frames_buffer = {}

    # Cicle through sequence's frames
    while True:

        frame_count += 1

        # Get frame from child processes
        if frame_count in frames_buffer.keys():
            frame = frames_buffer.pop(frame_count)
        else:
            frame = output_queue.get()
            frame_id = frame.get_id()
            if frame_id == -1:
                break
            while frame_id != frame_count:
                frames_buffer[frame_id] = frame
                frame = output_queue.get()
                frame_id = frame.get_id()

        # Update trackers state
        output = mot_tracker.update(frame.get_detections())

        if performance:
            performance_manager.latency_timer(frame.get_timestamp())
            performance_manager.get_resources()

        # Display results frame by frame
        if display:
            image_path = os.path.join(seq_path,image_files[frame_id-1])
            displayer.update_boxes(mot_tracker.trackers)
            displayer.show(image_path,output)

        if save_output:
            for o in output:
                id, x1, y1, x2, y2 = int(o[0]), o[1], o[2], o[3], o[4]
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(
                    frame_count,id,x1,y1,x2-x1,y2-y1), file=output_file)

    if performance:
        performance_manager.end_sequence_timer()
        performance_manager.save_seq_measurement()
    if display:
        displayer.stop()
    if save_output:
        output_file.close()

    # Reset tracker for new sequence
    mot_tracker.reset()
    KalmanBoxTracker.count = 0

# Save performance measurement
if performance:
    performance_manager.save_global_measurement()

if profile:
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('profile.prof')