import os
import configparser
import argparse
import gc

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
    args = parser.parse_args()
    return args


# Producer routine
def detection_producer(input_queue,output_queue):

    gc.collect()

    # Load YOLOv8 detector
    detector = YOLOv8Detector()

    while True:

        # Get new frame
        frame = input_queue.get()
        frame_id = frame.get_id()
        if frame_id == -1:
            break
        image_path = frame.get_image()

        # Produce detections
        detections = detector.get_detections(image_path)
        frame.set_detections(detections)

        # Send detections
        output_queue.put(frame)


# Children processes coordination
def producers_coordinator(seq_path,image_files,framerate,num_producers,
                          input_queue,output_queue,
                          performance_manager):
    
    # Create processes
    child_processes = []
    for _ in range(num_producers):

        p = mp.Process(target=detection_producer,
                       args=(input_queue,output_queue))
        
        p.start()
        child_processes.append(p)

        if performance_manager is not None:
            performance_manager.add_child(psutil.Process(p.pid))

    frame_count = 0

    # Capture frames
    for image_file in image_files:

        frame_count += 1

        image_path = os.path.join(seq_path,image_file)

        frame = Frame(image_path,frame_count)
        
        input_queue.put(frame)

        time.sleep(1/framerate)

    # Send every child a stop signal
    stop = Frame(None,-1)
    for _ in range(num_producers):
        input_queue.put(stop)
    
    # Wait for children
    for p in child_processes:
        p.join()

    # Tell main thread that task is done
    output_queue.put(stop)


# Script arguments
args = parse_arg()
display = args.display
test = args.test
save_output = args.save_output
profile = args.profile
performance = args.performance

# Number of children processes
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

# Object to show outputs
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

# Start profiling
if profile:
    profiler = cProfile.Profile()
    profiler.enable()

# Performance metrics
if performance:
    performance_manager = PerformanceManager()
    performance_manager.resources_init()
    performance_manager.start_global_measurement()

# Cicle through the train sequences
for seq in os.listdir(path):

    print(seq+': ','Processing...')

    # Sequence data path
    seq_path = os.path.join(path,seq,'img1')

    # Image files list
    image_files = [f for f in os.listdir(seq_path)]
    image_files.sort()

    # Initialize frame counter
    frame_count = 0

    # Get sequence info
    config.read(os.path.join(path,seq,'seqinfo.ini'))

    # Get sequence framerate
    framerate = config.getint('Sequence','frameRate')

    # Get sequence settings if needed
    if display:
        displayer.get_sequence_visualization_info(config)
        
    # Open output file
    if save_output:
        output_file = open(os.path.join('output','%s.txt'%(seq)),'w')

    # Start sequence measurement
    if performance:
        performance_manager.new_sequence_timer(config)
        performance_manager.start_sequence_timer()

    gc.disable()

    # New thread to manage children processes
    perf_arg = performance_manager if performance else None
    cooridinator_thread = threading.Thread(target=producers_coordinator,
                                        args=(seq_path,
                                                image_files,
                                                framerate,
                                                num_producers,
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

        # Measurements
        if performance:
            performance_manager.latency_timer(frame.get_timestamp())
            performance_manager.get_resources()

        # Display results frame by frame
        if display:
            image_path = os.path.join(seq_path,image_files[frame_id-1])
            displayer.update_boxes(mot_tracker.trackers)
            displayer.show(image_path,output)

        # Save outputs
        if save_output:
            for o in output:
                id, x1, y1, x2, y2 = int(o[0]), o[1], o[2], o[3], o[4]
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(
                    frame_count,id,x1,y1,x2-x1,y2-y1), file=output_file)

    gc.enable()

    # Stop measurements
    if performance:
        performance_manager.end_sequence_timer()
        performance_manager.save_seq_measurement()
        
    # Stop visualization
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

# Stop profiling
if profile:
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('profile.prof')