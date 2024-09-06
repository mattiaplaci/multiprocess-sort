import os
import cv2
import numpy as np
import configparser
import tkinter
import argparse

import sort

import time
import psutil
import matplotlib.pyplot as plt
from pynvml import *

import cProfile
import pstats


class Displayer:

    def __init__(self):

        # Screen info
        tk = tkinter.Tk()
        self.screen_width, self.screen_height = tk.winfo_screenwidth(), tk.winfo_screenheight()
        self.screen_ratio = self.screen_width/self.screen_height

    def get_sequence_visualization_info(self,config):

        # Sequence name
        self.seq_name = config.get('Sequence','name')

        # Visualization info
        self.width = config.getint('Sequence','imWidth')
        self.height = config.getint('Sequence','imHeight')
        self.ratio = self.width/self.height

    def show(self,image_path,output):

        frame = cv2.imread(os.path.join(image_path)) 

        # Drawing bounding boxes
        for o in output:
            id, x1, y1, x2, y2 = int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4])
            color = mot_tracker.trackers[id].get_color()
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f'ID: {id}', (x1, y1-5), cv2.FONT_ITALIC, 0.6, color, 2)

        # Adjust visualization in case of screen not big enough
        if self.width > self.screen_width or self.height > self.screen_height:
            if self.ratio > self.screen_ratio:
                frame = cv2.resize(frame,(self.screen_width,int(self.screen_width/self.ratio)))
            elif self.ratio < self.screen_ratio:
                frame = cv2.resize(frame,(int(self.screen_height*self.ratio),self.screen_height))
            else:
                frame = cv2.resize(frame,(self.screen_width,self.screen_height))

        cv2.imshow(self.seq_name,frame)
        cv2.waitKey(1)

    def stop(self):
        cv2.destroyAllWindows()
        self.reset()

    def reset(self):
        self.seq_name = ''
        self.width = 0
        self.height = 0
        self.ratio = 0.0

class PerformanceMeter:

    def __init__(self,use_gpu=False):

        self.use_gpu = use_gpu

        if self.use_gpu:
            performance_file_name = 'gpu_performance.txt'
        else:
            performance_file_name = 'cpu_performance.txt'

        self.performance_file = open(os.path.join('performances',performance_file_name),'w')

        # Process PID
        self.process = psutil.Process(os.getpid())

        # CPU usage
        self.cpu_usage = []
        self.process.cpu_percent()

        # Memory usage
        _, self.ax = plt.subplots()
        self.mem_usage = []

        # GPU usage
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = nvmlDeviceGetName(self.handle).decode("utf-8")
        self.used_gpu_memory = 0.0
        self.gpu_usage = []
        for p in nvmlDeviceGetComputeRunningProcesses(self.handle):
            if p.pid == os.getpid():
                self.used_gpu_memory = p.usedGpuMemory / 1024**2
                break

        # Timer
        self.global_frame_count = 0
        self.global_avg_frame_time = 0.0
        self.start_time = time.time()

    def new_sequence_measurement(self,seq_name):
        self.seq_name = seq_name
        self.avg_frame_time = 0.0
        self.frame_start_time = -1
        self.frame_count = 0

    def start_frame_measure(self):
        self.frame_count += 1
        self.global_frame_count += 1
        self.frame_start_time = time.time()
    
    def end_frame_measure(self):
        frame_end_time = time.time()
        frame_time = frame_end_time - self.frame_start_time

        self.cpu_usage.append(self.process.cpu_percent() / psutil.cpu_count())

        self.mem_usage.append(self.process.memory_info().rss / 1024**2)
            
        self.gpu_usage.append(nvmlDeviceGetUtilizationRates(self.handle).gpu)
            
        self.avg_frame_time += (frame_time - self.avg_frame_time) / self.frame_count
        self.global_avg_frame_time += (frame_time - self.global_avg_frame_time) / self.global_frame_count

    def save_seq_measurement(self):
        print(self.seq_name+
              '\n\tAvarage time per frame: {:.2f}'.format(self.avg_frame_time),
              '\n\tAvarage FPS: {:.2f}'.format(1/self.avg_frame_time),file=self.performance_file)
    
    def save_global_measurement(self):

        end_time = time.time()

        nvmlShutdown()

        print('\nGlobal avarage time per frame: {:.2f}'.format(self.global_avg_frame_time),file=self.performance_file)
        print('Global avarage FPS: {:.2f}'.format(1/self.global_avg_frame_time),file=self.performance_file)
        print('Total time: {:.2f}'.format(end_time-self.start_time),file=self.performance_file)
        print('\n\nAvarage CPU usage: {:.2f}%'.format(np.array(self.cpu_usage).mean()),file=self.performance_file)
        print('\nAvarage memory usage: {:.2f} MiB'.format(np.array(self.mem_usage).mean()),file=self.performance_file)
        print('\nGPU -',self.gpu_name+':',file=self.performance_file)
        print('\tAvarage GPU usage: {:.2f}%'.format(np.array(self.gpu_usage).mean()),file=self.performance_file)
        print('\tGPU memory used: {:.2f} MiB'.format(self.used_gpu_memory),file=self.performance_file)
        self.performance_file.close()

        self.ax.clear()
        self.ax.plot(self.mem_usage, label='Utilizzo memoria (MiB)')
        self.ax.set_title('Monitoraggio utilizzo memoria utilizzata in tempo reale')
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Utilizzo memoria (MiB)')
        self.ax.legend(loc='lower right')
        if self.use_gpu:
            plt.savefig(os.path.join('performances','memory_using_gpu.png'))
        else:
            plt.savefig(os.path.join('performances','memory_using_cpu.png'))

def process_frame(image_path,displayer=None):

    detections = detector.get_detections(image_path)

    # Update trackers state
    output = mot_tracker.update(detections)

    # Display results frame by frame
    if displayer is not None:
        displayer.show(image_path,output)

    if save_output:
        for o in output:
            id, x1, y1, x2, y2 = int(o[0]), o[1], o[2], o[3], o[4]
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(
                frame_count,id,x1,y1,x2-x1,y2-y1), file=output_file)

# Parse script arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='SORT by Mattia Corrado Placì')
    parser.add_argument('--display', dest='display', action='store_true', help='Display tracker output [False by default]')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='Use gpu to compute detections [False by default]')
    parser.add_argument('--test', dest='test', action='store_true', help='Use test set [False by default]')
    parser.add_argument('--save_output', dest='save_output', action='store_true', help='Save the tracker output [False by default]')
    parser.add_argument('--profile', dest='profile', action='store_true', help='Enable profiling [False by default]')
    parser.add_argument('--performance', dest='performance', action='store_true', help='Enable performance measurement [False by default]')
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
performance = args.performance

# Configuration files reader
config = configparser.ConfigParser()

# Load YOLOv8 detector
detector = sort.YOLOv8Detector(args.detection_score_threshold,use_gpu)

# Create SORT tracker object
mot_tracker = sort.SORT(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)


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
    performance_meter = PerformanceMeter(use_gpu)

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

    if display:
        displayer.get_sequence_visualization_info(config)
    
    # Open output file
    if save_output:
        output_file = open(os.path.join('output','%s.txt'%(seq)),'w')

    if performance:
        performance_meter.new_sequence_measurement(seq)

    # Cicle through sequence's frames
    for image_file in image_files:

        frame_count += 1

        # Initialize frame timer
        if performance:
            performance_meter.start_frame_measure()

        image_path = os.path.join(seq_path,image_file)

        if display:
            process_frame(image_path,displayer)
        else:
            process_frame(image_path)

        if performance:
            performance_meter.end_frame_measure()

    # Reset tracker for new sequence
    mot_tracker.reset()
    sort.KalmanBoxTracker.count = 0

    if display:
        displayer.stop()
    if save_output:
        output_file.close()
    if performance:
        performance_meter.save_seq_measurement()

# Save performance measurement
if performance:
    performance_meter.save_global_measurement()

if profile:
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('profile.prof')