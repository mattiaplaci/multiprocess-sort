import os
import cv2
import numpy as np

import threading

import time
import psutil
from pynvml import *


class Frame:

    def __init__(self,image,id):
        self.image = image
        self.id = id
        self.timestamp = time.time()
        self.detections = None

    def set_detections(self,detections):
        self.detections = detections

    def get_image(self):
        return self.image

    def get_id(self):
        return self.id
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_detections(self):
        return self.detections
    

class Displayer:

    def __init__(self):

        self.trackers_color = {}

    def get_sequence_visualization_info(self,config):

        # Sequence name
        self.seq_name = config.get('Sequence','name')

    def update_boxes(self,trackers):
        for trk in trackers:
            self.trackers_color[trk.get_id()] = trk.get_color()

    def show(self,image_path,output):

        frame = cv2.imread(os.path.join(image_path)) 

        # Drawing bounding boxes
        for o in output:
            id, x1, y1, x2, y2 = int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4])
            color = self.trackers_color[id]
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f'ID: {id}', (x1, y1-5), cv2.FONT_ITALIC, 0.6, color, 2)

        cv2.imshow(self.seq_name,frame)
        cv2.waitKey(1)

    def stop(self):
        cv2.destroyAllWindows()
        self.reset()

    def reset(self):
        self.seq_name = ''
        self.trackers_color = {}


def memory_measurement(performance_manager):

    while True:

        if performance_manager.stop_thread:
            break

        performance_manager.measure_memory()
        performance_manager.measure_gpu_memory()

        time.sleep(1)


class PerformanceManager:

    def __init__(self):

        self.performance_file = open(os.path.join('performances','performances.txt'),'w')
        self.output_file = open(os.path.join('performances','output.txt'),'w')

        # Process PID
        self.process = psutil.Process(os.getpid())
        self.children = []
        self.children_pids = []

        # Performances
        self.global_performance = []
        self.sequences_performance = {}


    def add_child(self, child):
        self.children.append(child)
        self.children_pids.append(child.pid)


    def resources_init(self):

        # CPU usage
        self.cpu_usage = []
        self.process.cpu_percent()

        # Memory usage
        self.mem_usage = 0.0
        self.stop_thread = False

        self.mem_thread = threading.Thread(target=memory_measurement,
                                           args=[self])

        # GPU usage
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = nvmlDeviceGetName(self.handle)
        self.used_gpu_memory = 0.0
        self.gpu_usage = []

        self.mem_thread.start()


    def measure_memory(self):
        mem = self.process.memory_info().rss / 1024**2
        for p in self.children:
            if p.is_running():
                mem += p.memory_info().rss / 1024**2
        self.mem_usage = max(mem, self.mem_usage)

    def measure_gpu_memory(self):
        gpu = 0.0
        for p in nvmlDeviceGetComputeRunningProcesses(self.handle):
            if p.pid in self.children_pids or p.pid == os.getpid():
                gpu += p.usedGpuMemory / 1024**2
        self.used_gpu_memory = max(gpu, self.used_gpu_memory)

    def measure_gpu_usage(self):
        self.gpu_usage.append(nvmlDeviceGetUtilizationRates(self.handle).gpu)

    def measure_cpu_usage(self):
        cpu = self.process.cpu_percent() / psutil.cpu_count()
        for p in self.children:
            if p.is_running():
                cpu += p.cpu_percent() / psutil.cpu_count()
        self.cpu_usage.append(cpu)

    def get_resources(self):

        self.measure_cpu_usage()
        self.measure_gpu_usage()
    

    def new_sequence_timer(self,config):
        self.seq_name = config.get('Sequence','name')
        self.tot_frames = config.getint('Sequence','seqLength')
        self.seq_latency = 0.0
        self.avg_frame_time = 0.0
        self.seq_start_time = -1

    def start_sequence_timer(self):
        self.seq_start_time = time.time()
    
    def end_sequence_timer(self):
        seq_end_time = time.time()
        self.global_frame_count += self.tot_frames
        seq_time = seq_end_time - self.seq_start_time

        self.avg_frame_time += seq_time / self.tot_frames
        self.seq_latency /= self.tot_frames
        self.global_avg_frame_time += seq_time

    def end_seq_measurement(self):
        
        print(self.seq_name+
              '\n\tAvarage time per frame: {:.2f}'.format(self.avg_frame_time),
              '\n\tAvarage FPS: {:.2f}'.format(1/self.avg_frame_time),
              '\n\tAvarage latency: {:.2f}'.format(self.seq_latency),
              file=self.performance_file)
        
        self.sequences_performance[self.seq_name] = []
        self.sequences_performance[self.seq_name].append(self.avg_frame_time)
        self.sequences_performance[self.seq_name].append(1/self.avg_frame_time)
        self.sequences_performance[self.seq_name].append(self.seq_latency)
    
    
    def latency_timer(self,start):
        end = time.time()
        latency = end-start
        self.avg_latency += latency
        self.seq_latency += latency


    def start_global_measurement(self):
        # Timer
        self.global_frame_count = 0
        self.global_avg_frame_time = 0.0
        self.avg_latency = 0.0
        self.start_time = time.time()
    
    def end_global_measurement(self):

        end_time = time.time()
        self.stop_thread = True

        self.global_avg_frame_time /= self.global_frame_count
        self.avg_latency /= self.global_frame_count

        nvmlShutdown()

        self.mem_thread.join()

        print('\nGlobal avarage time per frame: {:.2f}'.format(self.global_avg_frame_time),
              file=self.performance_file)
        print('Global avarage FPS: {:.2f}'.format(1/self.global_avg_frame_time),
              file=self.performance_file)
        print('Global avarage latency: {:.2f}'.format(self.avg_latency),
              file=self.performance_file)
        print('Total time: {:.2f}'.format(end_time-self.start_time),
              file=self.performance_file)
        print('\n\nAvarage CPU usage: {:.2f}%'.format(np.array(self.cpu_usage).mean()),
              file=self.performance_file)
        print('\nPeak memory usage: {:.2f} MiB'.format(self.mem_usage),
              file=self.performance_file)
        print('\nGPU -',self.gpu_name+':',
              file=self.performance_file)
        print('\tAvarage GPU usage: {:.2f}%'.format(np.array(self.gpu_usage).mean()),
              file=self.performance_file)
        print('\tPeak GPU memory usage: {:.2f} MiB'.format(self.used_gpu_memory),
              file=self.performance_file)
        self.performance_file.close()

        self.global_performance.append(self.global_avg_frame_time)
        self.global_performance.append(1/self.global_avg_frame_time)
        self.global_performance.append(self.avg_latency)
        self.global_performance.append(end_time-self.start_time)
        self.global_performance.append(np.array(self.cpu_usage).mean())
        self.global_performance.append(self.mem_usage)
        self.global_performance.append(np.array(self.gpu_usage).mean())
        self.global_performance.append(self.used_gpu_memory)

        print(self.global_performance,file=self.output_file)

        