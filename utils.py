import os
import cv2
import numpy as np
import tkinter

import threading

import time
import psutil
import matplotlib.pyplot as plt
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

        # Screen info
        tk = tkinter.Tk()
        self.screen_width, self.screen_height = tk.winfo_screenwidth(), tk.winfo_screenheight()
        self.screen_ratio = self.screen_width/self.screen_height

        self.trackers_color = {}

    def get_sequence_visualization_info(self,config):

        # Sequence name
        self.seq_name = config.get('Sequence','name')

        # Visualization info
        self.width = config.getint('Sequence','imWidth')
        self.height = config.getint('Sequence','imHeight')
        self.ratio = self.width/self.height

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
        self.trackers_color = {}


def memory_measurement(performance_manager):

    while True:

        if performance_manager.stop_thread:
            break

        performance_manager.measure_memory()
        performance_manager.measure_gpu_memory()

        time.sleep(1)


class PerformanceManager:

    def __init__(self,save_performance=True):

        self.save_performance = save_performance

        if self.save_performance:
            self.performance_file = open(os.path.join('performances','performances.txt'),'w')

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
        _, self.ax = plt.subplots()
        self.mem_usage = []
        self.stop_thread = False

        self.mem_thread = threading.Thread(target=memory_measurement,
                                           args=[self])

        # GPU usage
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = nvmlDeviceGetName(self.handle).decode("utf-8")
        self.used_gpu_memory = 0.0
        self.gpu_usage = []

        self.mem_thread.start()

    def get_resources(self):

        cpu = self.process.cpu_percent() / psutil.cpu_count()
        for p in self.children:
            if p.is_running():
                cpu += p.cpu_percent() / psutil.cpu_count()
        self.cpu_usage.append(cpu)

        self.gpu_usage.append(nvmlDeviceGetUtilizationRates(self.handle).gpu)

    def measure_memory(self):
        mem = self.process.memory_info().rss / 1024**2
        for p in self.children:
            if p.is_running():
                mem += p.memory_info().rss / 1024**2
        self.mem_usage.append(mem)

    def measure_gpu_memory(self):
        gpu = 0.0
        for p in nvmlDeviceGetComputeRunningProcesses(self.handle):
            if p.pid in self.children_pids or p.pid == os.getpid():
                gpu += p.usedGpuMemory / 1024**2
        self.used_gpu_memory = max(gpu, self.used_gpu_memory)
    

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
        
        if self.save_performance:
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

        if self.save_performance:

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
            print('\nAvarage memory usage: {:.2f} MiB'.format(np.array(self.mem_usage).mean()),
                  file=self.performance_file)
            print('\nGPU -',self.gpu_name+':',
                  file=self.performance_file)
            print('\tAvarage GPU usage: {:.2f}%'.format(np.array(self.gpu_usage).mean()),
                  file=self.performance_file)
            print('\tGPU memory used: {:.2f} MiB'.format(self.used_gpu_memory),
                  file=self.performance_file)
            self.performance_file.close()

            self.ax.clear()
            self.ax.plot(self.mem_usage, label='Utilizzo memoria (MiB)')
            self.ax.set_title('Monitoraggio utilizzo memoria utilizzata in tempo reale')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Utilizzo memoria (MiB)')
            self.ax.legend(loc='lower right')
            plt.savefig(os.path.join('performances','memory_plot.png'))

        self.global_performance.append(self.global_avg_frame_time)
        self.global_performance.append(1/self.global_avg_frame_time)
        self.global_performance.append(self.avg_latency)
        self.global_performance.append(end_time-self.start_time)
        self.global_performance.append(np.array(self.cpu_usage).mean())
        self.global_performance.append(np.array(self.mem_usage).mean())
        self.global_performance.append(np.array(self.gpu_usage).mean())
        self.global_performance.append(self.used_gpu_memory)

        return (self.global_performance, self.sequences_performance)

        