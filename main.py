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
    parser = argparse.ArgumentParser(description='Parallelized SORT by Mattia Corrado Placì')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display tracker output [False by default]')
    parser.add_argument('--profile', dest='profile', action='store_true',
                        help='Enable profiling [False by default]')
    parser.add_argument('--performance', dest='performance', action='store_true',
                        help='Enable performance measurement [False by default]')
    parser.add_argument('--save_output', dest='save_output', action='store_true',
                        help='Save the tracker output [False by default]')
    parser.add_argument('--realtime', dest='realtime', action='store_true',
                        help='Enable realtime simulation [Disabled by default]')
    parser.add_argument('-set', default=None, type=str,
                        help='Dataset to use (train, test, validation, None) if None use all of the dataset [None by default]')
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
def detection_producer(input_queue,output_queue,stop_queue,semaphore):

    # Load YOLOv8 detector
    detector = YOLOv8Detector()

    while True:

        # Get new frame
        frame = input_queue.get()
        frame_id = frame.get_id()
        if frame_id == 0:
            stop_queue.put(Frame(None,0))
            semaphore.acquire()
            continue
        if frame_id == -1:
            break
        image_path = frame.get_image()

        # Produce detections
        detections = detector.get_detections(image_path)
        frame.set_detections(detections)

        # Send detections
        output_queue.put(frame)

    return 0


# Children processes coordination
def frames_reader(seq_path,image_files,realtime,framerate,num_producers,input_queue,output_queue,stop_queue,semaphores):

    frame_count = 0

    # Capture frames
    for image_file in image_files:

        frame_count += 1

        image_path = os.path.join(seq_path,image_file)

        frame = Frame(image_path,frame_count)
        
        input_queue.put(frame)
        
        if realtime:
            time.sleep(1/30)
        else:
            time.sleep(1/framerate)

    for _ in range(num_producers):
        input_queue.put(Frame(None,0))

    for _ in range(num_producers):
        stop_queue.get()

    for i in range(num_producers):
        semaphores[i].release()

    # Tell main thread that task is done
    output_queue.put(Frame(None,-1))


def main(display=False,profile=False,performance=False,save_output=False,realtime=False,var_set=None,num_producers=4,max_age=1,min_hits=3,iou_threshold=0.3):

    # Configuration files reader
    config = configparser.ConfigParser()

    # Create SORT tracker object
    mot_tracker = SORT(max_age=max_age,
                    min_hits=min_hits,
                    iou_threshold=iou_threshold)
    
    # Object to show outputs
    if display:
        displayer = Displayer()

    # Dataset path
    match var_set:
        case 'train':
            paths = [os.path.join('data',var_set)]
        case 'test':
            paths = [os.path.join('data',var_set)]
        case 'validation':
            paths = [os.path.join('data',var_set)]
        case _:
            paths = []
            paths.append(os.path.join('data','train'))
            paths.append(os.path.join('data','test'))
            paths.append(os.path.join('data','validation'))

    # Create output directory
    if save_output and not os.path.exists('output'):
        os.makedirs('output')

    # Performance metrics
    if performance:
        performance_manager = PerformanceManager()
        performance_manager.resources_init()

    # Input and output queues
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    stop_queue = mp.Queue()
    semaphores = [mp.Semaphore(0) for _ in range(num_producers)]

    # Create processes
    child_processes = []
    for i in range(num_producers):

        p = mp.Process(target=detection_producer, args=(input_queue,output_queue,stop_queue,semaphores[i]))
        p.daemon = True
        p.start()
        child_processes.append(p)

        if performance:
            performance_manager.add_child(psutil.Process(p.pid))

    # Start measurement
    if performance:
        performance_manager.start_global_measurement()

    # Start profiling
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Cicle through the sequences
    for path in paths:
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

            # New thread to read frames
            frames_thread = threading.Thread(target=frames_reader,
                                                args=(seq_path,
                                                        image_files,
                                                        realtime,
                                                        framerate,
                                                        num_producers,
                                                        input_queue,
                                                        output_queue,
                                                        stop_queue,
                                                        semaphores))
            frames_thread.start()
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
                    while frame_id != frame_count and frame_id != -1:
                        frames_buffer[frame_id] = frame
                        frame = output_queue.get()
                        frame_id = frame.get_id()

                if frame_id == -1:
                    break

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

            # Stop measurements
            if performance:
                performance_manager.end_sequence_timer()
                performance_manager.end_seq_measurement()
                
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
        performance_manager.end_global_measurement()

    # Stop profiling
    if profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats('profile.prof')

    # Send every child a stop signal
    stop = Frame(None,-1)
    for _ in range(num_producers):
        input_queue.put(stop)

    # Wait for children
    for p in child_processes:
        p.join()
        if p.exitcode != 0:
            raise ValueError(f'Child {p.pid} exited with code {p.exitcode}')


if __name__ == '__main__':

    # Script arguments
    args = parse_arg()
    display = args.display
    profile = args.profile
    performance = args.performance
    save_output = args.save_output
    realtime = args.realtime

    # Dataset
    var_set = args.set

    # Number of children processes
    num_producers = args.num_producers

    # Parameters
    max_age = args.max_age
    min_hits = args.min_hits
    iou_threshold = args.iou_threshold

    main(display,profile,performance,save_output,realtime,var_set,num_producers,max_age,min_hits,iou_threshold)