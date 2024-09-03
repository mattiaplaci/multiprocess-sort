import os
import numpy as np
import pandas as pd
import motmetrics as mm
import configparser

def load_txt_file(path):

    return mm.io.loadtxt(path,fmt='mot15-2D')


def accumulate_metrics(accumulator,gt_file,results_file,num_frames):

    gt = load_txt_file(gt_file)
    results = load_txt_file(results_file)

    for frame in range(1,num_frames+1):

        if frame in gt.index.get_level_values(0):
            gt_objects = gt.loc[frame]
        else:
            gt_objects = pd.DataFrame()

        if frame in results.index.get_level_values(0):
            tracker_objects = results.loc[frame]
        else:
            tracker_objects = pd.DataFrame()
        
        gt_ids = gt_objects.index.get_level_values(0).to_list()
        tracker_ids = tracker_objects.index.get_level_values(0).to_list()
        
        gt_boxes = gt_objects[['X','Y','Width','Height']].values if len(gt_ids) > 0 else []
        tracker_boxes = tracker_objects[['X','Y','Width','Height']].values if len(tracker_ids) > 0 else []

        if len(gt_boxes) > 0 and len(tracker_boxes) > 0:
            iou_matrix = mm.distances.iou_matrix(gt_boxes,tracker_boxes)
            dists = 1. - iou_matrix
        else:
            dists = np.empty((len(gt_ids),len(tracker_ids)), dtype=np.float32)

        accumulator.update(gt_ids,tracker_ids,dists)


def calculate_metrics(accumulator):

    mh = mm.metrics.create()

    metrics = ['mota', 'motp', 'num_frames', 'mostly_tracked', 'mostly_lost',
               'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations','num_unique_objects']

    summary = mh.compute(accumulator, metrics=metrics, name='Metrics')
    summary['num_frames'] = summary['num_false_positives'] / summary['num_frames']
    summary['motp'] = 1. - summary['motp']
    summary['mostly_tracked'] = summary['mostly_tracked'] / summary['num_unique_objects']
    summary['mostly_lost'] = summary['mostly_lost'] / summary['num_unique_objects']
    summary = summary.drop(columns=['num_unique_objects'])

    summary = mm.io.render_summary(summary,
                                   formatters={'mota': '{:.2%}'.format,
                                               'motp': '{:.2%}'.format,
                                               'num_frames': '{:.2%}'.format,
                                               'mostly_tracked': '{:.2%}'.format,
                                               'mostly_lost': '{:.2%}'.format},
                                    namemap={'mota':'MOTA','motp':'MOTP','num_frames':'FAF',
                                             'mostly_tracked':'MT','mostly_lost':'ML',
                                             'num_false_positives':'FP','num_misses':'FN',
                                             'num_switches':'IDSW','num_fragmentations':'FRAG'})

    print()
    print(summary)
    print()


if not os.path.exists('output'):
    print('Output files not found, please execute sort.py')
    exit()

data_path = os.path.dirname('data/train/')

config = configparser.ConfigParser()

global_accumulator = mm.MOTAccumulator(auto_id=True)

for seq in os.listdir(data_path):

    gt_path = os.path.join(data_path,seq,'gt','gt.txt')
    results_path = os.path.join('output',seq + '.txt')

    config.read(os.path.join(data_path,seq,'seqinfo.ini'))
    num_frames = config.getint('Sequence','seqLength')

    accumulate_metrics(global_accumulator,gt_path,results_path,num_frames)

calculate_metrics(global_accumulator)


