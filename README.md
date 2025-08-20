# Multiprocess SORT

This project was first developed as final assignment for the "Artificial Intelligence Laboratory" course at Sapienza University of Rome, then parallelized and extended as work for my bachelor thesis.

## Overview

This project implements a multiprocess version of the SORT (Simple Online and Realtime Tracking) algorithm using YOLOv8n as the object detection model. The implementation is optimized for real-time applications and is trained/evaluated on the 2D MOT15 dataset.

The multiprocess parallelization strategy separates detections task into concurrent processes, allowing for optimal resource utilization and higher frame rates in real-time applications.

## Installation

### Prerequisites
- Pyhton 3.8 or higher
- pip (Pyhton package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/mattiaplaci/multiprocess-sort.git
cd multiprocess-sort
```
### 2. Install Python Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
### 3. Download the 2D MOT15 Dataset
```bash
# Create data directory
mkdir -p data

# Download MOT15 dataset
wget https://motchallenge.net/data/MOT15.zip -P data/

# Extract the dataset
unzip data/MOT15.zip -d data/

# The directory structure should look like:
# data/train/
# data/test/
```
If the direct download doesn't work, manually download from [MOTChallenge](https://motchallenge.net/data/MOT15/) and extract to the data directory.

### 4. YOLOv8n Model Weights
The model will be automatically downloaded on first run if not present.

## Usage

### Basic execution

```bash
# Run with default parameters
python src/main.py

# Run with real-time display
python src/main.py --display
```
### Other command line arguments
- ``` --save_output ``` Save algorithm results to file
- ``` --num_producers ``` Choose number of concurrent processes
- ``` --max_age ``` Maximum number of frames to keep alive a track without associated detections
- ``` --min_hits ``` Minimum number of associated detections before track is initialised
- ``` --iou_threshold ``` Intersection Over Union threshold for association

## References

- [SORT: Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MOTChallenge: 2D MOT 2015](https://motchallenge.net/data/MOT15/)