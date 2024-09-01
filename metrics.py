import os
import numpy as np
import pandas as pd
import motmetrics as mm

path = os.path.join('output','KITTI-17.txt')
dati = mm.io.loadtxt(path,fmt='mot15-2D')

print(dati)