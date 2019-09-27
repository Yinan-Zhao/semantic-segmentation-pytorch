import sys
import os
import numpy as np

with open('./data/ref_training.txt', 'w') as f: 
    for i in range(20210):
        f.write("%d %d %d\n" % (i, i, i))
