# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:22:01 2019

@author: pc
"""
# This script is used to check the classification result

import numpy as np
import pickle
import os

# set path
path= 'F:/Project5/MNIST_model/model5_experiment/result/'
path_dir = os.listdir(path)

result_anal = []
for fi in path_dir:
    file_fullname = path + fi
    with open(file_fullname, 'rb') as f1:
        result = pickle.load(f1)
        result_anal.append([fi, np.max(result[2][:, 3])])
        