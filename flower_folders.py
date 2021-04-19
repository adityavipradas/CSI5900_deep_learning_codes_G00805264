# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:31:10 2021

@author: aditya.vipradas
"""
import os
from scipy.io import loadmat
import numpy as np
import sys
from shutil import copyfile

np.set_printoptions(threshold=sys.maxsize)

flower_labels = loadmat('imagelabels.mat')
flower_labels = flower_labels['labels'][0]
unique_labels = np.unique(flower_labels)

parent_dir_train = 'E:/oakland_university_courses/winter_2021/CSI_5900_Deep_Learning/project/flower_data_segmented/train/'
parent_dir_val = 'E:/oakland_university_courses/winter_2021/CSI_5900_Deep_Learning/project/flower_data_segmented/val/'
parent_dir_test = 'E:/oakland_university_courses/winter_2021/CSI_5900_Deep_Learning/project/flower_data_segmented/test/'
mode = 0o777

for i in range(len(unique_labels)):
    folder_name = str(unique_labels[i])
    train_path = os.path.join(parent_dir_train, folder_name)
    os.mkdir(train_path,mode)
    val_path = os.path.join(parent_dir_val, folder_name)
    os.mkdir(val_path,mode)
    test_path = os.path.join(parent_dir_test, folder_name)
    os.mkdir(test_path,mode)

images_path = 'segmim/'
image_files = os.listdir(images_path)

data_split = loadmat('setid.mat')
train_split = data_split['trnid'][0]-1
train_labels = flower_labels[train_split]

val_split = data_split['valid'][0]-1
val_labels = flower_labels[val_split]

test_split = data_split['tstid'][0]-1
test_labels = flower_labels[test_split]

#Training images
for i in range(len(train_labels)):
    label = train_labels[i]
    sample = train_split[i]
    file_name = image_files[sample]
    src = images_path + file_name 
    dst = parent_dir_train + str(label) + '/' + file_name
    copyfile(src,dst)

#Validation images
for i in range(len(val_labels)):
    label = val_labels[i]
    sample = val_split[i]
    file_name = image_files[sample]
    src = images_path + file_name 
    dst = parent_dir_val + str(label) + '/' + file_name
    copyfile(src,dst)
    
#Testing images
for i in range(len(test_labels)):
    label = test_labels[i]
    sample = test_split[i]
    file_name = image_files[sample]
    src = images_path + file_name 
    dst = parent_dir_test + str(label) + '/' + file_name
    copyfile(src,dst)
    