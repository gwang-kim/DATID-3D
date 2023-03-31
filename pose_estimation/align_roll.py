import multiprocessing
import os
import re
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict
import cv2
from tqdm import tqdm
import multiprocessing
import scipy.io

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------
import sys
name = sys.argv[1]
custom_folder = sys.argv[2]
temp_folder = sys.argv[3]
lm_file = open('%s/%s_lm2d.txt'%(custom_folder,name),'r')
lm = np.zeros((68,2),dtype=np.float32)
lines = lm_file.readlines()
for i in range(68):
    lm[i,0] = lines[i].strip().split(' ')[0]
    lm[i,1] = lines[i].strip().split(' ')[1]
#print(lm)
#load_model = scipy.io.loadmat('/disk1/jiaxin/eg3d/eg3d/eg3d-pose-detection/align1500_test/epoch_20_000000/jenny.mat')

#import pdb;pdb.set_trace()
#json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

#----------------------------------------------------------------------------

def process_image(lm):#item_idx, item, dst_dir="realign1500", output_size=1500, transform_size=4096, enable_padding=True):

    output_size = 1300
    transform_size =4096
    enable_padding = True



    # Parse landmarks.
    # pylint: disable=unused-variable

    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    q_scale = 1.8
    x = q_scale * x
    y = q_scale * y
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    src_file ='%s/%s.jpg'%(custom_folder,name) 
    if not os.path.exists(src_file):
        src_file ='%s/%s.png'%(custom_folder,name) 
    img = PIL.Image.open(src_file)
    print(img.size)
    import time

    # Shrink.
    start_time = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # print("shrink--- %s seconds ---" % (time.time() - start_time))

    # Crop.
    start_time = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    # print("crop--- %s seconds ---" % (time.time() - start_time))

    # Pad.
    start_time = time.time()
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        low_res = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
        blur = qsize * 0.02*0.1
        low_res = scipy.ndimage.gaussian_filter(low_res, [blur, blur, 0])
        low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LANCZOS4)
        img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)
        median = np.median(median, axis=(0,1))
        img += (median - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # print("pad--- %s seconds ---" % (time.time() - start_time))

    # Transform.
    start_time = time.time()
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    # print("transform--- %s seconds ---" % (time.time() - start_time))

    # Save aligned image.
    os.makedirs('%s/'%(temp_folder),exist_ok=True)
    img.save('%s/%s.png'%(temp_folder,name))
   
process_image(lm)