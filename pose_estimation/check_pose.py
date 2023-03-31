import numpy as np
import scipy.io
def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0],
                   [0, cos(phi), sin(phi)],
                   [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
                   [0, 1, 0],
                   [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0],
                   [-sin(psi), cos(psi), 0],
                   [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R


import json
import os
from glob import glob
import sys

temp_folder = sys.argv[1]
output_folder = sys.argv[2]

import shutil
pose_template = np.load('util/pose_template.npy')
glob_names= sorted(glob('%s/cropped_images/*.png'%(temp_folder)))

for name_all in glob_names:

 if os.path.isfile('%s/cropped_images/cameras.json'%(temp_folder)):
       with open('%s/cropped_images/cameras.json'%(temp_folder), 'r') as file:     
              labels = json.load(file)#['labels']
              predict_pose = labels
 name = os.path.basename(name_all)[:-4]
 pose= np.array(predict_pose[name+'.png']['pose']).reshape(16)
 pose_template[:16] = pose

 np.save('%s/'%(output_folder)+name+'.npy',pose_template)
 shutil.copy('%s/cropped_images/'%(temp_folder)+name+'.png','%s/'%(output_folder)+name+'.png')


