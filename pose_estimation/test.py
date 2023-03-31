"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch
import json

def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True, rescale_factor=466.285): 
    im = Image.open(im_path).convert('RGB')
    _, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im_pil, lm, _, im_high = align_img(im, lm, lm3d_std, rescale_factor=rescale_factor)
    if to_tensor:
        im = torch.tensor(np.array(im_pil)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    else:
        im = im_pil
    return im, lm, im_pil, im_high

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)
    print("ROOT")
    print(name)
    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder)

    cropping_params = {}

    out_dir_crop1024 = os.path.join(name, "crop_1024")
    if not os.path.exists(out_dir_crop1024):
        os.makedirs(out_dir_crop1024)
    out_dir = os.path.join(name, 'epoch_%s_%06d'%(opt.epoch, 0))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            continue
        
        # 2 passes for cropping image for NeRF and for pose extraction
        for r in range(2):
            if r==0:
                rescale_factor = 300 # optimized for NeRF training
                center_crop_size = 700
                output_size = 512

                # left = int(im_high.size[0]/2 - center_crop_size/2)
                # upper = int(im_high.size[1]/2 - center_crop_size/2)
                # right = left + center_crop_size
                # lower = upper + center_crop_size
                # im_cropped = im_high.crop((left, upper, right,lower))
                # im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
                cropping_params[os.path.basename(im_path[i])] = {
                    'lm': np.loadtxt(lm_path[i]).astype(np.float32).tolist(),
                    'lm3d_std': lm3d_std.tolist(),
                    'rescale_factor': rescale_factor,
                    'center_crop_size': center_crop_size,
                    'output_size': output_size}

                # im_high.save(os.path.join(out_dir_crop1024, img_name+'.png'), compress_level=0)
                # im_cropped.save(os.path.join(out_dir_crop1024, img_name+'.png'), compress_level=0)
            elif not opt.skip_model:
                rescale_factor = 466.285
                im_tensor, lm_tensor, _, im_high = read_data(im_path[i], lm_path[i], lm3d_std, rescale_factor=rescale_factor)
             
                data = {
                    'imgs': im_tensor,
                    'lms': lm_tensor
                }
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
              #  visuals = model.get_current_visuals()  # get image results
              #  visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
                #    save_results=True, count=i, name=img_name, add_image=False)
             #   import pdb; pdb.set_trace()
                model.save_mesh(os.path.join(out_dir,img_name+'.obj'))
                model.save_coeff(os.path.join(out_dir,img_name+'.mat')) # save predicted coefficients

    with open(os.path.join(name, 'cropping_params.json'), 'w') as outfile:
        json.dump(cropping_params, outfile, indent=4)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt,opt.img_folder)
    
