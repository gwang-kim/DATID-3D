import argparse
import os
import json

import numpy as np
from PIL import Image
from tqdm import tqdm

# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=1024., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size
    img = img.resize((w, h), resample=Image.LANCZOS)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.LANCZOS)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])
    return img, lm, mask


# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=1024., rescale_factor=466.285):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    #img_new = img.resize((1024,1024),resample=Image.LANCZOS)
    #lm_new = lm*1024.0/512.0
    #mask_new=None
    # img.save("/home/koki/Projects/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/iphone/epoch_20_000000/img_new.jpg")    
    trans_params = np.array([w0, h0, s, t[0], t[1]])
    lm_new *= 224/1024.0
    img_new_low = img_new.resize((224, 224), resample=Image.LANCZOS)

    return trans_params, img_new_low, lm_new, mask_new, img_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--compress_level', type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.indir, 'cropping_params.json')) as f:
        cropping_params = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    for im_path, cropping_dict in tqdm(cropping_params.items()):
        im = Image.open(os.path.join(args.indir, im_path)).convert('RGB')

        _, H = im.size
        lm = np.array(cropping_dict['lm'])
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        _, im_pil, lm, _, im_high = align_img(im, lm, np.array(cropping_dict['lm3d_std']), rescale_factor=cropping_dict['rescale_factor'])

        left = int(im_high.size[0]/2 - cropping_dict['center_crop_size']/2)
        upper = int(im_high.size[1]/2 - cropping_dict['center_crop_size']/2)
        right = left + cropping_dict['center_crop_size']
        lower = upper + cropping_dict['center_crop_size']
        im_cropped = im_high.crop((left, upper, right,lower))
        im_cropped = im_cropped.resize((cropping_dict['output_size'], cropping_dict['output_size']), resample=Image.LANCZOS)

        im_cropped.save(os.path.join(args.outdir, os.path.basename(im_path)), compress_level=args.compress_level)