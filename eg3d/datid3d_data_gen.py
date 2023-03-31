
import sys, os
sys.path.append(os.getcwd())
from os.path import join as opj
import zipfile
import json
import pickle
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from torchvision.transforms import ToPILImage
from diffusers import StableDiffusionImg2ImgPipeline, PNDMScheduler
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Pose-aware dataset generation')
    parser.add_argument('--strength', default=0.7, type=float)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--data_type', default='ffhq', type=str) # ffhq, cat
    parser.add_argument('--guidance_scale', default=8, type=float)
    parser.add_argument('--num_images', default=1000, type=int)
    parser.add_argument('--sd_model_id', default='stabilityai/stable-diffusion-2-1-base', type=str)
    parser.add_argument('--num_inference_steps', default=30, type=int)
    parser.add_argument('--ffhq_eg3d_path', default='pretrained/ffhqrebalanced512-128.pkl', type=str)
    parser.add_argument('--cat_eg3d_path', default='pretrained/afhqcats512-128.pkl', type=str)
    parser.add_argument('--ffhq_pivot', default=0.2, type=float)
    parser.add_argument('--cat_pivot', default=0.05, type=float)
    parser.add_argument('--pitch_range', default=0.3, type=float)
    parser.add_argument('--yaw_range', default=0.3, type=float)
    parser.add_argument('--name_tag', default='', type=str)
    parser.add_argument('--seed', default=15, type=int)

    args = parser.parse_args()
    return args

def make_zip(base_dir, prompt, data_type='ffhq', name_tag=''):
    base_dir = os.path.abspath(base_dir)

    owd = os.path.abspath(os.getcwd())
    os.chdir(base_dir)

    json_path = opj(base_dir, "dataset.json")

    zip_path = opj(base_dir, f'data_{data_type}_{prompt.replace(" ", "_")}{name_tag}.zip')
    zip_file = zipfile.ZipFile(zip_path, "w")

    with open(json_path, 'r') as file:
        data = json.load(file)
    zip_file.write(os.path.relpath(json_path, base_dir), compress_type=zipfile.ZIP_STORED)

    for label in data['labels']:
        trg_img_path = label[0]
        zip_file.write(trg_img_path, compress_type=zipfile.ZIP_STORED)

    zip_file.close()
    os.chdir(owd)

def pts2pil(pts):
    pts = (pts + 1) / 2
    pts[pts > 1] = 1
    pts[pts < 0] = 0
    return ToPILImage()(pts[0])

if __name__ == '__main__':
    args = parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_type = args.data_type
    prompt = args.prompt
    strength = args.strength
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    num_images = args.num_images
    name_tag = args.name_tag

    # 3DG options
    ffhq_eg3d_path = args.ffhq_eg3d_path
    cat_eg3d_path = args.cat_eg3d_path
    cat_pivot = args.cat_pivot
    ffhq_pivot = args.ffhq_pivot
    pitch_range = args.pitch_range
    yaw_range = args.yaw_range
    num_frames = 240
    truncation_psi = 0.7
    truncation_cutoff = 14
    fov_deg = 18.837
    ft_img_size = 512

    # Load 3DG
    eg3d_path = None
    if data_type == 'ffhq':
        eg3d_path = args.ffhq_eg3d_path
        pivot = ffhq_pivot
    elif data_type == 'cat':
        eg3d_path = args.cat_eg3d_path
        pivot = cat_pivot

    with open(eg3d_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
    G.train()
    for param in G.parameters():
        param.requires_grad_(True)

    # SD options
    model_id = args.sd_model_id
    negative_prompt = None
    eta = 0.0
    batch_size = 1
    model_inversion = False

    # Load SD
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
        scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                num_train_timesteps=1000, set_alpha_to_one=False, steps_offset=1, skip_prk_steps=1),
    ).to(device)
    pipe.safety_checker = None
    print('SD model is loaded')

    # Outputs directory
    base_dir = opj(f'./exp_data/data_{data_type}_{prompt.replace(" ", "_")}{name_tag}')

    src_img_dir = opj(base_dir, "src_imgs")
    trg_img_dir = opj(base_dir, "trg_imgs")

    os.makedirs('exp_data', exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(src_img_dir, exist_ok=True)
    os.makedirs(trg_img_dir, exist_ok=True)
    labels = []

    # Fine-tuning 3D generator
    for i in tqdm(range(num_images)):
        G.eval()
        z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        with torch.no_grad():
            yaw_idx = np.random.randint(num_frames)
            pitch_idx = np.random.randint(num_frames)

            cam_pivot = torch.tensor([0, 0, pivot], device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + yaw_range * np.sin(2 * np.pi * yaw_idx / num_frames),
                                                      np.pi / 2 - 0.05 + pitch_range * np.cos(
                                                          2 * np.pi * pitch_idx / num_frames),
                                                      cam_pivot, radius=cam_radius, device=device,
                                                      batch_size=batch_size)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                                   device=device, batch_size=batch_size)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(batch_size, 1)],
                                      1)
            conditioning_params = torch.cat(
                [conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(batch_size, 1)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

            img_pts = G.synthesis(ws, camera_params)['image']

            src_img_pts = img_pts.detach()
            src_img_pts = F.interpolate(src_img_pts, (ft_img_size, ft_img_size), mode='bilinear', align_corners=False)
            with autocast("cuda"):
                trg_img_pil = pipe(prompt=prompt,
                                   image=src_img_pts,
                                   strength=strength,
                                   guidance_scale=guidance_scale,
                                   num_inference_steps=num_inference_steps,
                                   )['images'][0]

        src_idx = f'{i:05d}_src.png'
        trg_idx = f'{i:05d}_trg.png'

        src_img_pil_path = opj(src_img_dir, src_idx)
        trg_img_pil_path = opj(trg_img_dir, trg_idx)

        src_img_pil = pts2pil(src_img_pts.cpu())

        src_img_pil.save(src_img_pil_path)
        trg_img_pil.save(trg_img_pil_path)

        label = [trg_img_pil_path.replace(base_dir, '').replace('/trg_', 'trg_'), camera_params[0].tolist()]

        labels.append(label)


    json_path = opj(base_dir, "dataset.json")
    json_data = {'labels': labels}
    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    make_zip(base_dir, prompt, data_type, name_tag)
