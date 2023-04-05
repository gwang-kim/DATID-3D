# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', help='Network path', multiple=True, required=True)
@click.option('--w_pth', help='latent path')
@click.option('--generator_type', help='Generator type', type=click.Choice(['ffhq', 'cat']), required=False, metavar='STR', default='ffhq', show_default=True)
@click.option('--model_is_state_dict', type=bool, default=False)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--shape_only_first', type=bool, default=False)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape_format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
def generate_images(
    network: List[str],
    w_pth: str,
    generator_type: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    model_is_state_dict: bool,
    shape_only_first: bool,
):


    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda')

    if generator_type == 'ffhq':
        network_pkl_tmp = 'pretrained/ffhqrebalanced512-128.pkl'
    elif generator_type == 'cat':
        network_pkl_tmp = 'pretrained/afhqcats512-128.pkl'
    else:
        NotImplementedError()

    G_list = []
    outputs = []
    for network_path in network:
        print('Loading networks from "%s"...' % network_path)
        dir_label = network_path.split('/')[-2] + '___' + network_path.split('/')[-1]
        output = os.path.join(outdir, dir_label)
        outputs.append(output)
        if model_is_state_dict:
            with dnnlib.util.open_url(network_pkl_tmp) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
            ckpt = torch.load(network_path)
            G.load_state_dict(ckpt, strict=False)
        else:
            with dnnlib.util.open_url(network_path) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'])
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'])

        if generator_type == 'cat':
            G.rendering_kwargs['avg_camera_pivot'] = [0, 0, -0.06]
        elif generator_type == 'ffhq':
            G.rendering_kwargs['avg_camera_pivot'] = [0, 0, 0.2]

        G_list.append(G)

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    if w_pth is not None:
        seeds = [0]
    seed_idx = ''
    for i, seed in enumerate(seeds):
        if i < len(seeds) - 1:
            seed_idx += f'{seed}_'
        else:
            seed_idx += f'{seed}'

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    print(seeds)

    # Generate images.
    for G, output in zip(G_list, outputs):
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))

            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            imgs = []
            angle_p = -0.2
            for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                if w_pth is not None:
                    ws = torch.load(w_pth).cuda()
                    w_given_id = os.path.split(w_pth)[-1].split('.')[-2]
                    output_img = output + f'__{w_given_id}.png'
                    output_shape = output + f'__{w_given_id}.mrc'
                else:
                    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi,  truncation_cutoff=truncation_cutoff)
                    output_img = output + f'__{seed_idx:05d}.png'
                    output_shape = output + f'__{seed_idx:05d}.mrc'


                img = G.synthesis(ws, camera_params)['image']

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)

            img = torch.cat(imgs, dim=2)

            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_img)
            if shape_only_first and seed_idx != 0:
                continue


            if shapes:
                # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
                max_batch=1000000

                samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
                samples = samples.to(z.device)
                sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
                transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
                transformed_ray_directions_expanded[..., -1] = -1

                head = 0
                with tqdm(total = samples.shape[1]) as pbar:
                    with torch.no_grad():
                        while head < samples.shape[1]:
                            torch.manual_seed(0)
                            sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                            sigmas[:, head:head+max_batch] = sigma
                            head += max_batch
                            pbar.update(max_batch)

                sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                sigmas = np.flip(sigmas, 0)

                # Trim the border of the extracted cube
                pad = int(30 * shape_res / 256)
                pad_value = -1000
                sigmas[:pad] = pad_value
                sigmas[-pad:] = pad_value
                sigmas[:, :pad] = pad_value
                sigmas[:, -pad:] = pad_value
                sigmas[:, :, :pad] = pad_value
                sigmas[:, :, -pad:] = pad_value


                if shape_format == '.ply':
                    from shape_utils import convert_sdf_samples_to_ply
                    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, output_shape.replace('.mrc','.ply'), level=10)
                elif shape_format == '.mrc': # output mrc
                    with mrcfile.new_mmap(output_shape, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                        mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
