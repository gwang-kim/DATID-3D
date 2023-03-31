
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import torch
import legacy
from torchvision.transforms import transforms
from projector import w_projector,w_plus_projector
from PIL import Image
from glob import glob
from os.path import join as opj

@click.command()
@click.option('--image_path', help='path of image file or image directory', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True,  default='test-runs', metavar='STR', show_default=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
              default='w', show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type:str,
        image_path:str,
        c_path:str,
        num_steps:int
):
    os.makedirs(outdir, exist_ok=True)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']

    G =  G.to(device)
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr


    if os.path.isdir(image_path):
        img_paths = sorted(glob(opj(image_path,"*.png")))
    else:
        img_paths = [image_path]

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512, 512))
    ])


    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img_id = os.path.split(img_path)[-1].split('.')[0]
        img.save(f'{outdir}/{img_id}_orig.png')
        c = np.load(img_path.replace('png','npy'))
        c = np.reshape(c,(1,25))
        c = torch.FloatTensor(c).cuda()

        from_im = trans(img).cuda()
        id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255

        if latent_space_type == 'w':
            w = w_projector.project(G, c, outdir,id_image, device=torch.device('cuda'), w_avg_samples=600, num_steps = num_steps, w_name=img_id)
        else:
            w = w_plus_projector.project(G, c,outdir, id_image, device=torch.device('cuda'), w_avg_samples=600, w_name=img_id, num_steps = num_steps )

        result_img = G.synthesis(w, c, noise_mode='const')['image']
        vis_img = (result_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{img_id}_inv.png')

        torch.save(w.detach().cpu(), f'{outdir}/{img_id}_inv.pt')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------



