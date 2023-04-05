import os
from os.path import join as opj
import argparse
from glob import glob

### Parameters
parser = argparse.ArgumentParser()

# For all
parser.add_argument('--mode', type=str, required=True, choices=['image', 'video', 'manip', 'manip_from_inv'],
                    help="image: Sample images and shapes, "
                         "video: Sample pose-controlled videos, "
                         "manip: Manipulated 3D reconstruction from images, "
                         "manip_from_inv: Manipulated 3D reconstruction from inverted latent")
parser.add_argument('--network', type=str, nargs='+', required=True)
parser.add_argument('--generator_type', default='ffhq', type=str, choices=['ffhq', 'cat'])  # ffhq, cat
parser.add_argument('--outdir', type=str, default='test_runs')
parser.add_argument('--trunc', type=float, default=0.7)
parser.add_argument('--seeds', type=str, default='100-200')
parser.add_argument('--down_src_eg3d_from_nvidia', default=True)
parser.add_argument('--num_inv_steps', default=300, type=int)
# Manipulated 3D reconstruction
parser.add_argument('--indir', type=str, default='input_imgs')
parser.add_argument('--name_tag', type=str, default='')
# Sample images
parser.add_argument('--shape', default=True)
parser.add_argument('--shape_format',  type=str, choices=['.mrc', '.ply'], default='.mrc')
parser.add_argument('--shape_only_first', type=bool, default=False)
# Sample pose-controlled videos
parser.add_argument('--grid', default='1x1')


args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print()


network_command = ''
for network_path in args.network:
    network_command += f"--network {opj('..', network_path)} "



### Sample images
if args.mode == 'image':
    image_path = opj(args.outdir, f'image{args.name_tag}')
    os.makedirs(image_path, exist_ok=True)

    os.chdir('eg3d')
    command = f"""python gen_samples.py \
    {network_command} \
    --seeds={args.seeds}  \
    --generator_type={args.generator_type} \
    --outdir={opj('..', image_path)} \
    --shapes={args.shape} \
    --shape_format={args.shape_format} \
    --shape_only_first={args.shape_only_first} \
    --trunc={args.trunc} \
    """
    print(f"{command} \n")
    os.system(command)
    os.chdir('..')





### Sample pose-controlled videos
if args.mode == 'video':
    video_path = opj(args.outdir, f'video{args.name_tag}')
    os.makedirs(video_path, exist_ok=True)

    os.chdir('eg3d')
    command = f"""python gen_videos.py \
    {network_command} \
    --seeds={args.seeds} \
    --generator_type={args.generator_type} \
    --outdir={opj('..', video_path)} \
    --shapes=False \
    --trunc={args.trunc} \
    --grid={args.grid}
    """
    print(f"{command} \n")
    os.system(command)
    os.chdir('..')


### Manipulated 3D reconstruction from images
if args.mode == 'manip':
    input_path = opj(args.indir)
    align_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '1_align_result')
    pose_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '2_pose_result')
    inversion_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '3_inversion_result')
    manip_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '4_manip_result')

    os.makedirs(opj(args.outdir, f'manip_3D_recon{args.name_tag}'), exist_ok=True)
    os.makedirs(align_path, exist_ok=True)
    os.makedirs(pose_path, exist_ok=True)
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(manip_path, exist_ok=True)

    os.chdir('eg3d')
    if args.generator_type == 'cat':
        generator_id = 'afhqcats512-128.pkl'
    else:
        generator_id = 'ffhqrebalanced512-128.pkl'
    generator_path = f'pretrained/{generator_id}'
    if not os.path.exists(generator_path):
        os.makedirs(f'pretrained', exist_ok=True)
        print("Pretrained EG3D model cannot be found. Downloading the pretrained EG3D models.")
        if args.down_src_eg3d_from_nvidia == True:
            os.system(f'wget -c https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/{generator_id} -O {generator_path}')
        else:
            os.system(f'wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/nvidia_{generator_id} -O {generator_path}')
    os.chdir('..')

    ## Align images and Pose extraction
    os.chdir('pose_estimation')
    if not os.path.exists('checkpoints/pretrained/epoch_20.pth') or not os.path.exists('BFM'):
        print(f"BFM and pretrained DeepFaceRecon3D model cannot be found. Downloading the pretrained pose estimation model and BFM files, put epoch_20.pth in ./pose_estimation/checkpoints/pretrained/ and put unzip BFM.zip in ./pose_estimation/.")

        try:
            from gdown import download as drive_download
            drive_download(f'https://drive.google.com/uc?id=1mdqkEUepHZROeOj99pXogAPJPqzBDN2G', './BFM.zip', quiet=False)
            os.system('unzip BFM.zip')
            drive_download(f'https://drive.google.com/uc?id=1zawY7jYDJlUGnSAXn1pgIHgIvJpiSmj5', './checkpoints/pretrained/epoch_20.pth', quiet=False)
        except:
            os.system("pip install -U --no-cache-dir gdown --pre")
            from gdown import download as drive_download
            drive_download(f'https://drive.google.com/uc?id=1mdqkEUepHZROeOj99pXogAPJPqzBDN2G', './BFM.zip', quiet=False)
            os.system('unzip BFM.zip')
            drive_download(f'https://drive.google.com/uc?id=1zawY7jYDJlUGnSAXn1pgIHgIvJpiSmj5', './checkpoints/pretrained/epoch_20.pth', quiet=False)

        print()
    command =  f"""python extract_pose.py 0 \
    {opj('..', input_path)} {opj('..', align_path)} {opj('..', pose_path)}
    """
    print(f"{command} \n")
    os.system(command)
    os.chdir('..')

    ## Invert images to the latent space of 3D GANs
    os.chdir('eg3d')
    command = f"""python run_inversion.py  \
    --outdir={opj('..', inversion_path)} \
    --latent_space_type=w_plus  \
    --network={generator_path} \
    --image_path={opj('..', pose_path)} \
    --num_steps={args.num_inv_steps}
    """
    print(f"{command} \n")
    os.system(command)
    os.chdir('..')

    ## Generate videos, images and mesh
    os.chdir('eg3d')
    w_pths = sorted(glob(opj('..', inversion_path, '*.pt')))
    if len(w_pths) == 0:
        print("No inverted latent")
        exit()
    for w_pth in w_pths:
        print(f"{w_pth} \n")

        command = f"""python gen_samples.py \
        {network_command} \
        --w_pth={w_pth} \
        --seeds='100-200' \
        --generator_type={args.generator_type} \
        --outdir={opj('..', manip_path)} \
        --shapes={args.shape} \
        --shape_format={args.shape_format} \
        --shape_only_first={args.shape_only_first} \
        --trunc={args.trunc} \
        """
        print(f"{command} \n")
        os.system(command)

        command = f"""python gen_videos.py \
         {network_command} \
        --w_pth={w_pth} \
        --seeds='100-200' \
        --generator_type={args.generator_type} \
        --outdir={opj('..', manip_path)} \
        --shapes=False \
        --trunc={args.trunc} \
        --grid=1x1 
        """
        print(f"{command} \n")
        os.system(command)
    os.chdir('..')





### Manipulated 3D reconstruction from inverted latent
if args.mode == 'manip_from_inv':
    input_path = opj(args.indir)
    align_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '1_align_result')
    pose_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '2_pose_result')
    inversion_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '3_inversion_result')
    manip_path = opj(args.outdir, f'manip_3D_recon{args.name_tag}', '4_manip_result')

    os.makedirs(opj(args.outdir, f'manip_3D_recon{args.name_tag}'), exist_ok=True)
    os.makedirs(align_path, exist_ok=True)
    os.makedirs(pose_path, exist_ok=True)
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(manip_path, exist_ok=True)

    ## Generate videos, images and mesh
    os.chdir('eg3d')
    w_pths = sorted(glob(opj('..', inversion_path, '*.pt')))
    if len(w_pths) == 0:
        print("No inverted latent")
        exit()
    for w_pth in w_pths:
        print(f"{w_pth} \n")

        command = f"""python gen_samples.py \
         {network_command} \
        --w_pth={w_pth} \
        --seeds='100-200' \
        --generator_type={args.generator_type} \
        --outdir={opj('..', manip_path)} \
        --shapes={args.shape} \
        --shape_format={args.shape_format} \
        --shape_only_first={args.shape_only_first} \
        --trunc={args.trunc} \
        """
        print(f"{command} \n")
        os.system(command)

        command = f"""python gen_videos.py \
         {network_command} \
        --w_pth={w_pth} \
        --seeds='100-200' \
        --generator_type={args.generator_type} \
        --outdir={opj('..', manip_path)} \
        --shapes=False \
        --trunc={args.trunc} \
        --grid=1x1 
        """
        print(f"{command} \n")
        os.system(command)
    os.chdir('..')


