import os
import argparse

### Parameters
parser = argparse.ArgumentParser()

# For all
parser.add_argument('--mode', type=str, required=True, choices=['pdg', 'ft', 'both'],
                    help="pdg: Pose-aware dataset generation, ft: Fine-tuning 3D generative models, both: Doing both")
parser.add_argument('--down_src_eg3d_from_nvidia', default=True)
# Pose-aware dataset generation
parser.add_argument('--pdg_prompt', type=str, required=True)
parser.add_argument('--pdg_generator_type', default='ffhq', type=str, choices=['ffhq', 'cat'])  # ffhq, cat
parser.add_argument('--pdg_strength', default=0.7, type=float)
parser.add_argument('--pdg_guidance_scale', default=8, type=float)
parser.add_argument('--pdg_num_images', default=1000, type=int)
parser.add_argument('--pdg_sd_model_id', default='stabilityai/stable-diffusion-2-1-base', type=str)
parser.add_argument('--pdg_num_inference_steps', default=50, type=int)
parser.add_argument('--pdg_name_tag', default='', type=str)
parser.add_argument('--down_src_eg3d_from_nvidia', default=True)
# Fine-tuning 3D generative models
parser.add_argument('--ft_generator_type', default='same', help="None: The same type as pdg_generator_type", type=str, choices=['ffhq', 'cat', 'same'])
parser.add_argument('--ft_kimg', default=200, type=int)
parser.add_argument('--ft_batch', default=20, type=int)
parser.add_argument('--ft_tick', default=1, type=int)
parser.add_argument('--ft_snap', default=50, type=int)
parser.add_argument('--ft_outdir', default='../training_runs', type=str) #
parser.add_argument('--ft_gpus', default=1, type=str) #
parser.add_argument('--ft_workers', default=8, type=int) #
parser.add_argument('--ft_data_max_size', default=500000000, type=int) #
parser.add_argument('--ft_freeze_dec_sr', default=True, type=bool) #

args = parser.parse_args()


### Pose-aware target generation
if args.mode in ['pdg', 'both']:
    os.chdir('eg3d')
    if args.pdg_generator_type == 'cat':
        pdg_generator_id = 'afhqcats512-128.pkl'
    else:
        pdg_generator_id = 'ffhqrebalanced512-128.pkl'

    pdg_generator_path = f'pretrained/{pdg_generator_id}'
    if not os.path.exists(pdg_generator_path):
        os.makedirs(f'pretrained', exist_ok=True)
        print("Pretrained EG3D model cannot be found. Downloading the pretrained EG3D models.")
        if args.down_src_eg3d_from_nvidia == True:
            os.system(f'wget -c https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/{pdg_generator_id} -O {pdg_generator_path}')
        else:
            os.system(f'wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/nvidia_{pdg_generator_id} -O {pdg_generator_path}')
    command = f"""python datid3d_data_gen.py  \
   --prompt="{args.pdg_prompt}" \
   --data_type={args.pdg_generator_type} \
   --strength={args.pdg_strength} \
   --guidance_scale={args.pdg_guidance_scale} \
   --num_images={args.pdg_num_images} \
   --sd_model_id="{args.pdg_sd_model_id}" \
   --num_inference_steps={args.pdg_num_inference_steps} \
   --name_tag={args.pdg_name_tag} 
    """
    print(f"{command} \n")
    os.system(command)
    os.chdir('..')

### Filtering process
# TODO


### Fine-tuning 3D generative models
if args.mode in ['ft', 'both']:
    os.chdir('eg3d')
    if args.ft_generator_type == 'same':
        args.ft_generator_type = args.pdg_generator_type

    if args.ft_generator_type == 'cat':
        ft_generator_id = 'afhqcats512-128.pkl'
    else:
        ft_generator_id = 'ffhqrebalanced512-128.pkl'

    ft_generator_path = f'pretrained/{ft_generator_id}'
    if not os.path.exists(ft_generator_path):
        os.makedirs(f'pretrained', exist_ok=True)
        print("Pretrained EG3D model cannot be found. Downloading the pretrained EG3D models.")
        if args.down_src_eg3d_from_nvidia == True:
            os.system(f'wget -c https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/{ft_generator_id} -O {ft_generator_path}')
        else:
            os.system(f'wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/nvidia_{ft_generator_id} -O {ft_generator_path}')

    dataset_id = f'data_{args.pdg_generator_type}_{args.pdg_prompt.replace(" ", "_")}{args.pdg_name_tag}'
    dataset_path = f'./exp_data/{dataset_id}/{dataset_id}.zip'


    command = f"""python train.py  \
    --outdir={args.ft_outdir} \
    --cfg={args.ft_generator_type} \
    --data="{dataset_path}"   \
    --resume={ft_generator_path} --freeze_dec_sr={args.ft_freeze_dec_sr} \
    --batch={args.ft_batch} --workers={args.ft_workers} --gpus={args.ft_gpus} \
    --tick={args.ft_tick} --snap={args.ft_snap} --data_max_size={args.ft_data_max_size} --kimg={args.ft_kimg} \
    --gamma=5 --aug=ada --neural_rendering_resolution_final=128 --gen_pose_cond=True --gpc_reg_prob=0.8 --metrics=None 
    """
    print(f"{command} \n")
    os.system(command)
    os.chdir('..')
