import argparse
import gradio as gr
import os
import shutil
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image
import torchvision.transforms.functional as F
from functools import partial

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img.detach())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class Intermediate:
    def __init__(self):
        self.input_img = None
        self.input_img_cnt = 0


model_ckpts = {"elf": "ffhq-elf.pkl",
               "greek_statue": "ffhq-greek_statue.pkl",
               "hobbit": "ffhq-hobbit.pkl",
               "lego": "ffhq-lego.pkl",
               "masquerade": "ffhq-masquerade.pkl",
               "neanderthal": "ffhq-neanderthal.pkl",
               "orc": "ffhq-orc.pkl",
               "pixar": "ffhq-pixar.pkl",
               "skeleton": "ffhq-skeleton.pkl",
               "stone_golem": "ffhq-stone_golem.pkl",
               "super_mario": "ffhq-super_mario.pkl",
               "tekken": "ffhq-tekken.pkl",
               "yoda": "ffhq-yoda.pkl",
               "zombie": "ffhq-zombie.pkl",
               "cat_in_Zootopia": "cat-cat_in_Zootopia.pkl",
               "fox_in_Zootopia": "cat-fox_in_Zootopia.pkl",
               "golden_aluminum_animal": "cat-golden_aluminum_animal.pkl",
               }

manip_model_ckpts = {"super_mario": "ffhq-super_mario.pkl",
                     "lego": "ffhq-lego.pkl",
                     "neanderthal": "ffhq-neanderthal.pkl",
                     "orc": "ffhq-orc.pkl",
                     "pixar": "ffhq-pixar.pkl",
                     "skeleton": "ffhq-skeleton.pkl",
                     "stone_golem": "ffhq-stone_golem.pkl",
                     "tekken": "ffhq-tekken.pkl",
                     "greek_statue": "ffhq-greek_statue.pkl",
                     "yoda": "ffhq-yoda.pkl",
                     "zombie": "ffhq-zombie.pkl",
                     "elf": "ffhq-elf.pkl",
                   }


def TextGuidedImageTo3D(intermediate, img, model_name, num_inversion_steps, truncation):
    if img != intermediate.input_img:
        if os.path.exists('input_imgs_gradio'):
            shutil.rmtree('input_imgs_gradio')
        os.makedirs('input_imgs_gradio', exist_ok=True)
        img.save('input_imgs_gradio/input.png')
        intermediate.input_img = img
        intermediate.input_img_cnt += 1

    all_model_names = manip_model_ckpts.keys()
    generator_type = 'ffhq'

    if model_name == 'all':
        _no_video_models = []
        for _model_name in all_model_names:
            if not os.path.exists(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/finetuned___{model_ckpts[_model_name]}__input_inv.mp4'):
                print()
                _no_video_models.append(_model_name)

        model_names_command = ''
        for _model_name in _no_video_models:
            if not os.path.exists(f'finetuned/{model_ckpts[_model_name]}'):
                command = f"""wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/{model_ckpts[_model_name]} -O finetuned/{model_ckpts[_model_name]}
                """
                os.system(command)

            model_names_command += f"finetuned/{model_ckpts[_model_name]} "

        w_pths = sorted(glob(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/3_inversion_result/*.pt'))
        if len(w_pths) == 0:
            mode = 'manip'
        else:
            mode = 'manip_from_inv'

        if len(_no_video_models) > 0:
            command = f"""python datid3d_test.py --mode {mode} \
                      --indir='input_imgs_gradio' \
                      --generator_type={generator_type} \
                      --outdir='test_runs' \
                      --trunc={truncation} \
                      --network {model_names_command} \
                      --num_inv_steps={num_inversion_steps} \
                      --down_src_eg3d_from_nvidia=False \
                      --name_tag='_{intermediate.input_img_cnt}' \
                      --shape=False"""
            print(command)
            os.system(command)

        aligned_img_pth = sorted(glob(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/2_pose_result/*.png'))[0]
        aligned_img = Image.open(aligned_img_pth)

        result_imgs = []
        for _model_name in all_model_names:
            img_pth = f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/finetuned___{model_ckpts[_model_name]}__input_inv.png'
            result_imgs.append(read_image(img_pth))

        result_grid_pt = make_grid(result_imgs, nrow=1)
        result_img = F.to_pil_image(result_grid_pt)
    else:
        if not os.path.exists(f'finetuned/{model_ckpts[model_name]}'):
            command = f"""wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/{model_ckpts[model_name]} -O finetuned/{model_ckpts[model_name]}
            """
            os.system(command)

        if not os.path.exists(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/finetuned___{model_ckpts[model_name]}__input_inv.mp4'):
            w_pths = sorted(glob(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/3_inversion_result/*.pt'))
            if len(w_pths) == 0:
                mode = 'manip'
            else:
                mode = 'manip_from_inv'

            command = f"""python datid3d_test.py --mode {mode} \
          --indir='input_imgs_gradio' \
          --generator_type={generator_type} \
          --outdir='test_runs' \
          --trunc={truncation} \
          --network finetuned/{model_ckpts[model_name]} \
          --num_inv_steps={num_inversion_steps} \
          --down_src_eg3d_from_nvidia=0 \
          --name_tag='_{intermediate.input_img_cnt}' \
          --shape=False"""
            print(command)
            os.system(command)

        aligned_img_pth = sorted(glob(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/2_pose_result/*.png'))[0]
        aligned_img = Image.open(aligned_img_pth)

        result_img_pth = sorted(glob(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/*{model_ckpts[model_name]}*.png'))[0]
        result_img = Image.open(result_img_pth)




    if model_name=='all':
        result_video_pth = f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/finetuned___ffhq-all__input_inv.mp4'
        if os.path.exists(result_video_pth):
            os.remove(result_video_pth)
        command = 'ffmpeg '
        for _model_name in all_model_names:
            command += f'-i test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/finetuned___ffhq-{_model_name}.pkl__input_inv.mp4 '
        command += '-filter_complex "[0:v]scale=2*iw:-1[v0];[1:v]scale=2*iw:-1[v1];[2:v]scale=2*iw:-1[v2];[3:v]scale=2*iw:-1[v3];[4:v]scale=2*iw:-1[v4];[5:v]scale=2*iw:-1[v5];[6:v]scale=2*iw:-1[v6];[7:v]scale=2*iw:-1[v7];[8:v]scale=2*iw:-1[v8];[9:v]scale=2*iw:-1[v9];[10:v]scale=2*iw:-1[v10];[11:v]scale=2*iw:-1[v11];[v0][v1][v2][v3][v4][v5][v6][v7][v8][v9][v10][v11]xstack=inputs=12:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w4_h0|w4+w5_h0|w4+w5+w6_h0|0_h0+h4|w8_h0+h4|w8+w9_h0+h4|w8+w9+w10_h0+h4" '
        command += f" -vcodec libx264 {result_video_pth}"
        print()
        print(command)
        os.system(command)

    else:
        result_video_pth = sorted(glob(f'test_runs/manip_3D_recon_{intermediate.input_img_cnt}/4_manip_result/*{model_ckpts[model_name]}*.mp4'))[0]

    return aligned_img, result_img, result_video_pth


def SampleImage(model_name, num_samples, truncation, seed):
    seed_list = np.random.RandomState(seed).choice(np.arange(10000), num_samples).tolist()
    seeds = ''
    for seed in seed_list:
        seeds += f'{seed},'
    seeds = seeds[:-1]

    if model_name in ["fox_in_Zootopia", "cat_in_Zootopia", "golden_aluminum_animal"]:
        generator_type = 'cat'
    else:
        generator_type = 'ffhq'

    if not os.path.exists(f'finetuned/{model_ckpts[model_name]}'):
        command = f"""wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/{model_ckpts[model_name]} -O finetuned/{model_ckpts[model_name]}
        """
        os.system(command)

    command = f"""python datid3d_test.py --mode image \
    --generator_type={generator_type} \
    --outdir='test_runs' \
    --seeds={seeds} \
    --trunc={truncation} \
    --network=finetuned/{model_ckpts[model_name]} \
    --shape=False"""
    print(command)
    os.system(command)

    result_img_pths = sorted(glob(f'test_runs/image/*{model_ckpts[model_name]}*.png'))
    result_imgs = []
    for img_pth in result_img_pths:
        result_imgs.append(read_image(img_pth))

    result_grid_pt = make_grid(result_imgs, nrow=1)
    result_grid_pil = F.to_pil_image(result_grid_pt)
    return result_grid_pil




def SampleVideo(model_name, grid_height, truncation, seed):
    seed_list = np.random.RandomState(seed).choice(np.arange(10000), grid_height**2).tolist()
    seeds = ''
    for seed in seed_list:
        seeds += f'{seed},'
    seeds = seeds[:-1]

    if model_name in ["fox_in_Zootopia", "cat_in_Zootopia", "golden_aluminum_animal"]:
        generator_type = 'cat'
    else:
        generator_type = 'ffhq'

    if not os.path.exists(f'finetuned/{model_ckpts[model_name]}'):
        command = f"""wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/{model_ckpts[model_name]} -O finetuned/{model_ckpts[model_name]}
        """
        os.system(command)

    command = f"""python datid3d_test.py --mode video \
    --generator_type={generator_type} \
    --outdir='test_runs' \
    --seeds={seeds} \
    --trunc={truncation} \
    --grid={grid_height}x{grid_height} \
    --network=finetuned/{model_ckpts[model_name]} \
    --shape=False"""
    print(command)
    os.system(command)

    result_video_pth = sorted(glob(f'test_runs/video/*{model_ckpts[model_name]}*.mp4'))[0]

    return result_video_pth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help="public url")
    args = parser.parse_args()

    demo = gr.Blocks(title="DATID-3D Interactive Demo")
    os.makedirs('finetuned', exist_ok=True)
    intermediate = Intermediate()
    with demo:
        gr.Markdown("# DATID-3D Interactive Demo")
        gr.Markdown(
            "### Demo of the CVPR 2023 paper \"DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model\"")

        with gr.Tab("Text-guided Manipulated 3D reconstruction"):
            gr.Markdown("Text-guided Image-to-3D Translation")
            with gr.Row():
                with gr.Column(scale=1, variant='panel'):
                    t_image_input = gr.Image(source='upload', type="pil", interactive=True)

                    t_model_name = gr.Radio(["super_mario", "lego", "neanderthal", "orc",
                                             "pixar", "skeleton", "stone_golem","tekken",
                                             "greek_statue", "yoda", "zombie", "elf", "all"],
                                             label="Model fine-tuned through DATID-3D",
                                             value="super_mario", interactive=True)
                    with gr.Accordion("Advanced Options", open=False):
                        t_truncation = gr.Slider(label="Truncation psi", minimum=0, maximum=1.0, step=0.01, randomize=False, value=0.8)
                        t_num_inversion_steps = gr.Slider(300, 1000, value=300, step=1, label='Number of steps for the invresion')
                    with gr.Row():
                        t_button_gen_result = gr.Button("Generate Result", variant='primary')
                        # t_button_gen_video = gr.Button("Generate Video", variant='primary')
                        # t_button_gen_image = gr.Button("Generate Image", variant='secondary')
                    with gr.Row():
                        t_align_image_result = gr.Image(label="Alignment result", interactive=False)
                with gr.Column(scale=1, variant='panel'):
                    with gr.Row():
                        t_video_result = gr.Video(label="Video result", interactive=False)

                    with gr.Row():
                        t_image_result = gr.Image(label="Image result", interactive=False)


        with gr.Tab("Sample Images"):
            with gr.Row():
                with gr.Column(scale=1, variant='panel'):
                    i_model_name = gr.Radio(
                        ["elf", "greek_statue", "hobbit", "lego", "masquerade", "neanderthal", "orc", "pixar",
                         "skeleton", "stone_golem", "super_mario", "tekken", "yoda", "zombie", "fox_in_Zootopia",
                         "cat_in_Zootopia", "golden_aluminum_animal"],
                        label="Model fine-tuned through DATID-3D",
                        value="super_mario", interactive=True)
                    i_num_samples = gr.Slider(0, 20, value=4, step=1, label='Number of samples')
                    i_seed = gr.Slider(label="Seed", minimum=0, maximum=1000000000, step=1, value=1235)
                    with gr.Accordion("Advanced Options", open=False):
                        i_truncation = gr.Slider(label="Truncation psi", minimum=0, maximum=1.0, step=0.01, randomize=False, value=0.8)
                    with gr.Row():
                        i_button_gen_image = gr.Button("Generate Image", variant='primary')
                with gr.Column(scale=1, variant='panel'):
                    with gr.Row():
                        i_image_result = gr.Image(label="Image result", interactive=False)


        with gr.Tab("Sample Videos"):
            with gr.Row():
                with gr.Column(scale=1, variant='panel'):
                    v_model_name = gr.Radio(
                        ["elf", "greek_statue", "hobbit", "lego", "masquerade", "neanderthal", "orc", "pixar",
                         "skeleton", "stone_golem", "super_mario", "tekken", "yoda", "zombie", "fox_in_Zootopia",
                         "cat_in_Zootopia", "golden_aluminum_animal"],
                        label="Model fine-tuned through DATID-3D",
                        value="super_mario", interactive=True)
                    v_grid_height = gr.Slider(0, 5, value=2, step=1,label='Height of the grid')
                    v_seed = gr.Slider(label="Seed", minimum=0, maximum=1000000000, step=1, value=1235)
                    with gr.Accordion("Advanced Options", open=False):
                        v_truncation = gr.Slider(label="Truncation psi", minimum=0, maximum=1.0, step=0.01, randomize=False,
                                                 value=0.8)

                    with gr.Row():
                        v_button_gen_video = gr.Button("Generate Video", variant='primary')

                with gr.Column(scale=1, variant='panel'):

                    with gr.Row():
                        v_video_result = gr.Video(label="Video result", interactive=False)





        # functions
        t_button_gen_result.click(fn=partial(TextGuidedImageTo3D, intermediate),
                                  inputs=[t_image_input, t_model_name, t_num_inversion_steps, t_truncation],
                                  outputs=[t_align_image_result, t_image_result, t_video_result])
        i_button_gen_image.click(fn=SampleImage,
                              inputs=[i_model_name, i_num_samples, i_truncation, i_seed],
                              outputs=[i_image_result])
        v_button_gen_video.click(fn=SampleVideo,
                                 inputs=[i_model_name, v_grid_height, v_truncation, v_seed],
                                 outputs=[v_video_result])

    demo.queue(concurrency_count=1)
    demo.launch(share=args.share)

