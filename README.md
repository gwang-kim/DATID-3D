## DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model<br><sub>Official PyTorch implementation of the CVPR 2023 paper</sub>



[//]: # ([![Open In Spaces]&#40;https://camo.githubusercontent.com/00380c35e60d6b04be65d3d94a58332be5cc93779f630bcdfc18ab9a3a7d3388/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565&#41;]&#40;https://huggingface.co/&#41;)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e9NSVB7x_hjz-nr4K0jO4rfTXILnNGtA?usp=sharing) 
[![project_page](https://img.shields.io/badge/-project%20page-green)](https://gwang-kim.github.io/datid_3d/) [![arXiv](https://img.shields.io/badge/arXiv-2211.16374-red)](https://arxiv.org/abs/2211.16374) 
 

[//]: # ()
[//]: # ([![arXiv]&#40;https://img.shields.io/badge/paper-cvpr2022-cyan&#41;]&#40;https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html&#41; [![arXiv]&#40;https://img.shields.io/badge/arXiv-2110.02711-red&#41;]&#40;https://arxiv.org/abs/2110.02711&#41;)

[//]: # ([![video]&#40;https://img.shields.io/badge/video-green&#41;]&#40;https://youtu.be/YVCtaXw6fw8&#41; [![poster]&#40;https://img.shields.io/badge/poster-orange&#41;]&#40;https://drive.google.com/file/d/1QgRFIRba492dCZ6v7BcZB9zqyp91aTjL/view?usp=sharing&#41; )

<p align="center">
  <img src="assets/datid3d_result.gif"/>
</p> 

> **DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model**<br>
> [Gwanghyun Kim](https://gwang-kim.github.io/), [Se Young Chun](https://icl.snu.ac.kr/pi) <br>
> CVPR 2023 <br>
> 
> [gwang-kim.github.io/datid_3d](gwang-kim.github.io/datid_3d/)
> 
>**Abstract**: <br>
Recent 3D generative models have achieved remarkable performance in synthesizing high resolution photorealistic images with view consistency and detailed 3D shapes, but training them for diverse domains is challenging since it requires massive training images and their camera distribution information.  
Text-guided domain adaptation methods have shown impressive performance on converting the 2D generative model on one domain into the models on other domains with different styles by leveraging the CLIP (Contrastive Language-Image Pre-training), rather than collecting massive datasets for those domains. However, one drawback of them is that the sample diversity in the original generative model is not well-preserved in the domain-adapted generative models due to the deterministic nature of the CLIP text encoder. Text-guided domain adaptation will be even more challenging for 3D generative models not only because of catastrophic diversity loss, but also because of inferior text-image correspondence and poor image quality. 
**Here we propose DATID-3D, a novel pipeline of text-guided domain adaptation tailored for 3D generative models using text-to-image diffusion models that can synthesize diverse images per text prompt without collecting additional images and camera information for the target domain.** Unlike 3D extensions of prior text-guided domain adaptation methods, our novel pipeline was able to fine-tune the state-of-the-art 3D generator of the source domain to synthesize high resolution, multi-view consistent images in text-guided targeted domains without additional data, outperforming the existing text-guided domain adaptation methods in diversity and text-image correspondence. Furthermore, we propose and demonstrate diverse 3D image manipulations such as one-shot instance-selected adaptation and single-view manipulated 3D reconstruction to fully enjoy diversity in text.


## Recent Updates
- `2023.03.31`: Code & Colab demo are released.
- `2023.04.03`: Gradio demo is released. 


## Requirements

* We have used Linux (Ubuntu 20.04).
* We have used 1 NVIDIA A100 GPU for text-guided domain adaptation, and have used 1 NVIDIA A100 or RTX3090 GPU for the test using the shifted generators.     
1&ndash;8 high-end NVIDIA GPUs. We have done all testing and development using V100, RTX3090, and A100 GPUs.
* Python 3.8, PyTorch 1.12.1 (or later), CUDA toolkit 11.6 (or later).
* Python libraries: see [environment.yml](../environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
    ```.bash
    git clone https://github.com/gwang-kim/DATID-3D_tmp.git
    cd DATID-3D
    conda env create -n datid3d -f environment.yml
    conda activate datid3d
    ```
* We use the pretrained [EG3D](https://github.com/NVlabs/eg3d) models as our pretrained 3D generative models. The prtrained EG3D models will be downloaded automatically for convinence. Or you can download the pretrained [EG3D models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d/files), put `afhqcats512-128.pkl` and `affhqrebalanced512-128.pkl` in `~/eg3d/pretrained/`.

## Demo
### Gradio Demo
- We provide a interactive Gradio app demo.
```.bash
python gradio_app.py
```
<p align="center">
  <img src="assets/datid3d_gradio.gif" />
</p> 

### Colab Demo  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e9NSVB7x_hjz-nr4K0jO4rfTXILnNGtA?usp=sharing)
- We provide a Colab demo for you to play with DATID-3D! Due to 12GB of the VRAM limit in Colab, we only provide the codes of inference & applications with 3D generative models fine-tuned using DATID-3D, not fine-tuning code. 


## Download Fine-tuned 3D Generative Models

Fine-tuned 3D generative models using DATID-3D pipeline are stored as `*.pkl` files.
You can download the models in [our Hugginface model pages](https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/tree/main/finetuned_models).
```.bash
mkdir finetuned
wget https://huggingface.co/gwang-kim/datid3d-finetuned-eg3d-models/resolve/main/finetuned_models/ffhq-pixar.pkl -O finetuned
```


## Sample Images, Shapes and Videos
You can sample images and shapes (as .mrc files), pose-controlled videos using the shifted 3D generative model.
For example:
```.bash
# Sample images and shapes (as .mrc files) using the shifted 3D generative model

python datid3d_test.py --mode image \
--generator_type='ffhq' \
--outdir='test_runs' \
--seeds='100-200' \
--trunc='0.7' \
--shape=True \
--network=finetuned/ffhq-pixar.pkl 
```

```.bash
# Sample pose-controlled videos using the shifted 3D generative model

python datid3d_test.py --mode video \
--generator_type='ffhq' \
--outdir='test_runs' \
--seeds='100-200' \
--trunc='0.7' \
--grid=4x4 \
--network=finetuned/ffhq-pixar.pkl 
```
The results are saved to `~/test_runs/image` or `~/test_runs/video`. 

Following [EG3D](https://github.com/NVlabs/eg3d), we visualize our .mrc shape files with [UCSF Chimerax](https://www.cgl.ucsf.edu/chimerax/).

To visualize a shape in ChimeraX do the following:
1. Import the `.mrc` file with `File > Open`
1. Find the selected shape in the Volume Viewer tool
    1. The Volume Viewer tool is located under `Tools > Volume Data > Volume Viewer`
1. Change volume type to "Surface"
1. Change step size to 1
1. Change level set to 10
    1. Note that the optimal level can vary by each object, but is usually between 2 and 20. Individual adjustment may make certain shapes slightly sharper
1. In the `Lighting` menu in the top bar, change lighting to "Full"


## Single-shot Text-guided 2D-to-3D
### Text-guided Manipulated 3D Reconstruction
This includes `alignment -> pose extraction -> 3D GAN inversion -> generation of images using fine-tuned generator`.
We use [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) as the pose estimation models.
The prtrained pose estimation will be downloaded automatically for convinence.
Or you can download the pretrained [pose estimation model](https://drive.google.com/file/d/1zawY7jYDJlUGnSAXn1pgIHgIvJpiSmj5/view?usp=sharing) and [BFM files](https://drive.google.com/file/d/1mdqkEUepHZROeOj99pXogAPJPqzBDN2G/view?usp=sharing), put `epoch_20.pth` in `~/pose_estimation/checkpoints/pretrained/` and put unzip `BFM.zip` in `~/pose_estimation/`.
For example:
```.bash
# Text-guided manipulated 3D reconstruction from images using the shifted 3D generative model

python datid3d_test.py --mode manip \
--indir='input_imgs' \
--generator_type='ffhq' \
--outdir='test_runs' \
--trunc='0.7' \
--network=finetuned/ffhq-pixar.pkl 
```
The results are saved to `~/test_runs/manip_3D_recon/4_manip_result`.



## Text-guided Domain Adaptation of 3D Generator
You can do text-guided domain adaptation of 3D generator with your own text prompt using `datid3d_train.py`. For example:

```.bash
python datid3d_train.py \
   --mode='ft' \
   --pdg_prompt='a FHD photo of face of beautiful Elf with silver hair in the live action movie' \
   --pdg_generator_type='ffhq' \
   --pdg_strength=0.7 \
   --pdg_num_images=1000 \
   --pdg_sd_model_id='stabilityai/stable-diffusion-2-1-base' \
   --pdg_num_inference_steps=50 \
   --ft_generator_type='same' \
   --ft_batch=20 \
   --ft_kimg=200
```

The results of each training run are saved to a newly created directory, for example `~/training_runs/00011-ffhq-data_ffhq_a_FHD_photo_of_face_of_beautiful_Elf_with_silver_hair_in_the_live_action_movie-gpus1-batch20-gamma5`. 


## Citation

```
@inproceedings{kim2022datid3d,
  author = {Gwanghyun Kim and Se Young Chun},
  title = {DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model},
  booktitle = {CVPR},
  year = {2023}
}
```

## Acknowledgements

We thank the contributions of public projects for sharing their code. We apply our pipelines to [EG3D](https://github.com/NVlabs/eg3d), one of the 3D generative models, and adopt [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) as our text-to-image diffusion models and [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) as our pose estimation models. We also utilze a part of codes in [HFGI3D](https://github.com/jiaxinxie97/HFGI3D).
