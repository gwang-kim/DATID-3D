# DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model (CVPR 2023) 

[![arXiv](https://img.shields.io/badge/arXiv-2211.16374-red)](https://arxiv.org/abs/2211.16374) [![project_page](https://img.shields.io/badge/-project%20page-blue)](https://datid-3d.github.io/)

[//]: # ()
[//]: # ([![arXiv]&#40;https://img.shields.io/badge/paper-cvpr2022-cyan&#41;]&#40;https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html&#41; [![arXiv]&#40;https://img.shields.io/badge/arXiv-2110.02711-red&#41;]&#40;https://arxiv.org/abs/2110.02711&#41;)

[//]: # ([![video]&#40;https://img.shields.io/badge/video-green&#41;]&#40;https://youtu.be/YVCtaXw6fw8&#41; [![poster]&#40;https://img.shields.io/badge/poster-orange&#41;]&#40;https://drive.google.com/file/d/1QgRFIRba492dCZ6v7BcZB9zqyp91aTjL/view?usp=sharing&#41; )

<p align="center">

  <img src="assets/datid_3d_result.gif" />


</p> 

[comment]: <> (![]&#40;imgs/main1.png&#41;)

[comment]: <> (![]&#40;imgs/main2.png&#41;)

> **DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model**<br>
> [Gwanghyun Kim](https://gwang-kim.github.io/), [Se Young Chun](https://icl.snu.ac.kr/pi) <br>
> CVPR 2023
> 
>**Abstract**: <br>
Recent 3D generative models have achieved remarkable performance in synthesizing high resolution photorealistic images with view consistency and detailed 3D shapes, but training them for diverse domains is challenging since it requires massive training images and their camera distribution information.  
Text-guided domain adaptation methods have shown impressive performance on converting the 2D generative model on one domain into the models on other domains with different styles by leveraging the CLIP (Contrastive Language-Image Pre-training), rather than collecting massive datasets for those domains. However, one drawback of them is that the sample diversity in the original generative model is not well-preserved in the domain-adapted generative models due to the deterministic nature of the CLIP text encoder. Text-guided domain adaptation will be even more challenging for 3D generative models not only because of catastrophic diversity loss, but also because of inferior text-image correspondence and poor image quality. 
**Here we propose DATID-3D, a novel pipeline of text-guided domain adaptation tailored for 3D generative models using text-to-image diffusion models that can synthesize diverse images per text prompt without collecting additional images and camera information for the target domain.** Unlike 3D extensions of prior text-guided domain adaptation methods, our novel pipeline was able to fine-tune the state-of-the-art 3D generator of the source domain to synthesize high resolution, multi-view consistent images in text-guided targeted domains without additional data, outperforming the existing text-guided domain adaptation methods in diversity and text-image correspondence. Furthermore, we propose and demonstrate diverse 3D image manipulations such as one-shot instance-selected adaptation and single-view manipulated 3D reconstruction to fully enjoy diversity in text.

