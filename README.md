# **High Quality Text to Image Generation using Stable Diffusion, GFPGAN,Real-ESR and Swin IR**


[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HemantKArya/HqStableDiffusionColab/blob/main/HighQuality_Text2Image_Stable_Diffusion_ls.ipynb)
[![banner](./doc/bannerls.jpg)](https://www.instagram.com/iamhemantindia)

Generate 4K and FULL HD Images and Artworks for Free Using Stable Diffusion.

## No Need to generate token key for genrating images from huggingface...
Link to Coalb Notebook [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HemantKArya/HqStableDiffusionColab/blob/main/HighQuality_Text2Image_Stable_Diffusion_ls.ipynb)


For Upscale Only goto RealESR Notebook (4K Upscale)[![open in colabesr](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HemantKArya/HqStableDiffusionColab/blob/main/RealESR_Upscale.ipynb)
 
Run All the cell until you Reach your Prompt cell.
 ***In case if you have any human face in your images then it will restore Distorted figures(like eyes,nose,etc) in images, here is example.***  **In Stable Diffusin her Eyes and Lips are bit distorted.**
 ![indexface](./doc/indexface.jpg)

To upscale images to 2K or 4k using Real-ESR GAN. Note that after running Reasl-ESRGAN leave SwinIR until unless you are not satisfied with RealESR results.
![scupesr](./doc/sc4.png)
after running this cell you will get a comparison matrix like this.

**Input Images --> Upscaled Images(Real-ESR)**
![index3](./doc/index33.png)

After Upscaling you images using Real-ESRGAN rest of the cell are optional to run and not recommended (Cause limited GPU RAM in Colab, After running these cell may be it will show you error like ``cuda out of memory``) to run until you are not satisfied with result of Upscaled images of Real-ESR.
right Now I am going to show you difference b/w both Upscalers.
Using both Optional cell at the last of notebook. (It may full your current colab RAM)

**Input Images ------ Upscaled Images(SwinIR) ----- Upscaled Images(RealESRGAN)**
![index5](./doc/index35.png)

Visit Logical Spot for Video Help:-

 [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/c/LogicalSpot)
 
 
 # **Stable Diffusion** 🎨 
*...using `🧨diffusers`*

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.
This Colab notebook shows how to use Stable Diffusion with the 🤗 Hugging Face [🧨 Diffusers library](https://github.com/huggingface/diffusers) . 
https://github.com/CompVis/stable-diffusion

orignal-link to colab https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb

# **Real-ESRGAN**
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2107.10833)
[![GitHub Stars](https://img.shields.io/github/stars/xinntao/Real-ESRGAN?style=social)](https://github.com/xinntao/Real-ESRGAN)
[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![open in colabesr](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HemantKArya/HqStableDiffusionColab/blob/main/RealESR_Upscale.ipynb)
 

Real-ESRGAN aims at developing Practical Algorithms for General Image/Video Restoration.We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.

# **SwinIR**
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2108.10257)
[![GitHub Stars](https://img.shields.io/github/stars/JingyunLiang/SwinIR?style=social)](https://github.com/JingyunLiang/SwinIR)
[![download](https://img.shields.io/github/downloads/JingyunLiang/SwinIR/total.svg)](https://github.com/JingyunLiang/SwinIR/releases)

SwinIR achieves state-of-the-art performance on six tasks: image super-resolution (including classical, lightweight and real-world image super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact reduction. See our [paper](https://arxiv.org/abs/2108.10257) and [project page](https://github.com/JingyunLiang/SwinIR) for detailed results.

### (No colorization; No CUDA extensions required)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2101.04061)
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/GFPGAN?style=social)](https://github.com/TencentARC/GFPGAN)
[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)

## **GFPGAN** - Towards Real-World Blind Face Restoration with Generative Facial Prior

GFPGAN is a blind face restoration algorithm towards real-world face images. <br>
It leverages the generative face prior in a pre-trained GAN (*e.g.*, StyleGAN2) to restore realistic faces while precerving fidelity. <br>



