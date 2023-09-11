# StyleGAN_CLIP_GUI

## Description
StyleGAN_CLIP_GUI is a user-friendly software application tailored for artists, designers, and researchers. It harnesses the combined capabilities of StyleGAN2-ADA and CLIP to enable image creation, manipulation, and exploration. The graphical user interface (GUI) primarily integrates with StyleGAN for seamless image synthesis. Additionally, it incorporates a unique algorithm for latent vector editing with CLIP, significantly expanding the creative possibilities for StyleGAN-generated content.

## Build
| Item                           | Description                                      |
|--------------------------------|--------------------------------------------------|
| **Main programming language**  | Python 3.10                                      |
| **GPU computing platforms**    | Cuda 11.3                                       |
| **GPU-accelerated library**    | cuDNN 7.3.1                                    |
| **Python libraries**           |                                                |
| - GUI framework                | PyQT5 5.15.9                                   |
| - Deep learning framework      | Pytorch 1.13                                   |
| - Fundamental package for scientific computing | Numpy >=1.20                |
| **Google Workspace**           |                                                |
| - GPU                          | Nvidia Tesla T4 x1                             |
| - Machine                      | n1-standard-2 (2 vCPUs, 7.5 GB memory)         |
| - Disk                         | 100 GB                                         |

## Installation
Go to the project path and follow the instructions:
 - Install CUDA 11.8
 - ``` pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ```
 - ``` git clone https://github.com/NVlabs/stylegan3 ```
 - ``` git clone https://github.com/openai/CLIP ```
 - ``` pip install -e ./CLIP ```
 - ``` pip install einops ninja ```
 - ``` pip install -r requirements.txt ```
 - for other dependency check https://github.com/ouhenio/StyleGAN3-CLIP-notebooks

## Execution
```console
python main.py
```
## References
1. [StyleGAN3CLIP](https://github.com/ouhenio/StyleGAN3-CLIP-notebooks), from Ouhenio
2. [stylegan3](https://github.com/NVlabs/stylegan3/tree/main), from NVlabs
3. [CLIP](https://github.com/openai/CLIP), from OpenAI 


