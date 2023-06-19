# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch


# ----------------------------------------------------------------------------

def make_transform(translate: Tuple[float, float], angle: float):
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

# ----------------------------------------------------------------------------


class styleGan:

    def load_network(self, device, network_path):
        self.device = device
        self.network_pkl = network_path
        with open(network_path, 'rb') as fp:
            G = pickle.load(fp)['G_ema'].to(self.device)

            zs = torch.randn([10000, G.mapping.z_dim], device=self.device)
            w_stds = G.mapping(zs, None).std(0)
            self.w_stds = w_stds
            self.G = G
            return G, w_stds

    def set_output_dir(self, output_path):
        self.output_path = output_path

    def generate_images(self, seeds: List[int], truncation_psi: float, noise_mode: str, translate: Tuple[float, float], rotate: float, class_idx: Optional[int], output_path="output"):
        self.set_output_dir(output_path)
        os.makedirs(self.output_path, exist_ok=True)

        # Labels.
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        if self.G.c_dim != 0:
            if class_idx is None:
                raise ValueError(
                    'Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print(
                    'warn: --class=lbl ignored when running on an unconditional network')
        print(seeds, truncation_psi,
              noise_mode, translate, rotate, class_idx)
        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' %
                  (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(
                seed).randn(1, self.G.z_dim)).to(self.device)

            # Construct an inverse rotation/translation matrix and pass it to the generator.
            # The generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(self.G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = self.G(z, label, truncation_psi=truncation_psi,
                         noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 +
                   128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
                f'{self.output_path}/seed{seed:04d}.png')
        s = seeds[0]
        img_path = f'{self.output_path}/seed{s:04d}.png'
        return img_path

# ----------------------------------------------------------------------------


# if __name__ == "__main__":
#     # Example usage:
#     network_pkl = "C:\\Users\\nico\\Doc\\StyleGAN_CLIP_GUI\\network-snapshot-012000.pkl"
#     seeds = [0, 1]
#     truncation_psi = 1
#     noise_mode = "const"
#     output_path = "output_images"
#     translate = (0.0, 0.0)
#     rotate = 0
#     class_idx = None

#     generator = styleGan()
#     generator.set_model_dir(network_pkl)
#     generator.set_output_dir(output_path)
#     generator.generate_images(seeds, truncation_psi,
#                               noise_mode, translate, rotate, class_idx)
