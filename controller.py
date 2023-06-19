import sys
import torch
from myclip import myclip
from styleClip import styleClip
from styleGan import styleGan
import os


class controller:
    def __init__(self, view, s_gan: styleGan):
        self.view = view
        self.view.set_controller(self)
        self.s_gan = s_gan
        # self.set_default_paths()

    def load_network(self, path):
        device = torch.device('cuda:0')
        print("init styleGan")
        self.s_gan.load_network(device, path)
        print("init Clip")
        clip_model = myclip(device)
        print("init styleClip")
        self.s_clip = styleClip(device, self, clip_model, self.s_gan)

    def set_output_path(self, path):
        self.view.set_output_dir(path)
        self.s_clip.set_output_dir(path)
        self.s_gan.set_output_dir(path)

    def set_default_paths(self):
        current_path = os.getcwd()
        # path = os.path.join(current_path, 'network-snapshot-012000.pkl')
        output_path = os.path.join(current_path, 'output')
        self.set_output_path(output_path)

    def generate_image(self, seed, truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx):
        self.view.set_seeds(str(seed[0]))
        print(seed, truncation_psi, noise_mode,
              translate_x, translate_y, rotate, class_idx)
        # try:
        img_path = self.s_gan.generate_images(seed, truncation_psi,
                                              noise_mode, (translate_x, translate_y), rotate, class_idx)
        # except Exception as e:
        #     error_message = f"Error occurred during image generation:\n{str(e)}"
        #     QMessageBox.critical(self.view, "Error", error_message)
        self.view.set_image(img_path)

    def generate_image_from_text(self, text="a brown jacket", seed=14):
        print(text, seed)
        self.s_clip.r()
        # self.s_clip.run(texts=text, steps=50,
        #                 seed=seed, render_video=False, save_every=2)

    def update_image(self, img_path):
        self.view.set_image(img_path)
