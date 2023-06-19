import torch
from myclip import myclip
from styleClip import styleClip
from styleGan import styleGan
import os


class controller:
    def __init__(self, view, generator: styleGan):
        self.view = view
        self.view.set_controller(self)
        self.generator = generator
        self.set_default_paths()

    def load_network(self, path):
        print("init styleGan")
        self.device = torch.device('cuda:0')
        self.generator.load_network(self.device, path)

    def init_styleClip(self):

        print("init Clip")
        clip_model = myclip(self.device)
        print("init styleClip")
        sc = styleClip(self.device, clip_model, self.generator)
        print("run styleClip")
        sc.run(texts="a brown jacket", steps=50,
               seed=14, render_video=False, save_every=2)

    def set_output_path(self, path):
        self.view.set_output_dir(path)
        self.generator.set_output_dir(path)

    def set_default_paths(self):
        current_path = os.getcwd()
        path = os.path.join(current_path, 'network-snapshot-012000.pkl')
        output_path = os.path.join(current_path, 'output')
        self.set_output_path(output_path)

    def generate_image(self, seed, truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx):
        # self.view.set_seeds(str(seed[0]))
        # print(seed, truncation_psi, noise_mode,
        #       translate_x, translate_y, rotate, class_idx)
        # # try:
        # img_path = self.generator.generate_images(seed, truncation_psi,
        #                                           noise_mode, (translate_x, translate_y), rotate, class_idx)
        # # except Exception as e:
        # #     error_message = f"Error occurred during image generation:\n{str(e)}"
        # #     QMessageBox.critical(self.view, "Error", error_message)
        # self.view.set_image(img_path)
        self.init_styleClip()
