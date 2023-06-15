from styleGan import styleGan
from PyQt5.QtWidgets import QMessageBox
import os
import random


class Controller:
    def __init__(self, view, generator: styleGan):
        self.view = view
        self.view.set_controller(self)
        self.generator = generator
        self.set_default_paths()

    def set_model_path(self, path):
        self.view.set_model_dir(path)
        self.generator.set_model_dir(path)

    def set_output_path(self, path):
        self.view.set_output_dir(path)
        self.generator.set_output_dir(path)

    def set_default_paths(self):
        current_path = os.getcwd()
        path = os.path.join(current_path, 'network-snapshot-012000.pkl')
        output_path = os.path.join(current_path, 'output')
        self.set_model_path(path)
        self.set_output_path(output_path)

    def generate_image(self, seed, truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx):
        self.view.set_seeds(str(seed[0]))
        print(seed, truncation_psi, noise_mode,
              translate_x, translate_y, rotate, class_idx)
        # try:
        img_path = self.generator.generate_images(seed, truncation_psi,
                                                  noise_mode, (translate_x, translate_y), rotate, class_idx)
        # except Exception as e:
        #     error_message = f"Error occurred during image generation:\n{str(e)}"
        #     QMessageBox.critical(self.view, "Error", error_message)
        self.view.set_image(img_path)
