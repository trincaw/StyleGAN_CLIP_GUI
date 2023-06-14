from styleGan import styleGan
from PyQt5.QtWidgets import QMessageBox
import os


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
        # self.view.set_output_dir(path)
        self.generator.set_output_dir(path)

    def set_default_paths(self):
        current_path = os.getcwd()
        path = os.path.join(current_path, 'network-snapshot-012000.pkl')
        output_path = os.path.join(current_path, 'output')
        self.set_model_path(path)
        self.set_output_path(output_path)

    def generate_image(self):

        seeds = self.validate_int(self.view.seeds_edit.text(), default=1)
        truncation_psi = self.validate_float(
            self.view.trunc_edit.text(), default=0.7)
        noise_mode = 'const' if self.view.noise_edit.text(
        ) == '' else self.view.noise_edit.text()
        translate_x = self.validate_int(
            self.view.translate_x_edit.text(), default=0.0)
        translate_y = self.validate_int(
            self.view.translate_y_edit.text(), default=0.0)
        rotate = self.validate_int(self.view.rotate_edit.text(), default=0)
        class_idx = self.validate_int(
            self.view.class_edit.text(), default=None)

        print(seeds, truncation_psi, noise_mode,
              translate_x, translate_y, rotate, class_idx)
        # try:
        self.generator.generate_images(seeds, truncation_psi,
                                       noise_mode, (translate_x, translate_y), rotate, class_idx)
        # except Exception as e:
        #     error_message = f"Error occurred during image generation:\n{str(e)}"
        #     QMessageBox.critical(self.view, "Error", error_message)

    def validate_int(self, value, default=None):
        try:
            return int(value) if value else default
        except ValueError:
            return default

    def validate_float(self, value, default=None):
        try:
            return float(value) if value else default
        except ValueError:
            return default

    def validate_seeds(self, value):
        if value:
            seeds_list = value.split(',')
            try:
                seeds = [int(seed) for seed in seeds_list]
                return seeds
            except ValueError:
                pass
        return None

    def validate_translate(self, value):
        if value:
            try:
                translate_x, translate_y = value.split(',')
                return float(translate_x), float(translate_y)
            except ValueError:
                pass
        return 0, 0
