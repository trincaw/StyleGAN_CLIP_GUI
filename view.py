import sys
from PyQt5.QtWidgets import QDoubleSpinBox, QMainWindow, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox
from controller import controller
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import random


class view(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('StyleGAN GUI')
        self.setGeometry(100, 100, 820, 430)

        self.network_path = QLineEdit(self)
        self.network_path.setGeometry(20, 20, 260, 30)
        self.network_path.setPlaceholderText('Network PKL File')

        self.browse_btn = QPushButton('Browse', self)
        self.browse_btn.setGeometry(290, 20, 80, 30)
        self.browse_btn.clicked.connect(self.browse_model)

        self.output_path = QLineEdit(self)
        self.output_path.setGeometry(20, 70, 260, 30)
        self.output_path.setPlaceholderText('Output Path')

        self.output_btn = QPushButton('Select', self)
        self.output_btn.setGeometry(290, 70, 80, 30)
        self.output_btn.clicked.connect(self.browse_output_path)

        self.seeds_edit = QLineEdit(self)
        self.seeds_edit.setGeometry(20, 120, 150, 30)
        self.seeds_edit.setPlaceholderText('Seeds')

        self.trunc_edit = QDoubleSpinBox(self)
        self.trunc_edit.setGeometry(20, 170, 150, 30)
        self.trunc_edit.setRange(0, 10)  # Set the range of the spin box
        self.trunc_edit.setSingleStep(0.1)  # Set the step size
        self.trunc_edit.setDecimals(2)  # Set the number of decimals
        self.trunc_edit.setSuffix(' Psi')  # Add a suffix
        self.trunc_edit.setValue(0.7)  # Set the initial value

        self.noise_edit = QLineEdit(self)
        self.noise_edit.setGeometry(20, 220, 150, 30)
        self.noise_edit.setPlaceholderText('Noise Mode')

        self.translate_x_edit = QLineEdit(self)
        self.translate_x_edit.setGeometry(220, 120, 150, 30)
        self.translate_x_edit.setPlaceholderText('Translate X')

        self.translate_y_edit = QLineEdit(self)
        self.translate_y_edit.setGeometry(220, 170, 150, 30)
        self.translate_y_edit.setPlaceholderText('Translate Y')

        self.rotate_edit = QLineEdit(self)
        self.rotate_edit.setGeometry(220, 220, 150, 30)
        self.rotate_edit.setPlaceholderText('Rotate')

        self.class_edit = QLineEdit(self)
        self.class_edit.setGeometry(20, 270, 150, 30)
        self.class_edit.setPlaceholderText('Class Index')

        self.generate_btn = QPushButton('Generate', self)
        self.generate_btn.setGeometry(220, 270, 150, 30)
        self.generate_btn.clicked.connect(self.generate_image_from_seed)

        self.image_viewer = QLabel(self)
        self.image_viewer.setGeometry(390, 20, 360, 360)
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setScaledContents(True)

        self.generate_random_btn = QPushButton('Generate Random', self)
        self.generate_random_btn.setGeometry(20, 320, 350, 30)
        self.generate_random_btn.clicked.connect(self.generate_random_image)

        self.edit_text_btn = QPushButton('Edit', self)
        self.edit_text_btn.setGeometry(220, 370, 150, 30)
        self.edit_text_btn.clicked.connect(self.generate_with_text)

        self.text_edit = QLineEdit(self)
        self.text_edit.setGeometry(20, 370, 150, 30)
        self.text_edit.setPlaceholderText('Text')

    def set_controller(self, controller: controller):
        self.controller = controller

    def set_output_dir(self, path: str):
        self.output_path.setText(path)

    def set_model_dir(self, path: str):
        self.network_path.setText(path)

    def set_image(self, image_path: str):
        pixmap = QPixmap(image_path).scaled(
            360, 360, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_viewer.setPixmap(pixmap)

    def set_seeds(self, seeds: str):
        self.seeds_edit.setText(seeds)

    def generate_with_text(self):

        self.disable_buttons()
        text = self.text_edit.text()

        r = random.randint(1, 99999)

        s = self.validate_int(self.seeds_edit.text(), default=r)
        print(s)
        if text:
            self.controller.generate_image_from_text(text)
        self.enable_buttons()

    def browse_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Select Model PKL File', '', 'PKL Files (*.pkl)')
        self.network_path.setText(file_path)
        self.controller.load_network(file_path)

    def browse_output_path(self):
        file_dialog = QFileDialog()
        output_path = file_dialog.getExistingDirectory(
            self, 'Select Output Directory')
        self.controller.set_output_path(output_path)

    def generate_random_image(self):
        truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx = self.validate_generation()

        self.disable_buttons()
        r = random.randint(1, 99999)

        self.controller.generate_image(
            [r], truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx)
        self.enable_buttons()

    def generate_image_from_seed(self):

        truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx = self.validate_generation()

        self.disable_buttons()
        r = random.randint(1, 99999)

        s = self.validate_int(self.seeds_edit.text(), default=r)
        print(s)

        self.controller.generate_image(
            [s], truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx)

        self.enable_buttons()

    def disable_buttons(self):
        self.generate_btn.setEnabled(False)
        self.generate_random_btn.setEnabled(False)
        self.edit_text_btn.setEnabled(False)

    def enable_buttons(self):
        self.generate_btn.setEnabled(True)
        self.generate_random_btn.setEnabled(True)
        self.edit_text_btn.setEnabled(True)

    def validate_generation(self):

        truncation_psi = self.validate_float(
            self.trunc_edit.value(), default=0.7)

        noise_mode = 'const' if self.noise_edit.text(
        ) == '' else self.noise_edit.text()

        translate_x = self.validate_int(
            self.translate_x_edit.text(), default=0, error="Invalid Translate X value")

        translate_y = self.validate_int(
            self.translate_y_edit.text(), default=0, error="Invalid Translate Y value")

        rotate = self.validate_int(
            self.rotate_edit.text(), default=0, error="Invalid Rotate value")

        class_idx = self.validate_int(
            self.class_edit.text(), default=None, error="Invalid Class Index value")

        return truncation_psi, noise_mode, translate_x, translate_y, rotate, class_idx

    def validate_int(self, value, default=None, error="Generic Error"):
        try:
            return int(value) if value else default
        except ValueError:
            QMessageBox.critical(self, 'Error', error)
            return default

    def validate_float(self, value, default=None, error="Generic Error"):
        try:
            return float(value) if value else default
        except ValueError:
            QMessageBox.critical(self, 'Error', error)
            return default

    def validate_seeds(self, value, default, error="Generic Error"):
        if value:
            seeds_list = value.split(',')
            try:
                seeds = [int(seed) for seed in seeds_list]
                return value
            except ValueError:
                QMessageBox.critical(self, 'Error', error)
        return default
