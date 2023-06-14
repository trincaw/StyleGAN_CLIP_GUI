import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QLineEdit
from controller import Controller


class View(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('StyleGAN Model Viewer')
        self.setGeometry(100, 100, 400, 300)

        self.network_path = QLineEdit(self)
        self.network_path.setGeometry(20, 20, 260, 30)
        self.network_path.setPlaceholderText('Network PKL File')

        # self.output_path = QLineEdit(self)
        # self.output_path.setGeometry(20, 20, 260, 30)
        # self.output_path.setPlaceholderText('Output path')
        self.output_path = 'output'
        self.browse_btn = QPushButton('Browse', self)
        self.browse_btn.setGeometry(290, 20, 80, 30)
        self.browse_btn.clicked.connect(self.browse_model)

        self.seeds_edit = QLineEdit(self)
        self.seeds_edit.setGeometry(20, 70, 150, 30)
        self.seeds_edit.setPlaceholderText('Seeds')

        self.trunc_edit = QLineEdit(self)
        self.trunc_edit.setGeometry(20, 120, 150, 30)
        self.trunc_edit.setPlaceholderText('Truncation Psi')

        self.noise_edit = QLineEdit(self)
        self.noise_edit.setGeometry(20, 170, 150, 30)
        self.noise_edit.setPlaceholderText('Noise Mode')

        self.translate_x_edit = QLineEdit(self)
        self.translate_x_edit.setGeometry(220, 70, 150, 30)
        self.translate_x_edit.setPlaceholderText('Translate X')

        self.translate_y_edit = QLineEdit(self)
        self.translate_y_edit.setGeometry(220, 120, 150, 30)
        self.translate_y_edit.setPlaceholderText('Translate Y')

        self.rotate_edit = QLineEdit(self)
        self.rotate_edit.setGeometry(220, 170, 150, 30)
        self.rotate_edit.setPlaceholderText('Rotate')

        self.class_edit = QLineEdit(self)
        self.class_edit.setGeometry(20, 220, 150, 30)
        self.class_edit.setPlaceholderText('Class Index')

        self.generate_btn = QPushButton('Generate', self)
        self.generate_btn.setGeometry(220, 220, 150, 30)
        self.generate_btn.clicked.connect(self.generate_image)

    def set_controller(self, controller: Controller):
        self.controller = controller

    def set_output_dir(self, path: str):
        self.output_path.setText(path)

    def set_model_dir(self, path: str):
        self.network_path.setText(path)

    def browse_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Select Model PKL File', '', 'PKL Files (*.pkl)')
        self.network_path.setText(file_path)
        self.controller.set_model_path(file_path)

    def generate_image(self):
        self.controller.generate_image()
