from controller import Controller
import sys
from PyQt5.QtWidgets import QApplication
from view import View
from controller import Controller
from styleGan import styleGan
import os

# C:\Users\nico\AppData\Local\torch_extensions\torch_extensions\Cache\py38_cu118\

if __name__ == '__main__':
    app = QApplication(sys.argv)

    view = View()
    styleGan = styleGan()
    controller = Controller(view, styleGan)

    view.show()
    sys.exit(app.exec_())
