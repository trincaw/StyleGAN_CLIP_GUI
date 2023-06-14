from controller import Controller
import sys
from PyQt5.QtWidgets import QApplication
from view import View
from controller import Controller
from styleGan import styleGan
import os


if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = View()
    view.show()

    styleGan = styleGan()

    controller = Controller(view, styleGan)
    sys.exit(app.exec_())
