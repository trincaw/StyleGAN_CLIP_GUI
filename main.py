import sys
from PyQt5.QtWidgets import QApplication
from view import view
from controller import controller
from styleGan import styleGan
import os

# C:\Users\nico\AppData\Local\torch_extensions\torch_extensions\Cache\py38_cu118\


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    import sys
    sys.excepthook = except_hook
    app = QApplication(sys.argv)

    v = view()
    s = styleGan()
    c = controller(v, s)

    v.show()
    sys.exit(app.exec_())
