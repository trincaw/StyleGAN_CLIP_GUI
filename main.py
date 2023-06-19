import sys
from PyQt5.QtWidgets import QApplication
from view import view
from controller import controller
from styleGan import styleGan
import os

# C:\Users\nico\AppData\Local\torch_extensions\torch_extensions\Cache\py38_cu118\

if __name__ == '__main__':
    app = QApplication(sys.argv)

    v = view()
    s = styleGan()
    c = controller(v, s)

    v.show()
    sys.exit(app.exec_())
