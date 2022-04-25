from sys import path
path.append(__file__.replace('\\main.py', ''))
path.append(__file__.replace('\\omt_gui\\main.py', ''))

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from gui import create_gui


def launch_app() -> None:
    """
    Build and start app
    :return: None
    """
    app = QApplication([])
    #icon_path = __file__.replace('main.py', 'gui\\icons\\256.png')
    #app.setWindowIcon(QIcon(icon_path))

    gui = create_gui()
    gui.show()
    app.exec()


if __name__ == '__main__':  # pragma: no cover
    launch_app()

