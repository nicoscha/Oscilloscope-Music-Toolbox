from datetime import date as dt_date
from functools import partial
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QComboBox, QCheckBox, QDateEdit, QDoubleSpinBox, QFileDialog, QHBoxLayout, QPushButton, QRadioButton, QSpinBox, QSizePolicy, QSpacerItem, QStackedWidget, QTabWidget, QVBoxLayout, QWidget
from typing import Callable, List, Union
from collections import namedtuple

from uuid import uuid4
from omt_utils import gen_sin, gen_cos, gen_sawtooth, gen_triangle, gen_rectangle, write, scale

SAMPLE_RATE = 48000
SAMPLES = 48000*1
parameter = namedtuple('parameter', 'amplitude function frequency side')
parameters = {}


class Selector(QHBoxLayout):
    def build(self, uu: str, side: str) -> None:
        self.uu = uu
        self.side = side
        self.amplitude_spin_box = QDoubleSpinBox()
        self.amplitude_spin_box.setMaximum(10)
        self.amplitude_spin_box.setSingleStep(0.1)
        self.amplitude_spin_box.setValue(1.0)
        self.amplitude_spin_box.valueChanged.connect(self.update_parameters)

        self.combo_box = QComboBox()
        self.combo_box.addItems(['sin', 'cos', 'saw', 'tri', 'rec'])
        self.combo_box.currentTextChanged.connect(self.update_parameters)

        self.frequency_spin_box = QDoubleSpinBox()
        self.frequency_spin_box.setRange(0, SAMPLE_RATE/2)
        self.frequency_spin_box.setSingleStep(0.02)
        self.frequency_spin_box.setValue(400)
        self.frequency_spin_box.valueChanged.connect(self.update_parameters)

        self.update_parameters()

        self.addWidget(self.amplitude_spin_box)
        self.addWidget(self.combo_box)
        self.addWidget(self.frequency_spin_box)

    def update_parameters(self) -> None:
        amplitude = self.amplitude_spin_box.value()
        function = self.combo_box.currentText()
        frequency = self.frequency_spin_box.value()
        p = parameter(amplitude, function, frequency, self.side)
        parameters[self.uu] = p
        print(parameters[self.uu])


gen_sig = {'sin': gen_sin, 'cos': gen_cos, 'saw': gen_sawtooth,
           'tri': gen_triangle, 'rec': gen_rectangle}


def calc():
    x_samples = []
    y_samples = []
    for param in parameters.values():
        f = param.frequency
        signal = gen_sig[param.function](f, SAMPLE_RATE, SAMPLES)
        if param.amplitude != 1.0:
            samples = scale(x_samples, param.amplitude)
        else:
            samples = signal
        if param.side == 'x':
            x_samples = samples
        elif param.side == 'y':
            y_samples = samples
    write(x_samples, y_samples, SAMPLE_RATE=SAMPLE_RATE)
    print('calc done')


class XYLayout(QVBoxLayout):
    def build(self, side) -> None:
        self.side = side

        add_button = QPushButton()
        add_button.setText('Add signal')
        add_button.clicked.connect(self.add_selector)

        self.addWidget(add_button)
        self.add_selector()

    def add_selector(self) -> None:
        selector = Selector()
        uu = uuid4()
        selector.build(uu, side=self.side)
        self.addLayout(selector)


def build_gui() -> QWidget:
    """
    Build the gui and connect all signals and slots
    :return: QWidget containing the gui
    """

    # Start button
    start_button = QPushButton()
    start_button.setText('Start')
    start_button.clicked.connect(calc)

    # Add layouts to main widget
    main_h_box = QHBoxLayout()
    x_layout = XYLayout()
    x_layout.build('x')
    y_layout = XYLayout()
    y_layout.build('y')
    main_h_box.addLayout(x_layout)
    main_h_box.addLayout(y_layout)
    main_h_box.addWidget(start_button)

    main_widget = QWidget()
    main_widget.setLayout(main_h_box)

    return main_widget
