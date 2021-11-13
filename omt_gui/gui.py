from datetime import date as dt_date
from functools import partial
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QComboBox, QCheckBox, QDateEdit, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSpinBox, QSizePolicy, QSpacerItem, QStackedWidget, QTabWidget, QVBoxLayout, QWidget
from typing import Callable, List, Union
from collections import namedtuple

from uuid import uuid4
from omt_utils import gen_sin, gen_cos, gen_sawtooth, gen_triangle, gen_rectangle, offset, write, scale, multiply

SAMPLE_RATE = 48000
SAMPLES = 48000*5
parameter = namedtuple('parameter', 'amplitude function frequency offset side level head_uu')
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
        if side == 'x':
            self.combo_box.setCurrentText('cos')
        self.combo_box.currentTextChanged.connect(self.update_parameters)

        self.frequency_spin_box = QDoubleSpinBox()
        self.frequency_spin_box.setRange(0, SAMPLE_RATE/2)
        self.frequency_spin_box.setSingleStep(0.02)
        self.frequency_spin_box.setValue(400)
        self.frequency_spin_box.valueChanged.connect(self.update_parameters)

        self.offset_spin_box = QDoubleSpinBox()
        self.offset_spin_box.setRange(-1, 1)
        self.offset_spin_box.setSingleStep(0.02)
        self.offset_spin_box.setValue(0)
        self.offset_spin_box.valueChanged.connect(self.update_parameters)

        self.update_parameters()

        self.addWidget(self.amplitude_spin_box)
        self.addWidget(self.combo_box)
        self.addWidget(self.frequency_spin_box)
        self.addWidget(self.offset_spin_box)

    def update_parameters(self) -> None:
        amplitude = self.amplitude_spin_box.value()
        function = self.combo_box.currentText()
        frequency = self.frequency_spin_box.value()
        offset = self.offset_spin_box.value()
        p = parameter(amplitude=amplitude, function=function,
                      frequency=frequency, offset=offset, side=self.side,
                      level=0, head_uu=None)
        parameters[self.uu] = p
        print(parameters[self.uu])


gen_sig = {'sin': gen_sin, 'cos': gen_cos, 'saw': gen_sawtooth,
           'tri': gen_triangle, 'rec': gen_rectangle}


def calc():
    x_samples = []
    y_samples = []
    for param in parameters.values():
        f = param.frequency
        # Signal
        signal = gen_sig[param.function](f, SAMPLE_RATE, SAMPLES)
        # Amplitude
        if param.amplitude != 1.0:
            samples = scale(signal, param.amplitude)
        else:
            samples = signal
        # Offset
        if param.offset != 0:
            samples = offset(samples, param.offset)
        # Multiply
        if param.side == 'x':
            if x_samples:
                x_samples = multiply(x_samples, samples)
            else:
                x_samples = samples
        elif param.side == 'y':
            if y_samples:
                y_samples = multiply(y_samples, samples)
            else:
                y_samples = samples
    write(x_samples, y_samples, SAMPLE_RATE=SAMPLE_RATE)
    print('calc done')


class XYLayout(QVBoxLayout):
    def build(self, side) -> None:
        self.side = side

        add_button = QPushButton()
        add_button.setText('Add signal')
        add_button.clicked.connect(self.add_selector)

        label = QLabel()
        label.setText('   Amp         Function       f in Hz            Offset')

        self.addWidget(add_button)
        self.addWidget(label)
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
