from datetime import date as dt_date
from functools import partial
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QComboBox, QCheckBox, QDateEdit, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSpinBox, QSizePolicy, QSpacerItem, QStackedWidget, QTabWidget, QVBoxLayout, QWidget
from typing import Callable, List, Union
from collections import namedtuple
import csv

from uuid import uuid4
from omt_utils import gen_sin, gen_cos, gen_sawtooth, gen_triangle, gen_rectangle, offset, write, scale, multiply

SAMPLE_RATE = 192000
SAMPLES = SAMPLE_RATE*5
parameter = namedtuple('parameter', 'operator amplitude function frequency offset side level head_uu')
parameters = {}


def show_load_file_pop_up(focus_root: Union[QWidget, None] = None) -> str:
    """
    Show a file dialog to select a file.
    Sets the focus back to the main window, if focus_root is set to a parent widget.
    :param focus_root: QWidget to set focus to
    :return: Path as string
    """
    w = QFileDialog()
    w.setFileMode(QFileDialog.ExistingFile)
    local_path = __file__.replace('gui.py', '')
    file_path, _ = w.getOpenFileName(caption='Open File',
                                     filter="CSV File (*.csv)",
                                     directory=local_path)
    # Set focus back to root when called from outside
    if focus_root:
        focus_root.activateWindow()
    # Check path and create db
    return file_path


def show_save_file_pop_up(focus_root: Union[QWidget, None] = None) -> str:
    """
    Show a file dialog to select a file.
    Sets the focus back to the main window, if focus_root is set to a parent widget.
    :param focus_root: QWidget to set focus to
    :return: Path as string
    """
    w = QFileDialog()
    w.setFileMode(QFileDialog.ExistingFile)
    local_path = __file__.replace('gui.py', '')
    file_path, _ = w.getSaveFileName(caption='Open File',
                                     filter="CSV File (*.csv)",
                                     directory=local_path)
    # Set focus back to root when called from outside
    if focus_root:
        focus_root.activateWindow()
    # Check path and create db
    return file_path


class Selector(QHBoxLayout):
    def build(self, uu: str, side: str, signal=None, operator='+', amplitude=1.0,
              frequency=400, offset=0) -> None:
        self.uu = uu
        self.side = side

        self.operator_combo_box = QComboBox()
        self.operator_combo_box.addItems(['+', '*'])
        self.operator_combo_box.setCurrentText(operator)

        self.amplitude_spin_box = QDoubleSpinBox()
        self.amplitude_spin_box.setMaximum(10)
        self.amplitude_spin_box.setSingleStep(0.1)
        self.amplitude_spin_box.setValue(amplitude)
        self.amplitude_spin_box.valueChanged.connect(self.update_parameters)

        self.combo_box = QComboBox()
        self.combo_box.addItems(['sin', 'cos', 'saw', 'tri', 'rec'])
        if side == 'x' and signal == None:
            self.combo_box.setCurrentText('cos')
        if signal:
            self.combo_box.setCurrentText(signal)
        self.combo_box.currentTextChanged.connect(self.update_parameters)

        self.frequency_spin_box = QDoubleSpinBox()
        self.frequency_spin_box.setRange(0, SAMPLE_RATE/2)
        self.frequency_spin_box.setSingleStep(0.02)
        self.frequency_spin_box.setValue(frequency)
        self.frequency_spin_box.valueChanged.connect(self.update_parameters)

        self.offset_spin_box = QDoubleSpinBox()
        self.offset_spin_box.setRange(-1, 1)
        self.offset_spin_box.setSingleStep(0.02)
        self.offset_spin_box.setValue(offset)
        self.offset_spin_box.valueChanged.connect(self.update_parameters)

        self.update_parameters()

        self.addWidget(self.operator_combo_box)
        self.addWidget(self.amplitude_spin_box)
        self.addWidget(self.combo_box)
        self.addWidget(self.frequency_spin_box)
        self.addWidget(self.offset_spin_box)

    def update_parameters(self) -> None:
        operator = self.operator_combo_box.currentText()
        amplitude = self.amplitude_spin_box.value()
        function = self.combo_box.currentText()
        frequency = self.frequency_spin_box.value()
        offset = self.offset_spin_box.value()
        self.parameter = parameter(operator=operator, amplitude=amplitude, function=function,
                      frequency=frequency, offset=offset, side=self.side,
                      level=0, head_uu=None)
        parameters[self.uu] = self.parameter
        #print(parameters[self.uu])

    def remove(self) -> None:
        self.operator_combo_box.deleteLater()
        self.amplitude_spin_box.deleteLater()
        self.combo_box.deleteLater()
        self.frequency_spin_box.deleteLater()
        self.offset_spin_box.deleteLater()
        del parameters[self.uu]


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
    def build(self, side, default_selector=True) -> None:
        self.side = side

        add_button = QPushButton()
        add_button.setText('Add signal')
        add_button.clicked.connect(self.add_selector)

        label = QLabel()
        label.setText('   Amp         Function       f in Hz            Offset')

        self.addWidget(add_button)
        self.addWidget(label)
        if default_selector:
            self.add_selector()

    def add_selector(self, selector=None) -> None:
        if selector:
            _selector = selector
        else:
            _selector = Selector()
            uu = uuid4()
            _selector.build(uu, side=self.side)
        self.addLayout(_selector)

    def remove_selector(self, uu) -> None:
        for child in self.findChildren(Selector):
            if child.uu == uu:
                child.remove()
                child.deleteLater()


class GUI(QWidget):
    def build(self):
        """
        Build the gui and connect all signals and slots
        """

        load_button = QPushButton()
        load_button.setText('Load')
        load_button.clicked.connect(self.load)

        save_button = QPushButton()
        save_button.setText('Save')
        save_button.clicked.connect(self.save)

        start_button = QPushButton()
        start_button.setText('Start')
        start_button.clicked.connect(calc)

        control_layout = QVBoxLayout()
        control_layout.addWidget(load_button)
        control_layout.addWidget(save_button)
        control_layout.addWidget(start_button)

        # Add layouts to main widget
        main_h_box = QHBoxLayout()
        self.x_layout = XYLayout()
        self.x_layout.build('x')
        self.y_layout = XYLayout()
        self.y_layout.build('y')
        main_h_box.addLayout(self.x_layout)
        main_h_box.addLayout(self.y_layout)
        main_h_box.addLayout(control_layout)

        self.setLayout(main_h_box)

    def load(self):
        file_path = show_load_file_pop_up()
        if not file_path:
            return None

        # clear x and y layout
        for selector in self.x_layout.findChildren(Selector):
            self.x_layout.remove_selector(selector.uu)
        for selector in self.y_layout.findChildren(Selector):
            self.y_layout.remove_selector(selector.uu)

        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for description in reader:
                uu = description['uu']
                side = description['side']
                operator = description['operator']
                amplitude = float(description['amplitude'])
                frequency = float(description['frequency'])
                signal = description['function']
                offset = float(description['offset'])

                s = Selector()
                s.build(uu=uu, side=side, signal=signal, operator=operator,
                        amplitude=amplitude, frequency=frequency, offset=offset)

                if description['side'] == 'x':
                    self.x_layout.add_selector(selector=s)
                elif description['side'] == 'y':
                    self.y_layout.add_selector(selector=s)

    def save(self):
        file_path = show_save_file_pop_up()
        if not file_path:
            return None

        with open(file_path, 'w') as file:
            print(parameters)
            writer = csv.writer(file)
            writer.writerow(('operator', 'amplitude', 'function', 'frequency', 'offset', 'side', 'level', 'head_uu', 'uu'))
            for uu, p in parameters.items():
                data = [p.amplitude, p.function, p.frequency, p.offset, p.side,
                        p.level, p.head_uu, uu]
                writer.writerow(data)
