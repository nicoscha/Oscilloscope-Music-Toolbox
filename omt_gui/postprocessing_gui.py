from collections import namedtuple, OrderedDict
import csv
import os
from os import remove
from typing import Union

import warnings

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QComboBox, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QRadioButton, QSpinBox, QSizePolicy, QSpacerItem, QTabWidget, QLineEdit, QVBoxLayout, QWidget


import omt_image_utils

from uuid import uuid4
from omt_utils import add, clip, gen_morph, offset, write, read, scale, multiply
from omt_image_utils import binominal_filter
SAMPLE_RATE = 192000
SAMPLES = SAMPLE_RATE*5
parameter = namedtuple('parameter', 'function filter_type filter_length cutoff_frequency window file level hierarchy')


def show_load_file_pop_up(focus_root: Union[QWidget, None] = None,
                          _filter: str = "CSV File (*.csv)") -> str:
    """
    Show a file dialog to select a file.
    Sets the focus back to the main window, if focus_root is set to a parent widget.
    :param focus_root: QWidget to set focus to
    :param _filter: File type filter. Default: "CSV File (*.csv)"
    :return: Path as string
    """
    w = QFileDialog()
    w.setFileMode(QFileDialog.ExistingFile)
    local_path = __file__.replace('postprocessing_gui.py', '')
    file_path, _ = w.getOpenFileName(caption='Open File',
                                     filter=_filter,
                                     directory=local_path)
    # Set focus back to root when called from outside
    if focus_root:
        focus_root.activateWindow()
    # Check path and create db
    return file_path


def show_save_file_pop_up(focus_root: Union[QWidget, None] = None,
                          _filter: str = "CSV File (*.csv)") -> str:
    """
    Show a file dialog to select a file.
    Sets the focus back to the main window, if focus_root is set to a parent widget.
    :param focus_root: QWidget to set focus to
    :return: Path as string
    """
    w = QFileDialog()
    w.setFileMode(QFileDialog.ExistingFile)
    local_path = __file__.replace('postprocessing_gui.py', '')
    file_path, _ = w.getSaveFileName(caption='Path to save File',
                                     filter=_filter,
                                     directory=local_path)
    # Set focus back to root when called from outside
    if focus_root:
        focus_root.activateWindow()
    # Check path and create db
    return file_path


class HierarchyButtons(QVBoxLayout):
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu: str, _parameters: dict) -> None:
        self.uu = uu
        self.parameters = _parameters
        self.setSpacing(0)

        self.hierarchy_up = QPushButton('⯅')
        self.hierarchy_up.setToolTip('Move 1 hierarchy up')
        self.hierarchy_up.setMaximumWidth(15)
        self.hierarchy_up.setMaximumHeight(15)
        self.hierarchy_up.clicked.connect(self.up_clicked)
        self.hierarchy_down = QPushButton('⯆')
        self.hierarchy_down.setToolTip('Move 1 hierarchy down')
        self.hierarchy_down.setMaximumWidth(15)
        self.hierarchy_down.setMaximumHeight(15)
        self.hierarchy_down.clicked.connect(self.down_clicked)

        self.addWidget(self.hierarchy_up)
        self.addWidget(self.hierarchy_down)

    def up_clicked(self) -> None:
        hierarchy = self.parameters[self.uu].hierarchy
        if hierarchy > 0:
            self.parameters[self.uu] = self.parameters[self.uu]._replace(hierarchy=None)
            self.hierarchy_changed.emit((self.uu, hierarchy, -1))

    def down_clicked(self) -> None:
        hierarchy = self.parameters[self.uu].hierarchy
        current_max = max([p.hierarchy for p in self.parameters.values()])
        if hierarchy < current_max:  # Arbitrary limit
            self.parameters[self.uu] = self.parameters[self.uu]._replace(hierarchy=None)
            self.hierarchy_changed.emit((self.uu, hierarchy, +1))

    def remove(self) -> None:
        self.hierarchy_up.deleteLater()
        self.hierarchy_down.deleteLater()


class LevelButtons(QHBoxLayout):
    level_changed = pyqtSignal()
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu: str, _parameters: dict) -> None:
        self.uu = uu
        self.parameters = _parameters
        self.setSpacing(0)

        self.level_up = QPushButton('⯇')
        self.level_up.setToolTip('Move 1 level up')
        self.level_up.setMaximumWidth(15)
        self.level_up.clicked.connect(self.up_clicked)
        self.level_down = QPushButton('⯈')
        self.level_down.setToolTip('Move 1 level down')
        self.level_down.setMaximumWidth(15)
        self.level_down.clicked.connect(self.down_clicked)
        self.hierarchy_buttons = HierarchyButtons()
        self.hierarchy_buttons.build(self.uu, self.parameters)

        self.addWidget(self.level_up)
        self.addLayout(self.hierarchy_buttons)
        self.addWidget(self.level_down)

        self.hierarchy_buttons.hierarchy_changed.connect(self.hierarchy_changed.emit)

    def up_clicked(self) -> None:
        if self.parameters[self.uu].level > 0:
            level = self.parameters[self.uu].level
            self.parameters[self.uu] = self.parameters[self.uu]._replace(level=level - 1)
            self.level_changed.emit()

    def down_clicked(self) -> None:
        if self.parameters[self.uu].level <= 5:  # Arbitrary limit
            level = self.parameters[self.uu].level
            self.parameters[self.uu] = self.parameters[self.uu]._replace(level=level + 1)
            self.level_changed.emit()

    def remove(self) -> None:
        self.level_up.deleteLater()

        self.hierarchy_buttons.remove()
        self.hierarchy_buttons.deleteLater()

        self.level_down.deleteLater()


class Filter(QWidget):
    parameter_changed = pyqtSignal()

    def __init__(self, uu, filter_type='binominal'):
        super().__init__()
        self.uu = uu

        self.filter_box = QComboBox()
        self.filter_box.setToolTip('Function')
        self.filter_box.addItems(('No filter', 'Binominal', 'Low-pass', 'High-pass',
                                  'Band-pass', 'Butterworth'))
        self.filter_box.setCurrentText(filter_type)
        self.filter_box.currentTextChanged.connect(self.change_visible_widgets)
        self.filter_box.currentTextChanged.connect(self.update_parameters)

        self.filter_length_spin_box = QSpinBox()
        self.filter_length_spin_box.setToolTip('Filter length')
        self.filter_length_spin_box.setMinimum(1)
        self.filter_length_spin_box.setMaximum(1024)
        self.filter_length_spin_box.setValue(3)
        self.filter_length_spin_box.setSingleStep(1)

        self.cutoff_frequency_box = QDoubleSpinBox()
        self.cutoff_frequency_box.setToolTip('cutoff frequency 0..1')
        self.cutoff_frequency_box.setMinimum(0.0)
        self.cutoff_frequency_box.setMaximum(1.0)
        self.cutoff_frequency_box.setValue(0.5)
        self.cutoff_frequency_box.setSingleStep(0.01)

        self.window_box = QComboBox()
        self.window_box.setToolTip('Windows function')
        self.window_box.addItems(('No window', 'Hanning', 'Hamming'))
        # Hanning cos^2 Hamming cos^2 auf podest Butterworth
        self.window_box.setCurrentText(filter_type)
        self.window_box.currentTextChanged.connect(self.change_visible_widgets)
        self.window_box.currentTextChanged.connect(self.update_parameters)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.filter_box)
        self.layout.addWidget(self.filter_length_spin_box)
        self.layout.addWidget(self.cutoff_frequency_box)
        self.layout.addWidget(self.window_box)

        self.setLayout(self.layout)

        self.change_visible_widgets()
        #self.setVisible(False)
        #self.filter_box.setVisible(False)
        #self.filter_length_spin_box.setVisible(False)

    def change_visible_widgets(self) -> None:
        if self.filter_box.currentText() == 'No filter':
            self.filter_length_spin_box.setVisible(False)

            self.cutoff_frequency_box.setVisible(False)
        elif self.filter_box.currentText() == 'Binominal':
            self.filter_length_spin_box.setVisible(True)

            self.cutoff_frequency_box.setVisible(False)
        else:
            self.cutoff_frequency_box.setVisible(True)

            self.filter_length_spin_box.setVisible(False)

    def update_parameters(self):
        self.parameter_changed.emit()


class Processor(QHBoxLayout):
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu: str, side: str, function=None, operator: str = '*',
              amplitude: float = 1.0, frequency: float = 400,
              offset: float = 0.0, clip: float = 1.0, file: str = '',
              out_file: str = '',
              level: int = 0, hierarchy=None) -> None:
        self.uu = uu
        self.side = side

        self.level = level
        if hierarchy:
            self.hierarchy = hierarchy
        else:
            hierarchies = [_.hierarchy for _ in self.parameters.values()]
            if hierarchies:
                self.hierarchy = max(hierarchies) + 1
            else:
                self.hierarchy = 0

        self.spacer = QSpacerItem(0, 0, vPolicy=QSizePolicy.Maximum)

        self.level_buttons = LevelButtons()
        self.level_buttons.build(self.uu, self.parameters)

        self.combo_box = QComboBox()
        self.combo_box.setToolTip('Function: ' + function)
        self.combo_box.addItems(('file', 'morph'))
        self.combo_box.setCurrentText(function)
        self.combo_box.currentTextChanged.connect(self.change_visible_widgets)
        self.combo_box.currentTextChanged.connect(self.update_parameters)

        self.filter = Filter(self.uu)
        self.filter.parameter_changed.connect(self.update_parameters)

        self.file = file
        self.file_select_button = QPushButton()
        self.file_select_button.setText('Select file')
        self.file_select_button.clicked.connect(self.select_file)

        self.file_show_path_line_edit = QLineEdit()
        self.file_show_path_line_edit.setToolTip('Selected filepath ' + self.file)
        self.file_show_path_line_edit.setEnabled(False)
        if self.file:
            self.file_show_path_line_edit.setText(self.file)

        self.output_file = out_file
        self.output_file_select_button = QPushButton()
        self.output_file_select_button.setText('Select output file')
        self.output_file_select_button.clicked.connect(self.select_file)

        self.output_file_show_path_line_edit = QLineEdit()
        self.output_file_show_path_line_edit.setToolTip('Selected output filepath ' + self.output_file)
        self.output_file_show_path_line_edit.setEnabled(False)
        if self.output_file:
            self.output_file_show_path_line_edit.setText(self.output_file)

        self.delete_button = QPushButton('x')
        self.delete_button.setToolTip('delete line')
        self.delete_button.setMaximumWidth(15)
        self.delete_button.clicked.connect(self.delete)

        self.end_spacer = QSpacerItem(0, 0, hPolicy=QSizePolicy.Expanding)

        self.update_parameters()
        # Events
        self.level_buttons.level_changed.connect(self.update_spacer)
        self.level_buttons.level_changed.connect(self.adjust_enabled_on_operator_combo_box)
        self.level_buttons.hierarchy_changed.connect(self.hierarchy_changed.emit)
        self.level_buttons.hierarchy_changed.connect(self.adjust_enabled_on_operator_combo_box)

        self.addSpacerItem(self.spacer)
        self.addLayout(self.level_buttons)

        self.addWidget(self.combo_box)
        self.addWidget(self.filter)
        self.addWidget(self.file_select_button)
        self.addWidget(self.file_show_path_line_edit)
        self.addWidget(self.output_file_select_button)
        self.addWidget(self.output_file_show_path_line_edit)
        if function == 'file':  # TODO change_visible_widgets aufrufen
            self.file_select_button.setVisible(True)
            self.file_show_path_line_edit.setVisible(True)

            self.filter.setVisible(False)
            self.output_file_select_button.setVisible(False)
            self.output_file_show_path_line_edit.setVisible(False)
        elif function == 'filter':
            self.filter.setVisible(True)

            self.file_select_button.setVisible(False)
            self.file_show_path_line_edit.setVisible(False)
            self.output_file_select_button.setVisible(False)
            self.output_file_show_path_line_edit.setVisible(False)
        else:
            self.file_select_button.setVisible(False)
            self.file_show_path_line_edit.setVisible(False)
            self.filter.setVisible(False)
        self.addWidget(self.delete_button)
        self.addSpacerItem(self.end_spacer)

        self.update_spacer()

    def add_parameters(self, _parameters: dict) -> None:
        self.parameters = _parameters

    def update_parameters(self) -> None:
        function = self.combo_box.currentText()
        file = self.file
        filter_type = self.filter.filter_box.currentText()
        filter_length = round(self.filter.filter_length_spin_box.value(), 0)
        cutoff_frequency = round(self.filter.cutoff_frequency_box.value(), 2)
        window = self.filter.window_box.currentText()

        self.parameter = parameter(function=function,
                                   filter_type=filter_type,
                                   filter_length=filter_length,
                                   cutoff_frequency=cutoff_frequency,
                                   window=window,
                                   file=file,
                                   level=self.level, hierarchy=self.hierarchy)
        self.parameters[self.uu] = self.parameter
        #print(parameters[self.uu])

    def adjust_enabled_on_operator_combo_box(self) -> None:
        # Deactivate operator for the first element in a hierarchy
        h_and_l_on_side = [(p.hierarchy, p.level) for p in self.parameters.values()]
        l_above = [l for (h, l) in h_and_l_on_side if self.hierarchy - 1 == h]
        if l_above:
            disable_operator = l_above[0] + 1 <= self.level
            if disable_operator:
                self.operator_combo_box.setEnabled(False)
            else:
                self.operator_combo_box.setEnabled(True)

    def update_spacer(self) -> None:
        self.level = self.parameters[self.uu].level
        self.spacer.changeSize(50 * self.level, 0)
        self.invalidate()

    def change_visible_widgets(self) -> None:
        function = self.combo_box.currentText()
        if function == 'file':
            self.file_select_button.setVisible(True)
            self.file_show_path_line_edit.setVisible(True)

            self.filter.setVisible(False)
        elif function == 'filter':
            self.filter.setVisible(True)

            self.file_select_button.setVisible(False)
            self.file_show_path_line_edit.setVisible(False)
        else:
            self.file_select_button.setVisible(False)
            self.file_show_path_line_edit.setVisible(False)
            self.filter.setVisible(False)

    def select_file(self) -> None:
        if self.uu == 'output file':
            path = show_save_file_pop_up(_filter="WAV File (*.wav)")
            print(123)
        else:
            print(self.uu)
            path = show_load_file_pop_up(_filter="WAV File (*.wav)")
        if os.path.isfile(path):
            # TODO save relative path not absolute
            self.file = path
            self.file_show_path_line_edit.setText(path)
            self.file_show_path_line_edit.setToolTip('Selected filepath ' + self.file)

            self.update_parameters()

    def delete(self) -> None:
        """Implements functionality of the delete button"""
        number_selectors = len([None for p in self.parameters.values()])
        if number_selectors > 1:
            self.remove()
            self.hierarchy_changed.emit((self.uu, 999, 999))
        else:
            show_error_message('Deletion error', 'Can\'t delete only remaining selector')

    def remove(self) -> None:
        self.level_buttons.remove()
        self.level_buttons.deleteLater()

        self.combo_box.deleteLater()
        self.file_select_button.deleteLater()
        self.file_show_path_line_edit.deleteLater()
        self.delete_button.deleteLater()
        del self.parameters[self.uu]


gen_sig = {'Binominal': binominal_filter, 'Low-pass': binominal_filter,
           'High-pass': binominal_filter, 'Band-pass': binominal_filter,
           'Butterworth': binominal_filter}


def calc_signal(param: parameter) -> list[float]:
    if param.function == 'comb':
        raise ValueError
    f = param.frequency
    # Signal
    if param.function == 'file':
        sr, x_y_wav = read(param.file, norm=True)
        if sr != SAMPLE_RATE:
            warnings.warn(f'Sample rate of file: {param.file} does not match project setting')
        if len(x_y_wav) != 2:
            show_error_message('File error', f'{param.file} needs two audio channels')
            return gen_sig['sin'](f, SAMPLE_RATE, SAMPLES)
        if len(x_y_wav[0]) < SAMPLES:
            show_error_message('Content error', f'{param.file} does not contain enough samples')
            return gen_sig['sin'](f, SAMPLE_RATE, SAMPLES)
        if param.side == 'x':
            signal = x_y_wav[0][:SAMPLES]
        else:
            signal = x_y_wav[1][:SAMPLES]
    else:
        signal = gen_sig[param.function](f, SAMPLE_RATE, SAMPLES)
    # Amplitude
    if param.amplitude != 1.0:
        samples = scale(signal, param.amplitude)
    else:
        samples = signal
    # Offset
    if param.offset != 0:
        samples = offset(samples, param.offset)
    # Clip
    if param.clip != 1.0:
        samples = clip(samples, param.clip)
    return samples


def filter_signal(parameters: dict) -> tuple[np.array, np.array]:
    # read files
    in_path = [p.file for uu, p in parameters.items() if uu == 'input file'][0]
    sr, signal = read(in_path)
    # morph

    # filter
    filter_typ = parameters['filter'].filter_type
    filter_len = parameters['filter'].filter_length
    if filter_typ == 'Binominal':
        signal = gen_sig[filter_typ](signal[0], signal[1], filter_length=filter_len)

    # write files
    out_path = [p.file for uu, p in parameters.items() if p.file != ''][0]  ########## out file fehlt
    #file_name = 'abc.wav'
    write(signal[0], signal[1], sample_rate=sr, file_name=out_path)

    return signal


class PostLayout(QVBoxLayout):
    def build(self, side, _parameters, default_selector=True) -> None:
        self.side = side

        add_button = QPushButton()
        add_button.setText(f'Add processor')
        add_button.clicked.connect(self.add_selector)

        self.addWidget(add_button)
        self.setAlignment(Qt.AlignTop)
        self.parameters = _parameters
        if default_selector:
            self.add_selector()

    def add_selector(self, selector=None) -> None:
        if selector:
            _selector = selector
            _selector.hierarchy_changed.connect(self.update_order)
            self.addLayout(_selector)
        else:
            _selector_in_file = Processor()
            uu = 'input file'
            _selector_in_file.add_parameters(self.parameters)
            _selector_in_file.build(uu, side=self.side, function='file')
            _selector_in_file.hierarchy_changed.connect(self.update_order)
            _selector_in_file.level_buttons.level_up.setEnabled(False)
            _selector_in_file.level_buttons.level_down.setEnabled(False)
            _selector_in_file.level_buttons.hierarchy_buttons.hierarchy_up.setEnabled(False)
            _selector_in_file.level_buttons.hierarchy_buttons.hierarchy_down.setEnabled(False)

            _selector_in_file.hierarchy_changed.connect(self.update_order)
            self.addLayout(_selector_in_file)

            _selector = Processor()
            uu = 'filter'
            _selector.add_parameters(self.parameters)
            _selector.build(uu, side=self.side, function='filter')
            _selector.combo_box.setEnabled(False)
            _selector.level_buttons.level_up.setEnabled(False)
            _selector.level_buttons.level_down.setEnabled(False)
            _selector.level_buttons.hierarchy_buttons.hierarchy_up.setEnabled(False)
            _selector.level_buttons.hierarchy_buttons.hierarchy_down.setEnabled(False)

            _selector.hierarchy_changed.connect(self.update_order)
            self.addLayout(_selector)

            _selector_out_file = Processor()
            uu = 'output file'
            _selector_out_file.add_parameters(self.parameters)
            _selector_out_file.build(uu, side=self.side, function='output file')
            _selector_out_file.combo_box.setEnabled(False)
            _selector_out_file.hierarchy_changed.connect(self.update_order)
            _selector_out_file.level_buttons.level_up.setEnabled(False)
            _selector_out_file.level_buttons.level_down.setEnabled(False)
            _selector_out_file.level_buttons.hierarchy_buttons.hierarchy_up.setEnabled(False)
            _selector_out_file.level_buttons.hierarchy_buttons.hierarchy_down.setEnabled(False)

            _selector_out_file.hierarchy_changed.connect(self.update_order)
            self.addLayout(_selector_out_file)

    def remove_selector(self, uu: str) -> None:
        for child in self.findChildren(Processor):
            if child.uu == uu:
                child.remove()
                child.deleteLater()

    def update_order(self, changed_selector: tuple[str, int, int]):
        c_parameters = self.parameters.copy()
        # Remove all selectors on this side
        _parameters = []
        for uu in c_parameters:
            _parameters.append((uu, self.parameters[uu]))
            self.remove_selector(uu)

        # Order selectors
        changed_selector_uu = changed_selector[0]
        changed_selector_hierarchy = changed_selector[1]
        change = changed_selector[2]  # +1 or -1
        new_index = changed_selector_hierarchy + change

        new_h = [('', None)] * len(_parameters)

        if change == 999:  # Rearrange selector after deletion
            for i, (uu, param) in enumerate(_parameters):
                t_param = param._replace(hierarchy=i)
                new_h[i] = (uu, t_param)
        else:  # Rearrange selector after hierarchy change
            for uu, param in _parameters:
                if uu != changed_selector_uu:
                    old_index = param.hierarchy
                    if old_index < new_index:
                        new_h[old_index] = (uu, param)
                    elif old_index > new_index:
                        new_h[old_index] = (uu, param)
                    elif old_index == new_index:
                        t_param = param._replace(hierarchy=old_index - change)
                        new_h[old_index - change] = (uu, t_param)
                else:
                    new_param = param._replace(hierarchy=new_index)
                    new_h[new_index] = (uu, new_param)

        # Build new selectors
        for (uu, param) in new_h:
            (operator, amplitude, signal, frequency, offset, clip, side, file, level, hierarchy) = param

            s = Processor()
            s.add_parameters(self.parameters)
            s.build(uu=uu, side=side, function=signal, operator=operator,
                    amplitude=amplitude, frequency=frequency, offset=offset,
                    clip=clip, file=file, level=level, hierarchy=hierarchy)

            self.add_selector(selector=s)


def set_sample_rate(sample_rate_str: str) -> None:
    global SAMPLE_RATE
    SAMPLE_RATE = int(float(sample_rate_str.replace('k Hz', '')) * 1000)


def set_samples(samples_str: str) -> None:
    global SAMPLES
    if samples_str == 'calc lcm':
        SAMPLES = 48000 #int(SAMPLE_RATE / math.lcm(*[int(p.frequency) for p in parameters.values()]))
        print('lcm of', SAMPLES, 'samples')
    else:
        SAMPLES = int(samples_str.replace('s', '')) * SAMPLE_RATE


def show_error_message(title: str, message: str):
    error_message = QMessageBox()
    error_message.setWindowTitle(title)
    error_message.setText(message)
    error_message.exec_()


class GUI(QWidget):
    def build(self) -> None:
        """
        Build the gui and connect all signals and slots
        """
        load_button = QPushButton()
        load_button.setToolTip('Load configuration file')
        load_button.setText('Load')
        load_button.clicked.connect(self.load)

        save_button = QPushButton()
        save_button.setToolTip('Save current configuration')
        save_button.setText('Save')
        save_button.clicked.connect(self.save)

        sample_rate = QComboBox()
        sample_rate.setToolTip('Sample rate')
        sample_rate.addItems(('192k Hz', '96k Hz', '48k Hz', '44.1k Hz'))
        sample_rate.setEnabled(False)
        sample_rate.currentTextChanged.connect(set_sample_rate)

        start_button = QPushButton()
        start_button.setToolTip('Start rendering and save wav')
        start_button.setText('Start')
        start_button.clicked.connect(self.start)

        control_layout = QHBoxLayout()
        control_layout.addWidget(load_button)
        control_layout.addWidget(save_button)
        control_layout.addWidget(sample_rate)
        control_layout.addWidget(start_button)

        x_y_widget = self._build_x_y_widget()

        # main layout
        self.main_h_box = QVBoxLayout()
        self.main_h_box.addWidget(x_y_widget)
        self.main_h_box.addLayout(control_layout)
        self.setLayout(self.main_h_box)

    def _build_x_y_widget(self) -> QWidget:
        # Add layouts to main widget
        self.post_layout = PostLayout()
        self.x_parameters = OrderedDict()
        self.post_layout.build('x', self.x_parameters)

        x_y_widget = QWidget()
        x_y_widget.setLayout(self.post_layout)
        return x_y_widget

    def load(self, path: str = '') -> None:
        if not os.path.isfile(path):
            file_path = show_load_file_pop_up()
            if not file_path:
                return None
        else:
            file_path = path

        # clear x and y layout
        for selector in self.post_layout.findChildren(Processor):
            self.post_layout.remove_selector(selector.uu)

        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for description in reader:
                if 'clip' not in description:
                    description['clip'] = '1.0'
                if 'file' not in description:
                    description['file'] = ''

                uu = description['uu']
                side = description['side']
                operator = description['operator']
                amplitude = float(description['amplitude'])
                frequency = float(description['frequency'])
                signal = description['function']
                offset = float(description['offset'])
                clip = float(description['clip'])
                file = str(description['file'])
                level = int(description['level'])
                hierarchy = int(description['hierarchy'])

                # Check if file currently exists
                if not os.path.isfile(file):
                    file = ''

                s = Processor()
                if side == 'x':
                    s.add_parameters(self.x_parameters)
                s.build(uu=uu, side=side, function=signal, operator=operator,
                        amplitude=amplitude, frequency=frequency, offset=offset,
                        clip=clip, file=file, level=level, hierarchy=hierarchy)

                if description['side'] == 'x':
                    self.post_layout.add_selector(selector=s)

    def save(self) -> None:
        file_path = show_save_file_pop_up()
        if not file_path:
            return None

        with open(file_path, 'w') as file:
            print(self.x_parameters)
            print(self.y_parameters)
            writer = csv.writer(file)
            writer.writerow(('operator', 'amplitude', 'function', 'frequency',
                             'offset', 'clip', 'side', 'file', 'level',
                             'hierarchy', 'uu'))
            for uu, p in self.x_parameters.items():
                data = [p.operator, p.amplitude, p.function, p.frequency,
                        p.offset, p.clip, p.side, p.file, p.level, p.hierarchy, uu]
                writer.writerow(data)

    def start(self) -> None:
        print('calc start')

        samples = filter_signal(self.x_parameters)
        print('calc done')

        # Check for for to large values
        if max(samples[0]) > 1.0 or min(samples[0]) < -1.0:
            warnings.warn('X samples to big to convert, samples will be rerendered with clipping')
            x_samples = clip(samples[0], 1.0)
        if max(samples[1]) > 1.0 or min(samples[1]) < -1.0:
            warnings.warn('Y samples to big to convert, samples will be rerendered with clipping')
            y_samples = clip(samples[1], 1.0)
        write(samples[0], samples[1], sample_rate=SAMPLE_RATE)

        del x_samples



def create_gui(*, path: str = ''):
    root = GUI()
    root.build()
    if path != '':
        root.load(path)
    root.setWindowTitle('OMT-PostProcessing-GUI ' + path)
    return root
