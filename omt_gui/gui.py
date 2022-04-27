from collections import namedtuple, OrderedDict
import csv
import os
from os import remove
from typing import Union

import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QComboBox, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QRadioButton, QSpinBox, QSizePolicy, QSpacerItem, QTabWidget, QLineEdit, QVBoxLayout, QWidget


import omt_image_utils

try:
    import matplotlib.image as mat_img
    matplotlib_missing = False
except ImportError:
    matplotlib_missing = True

from uuid import uuid4
from omt_utils import add, clip, gen_sin, gen_cos, gen_morph, gen_triangle, gen_sawtooth, gen_rectangle, gen_x_over_y, offset, write, read, scale, multiply

SAMPLE_RATE = 192000
SAMPLES = SAMPLE_RATE*5
parameter = namedtuple('parameter', 'operator amplitude function frequency offset clip side file level hierarchy')
merges = OrderedDict()


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
    local_path = __file__.replace('gui.py', '')
    file_path, _ = w.getOpenFileName(caption='Open File',
                                     filter=_filter,
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


class Selector(QHBoxLayout):
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu: str, side: str, signal=None, operator: str = '*',
              amplitude: float = 1.0, frequency: float = 400,
              offset: float = 0.0, clip: float = 1.0, file: str = '',
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

        self.operator_combo_box = QComboBox()
        self.operator_combo_box.setToolTip('Operation')
        self.operator_combo_box.addItems(['*', '+'])
        self.operator_combo_box.setCurrentText(operator)
        self.operator_combo_box.currentTextChanged.connect(self.update_parameters)

        self.amplitude_spin_box = QDoubleSpinBox()
        self.amplitude_spin_box.setToolTip('Amplitude 0..10')
        self.amplitude_spin_box.setMaximum(10)
        self.amplitude_spin_box.setSingleStep(0.1)
        self.amplitude_spin_box.setValue(amplitude)
        self.amplitude_spin_box.valueChanged.connect(self.update_parameters)

        self.combo_box = QComboBox()
        self.combo_box.setToolTip('Function')
        self.combo_box.addItems(('sin', 'cos', 'saw', 'tri', 'rec', 'comb', 'x^f', 'file'))
        if side == 'x' and signal == None:
            self.combo_box.setCurrentText('cos')
        else:
            self.combo_box.setCurrentText(signal)
        self.combo_box.currentTextChanged.connect(self.change_visible_widgets)
        self.combo_box.currentTextChanged.connect(self.update_parameters)

        self.file = file
        self.file_select_button = QPushButton()
        self.file_select_button.setText('Select file')
        self.file_select_button.clicked.connect(self.select_file)

        self.file_show_path_line_edit = QLineEdit()
        self.file_show_path_line_edit.setToolTip('Selected filepath ' + self.file)
        self.file_show_path_line_edit.setEnabled(False)
        if self.file:
            self.file_show_path_line_edit.setText(self.file)

        self.frequency_spin_box = QDoubleSpinBox()
        self.frequency_spin_box.setToolTip('f in Hz')
        self.frequency_spin_box.setRange(0, SAMPLE_RATE/2)
        self.frequency_spin_box.setSingleStep(0.02)
        self.frequency_spin_box.setValue(frequency)
        self.frequency_spin_box.valueChanged.connect(self.update_parameters)

        self.offset_spin_box = QDoubleSpinBox()
        self.offset_spin_box.setToolTip('Offset')
        self.offset_spin_box.setRange(-1, 1)
        self.offset_spin_box.setSingleStep(0.02)
        self.offset_spin_box.setValue(offset)
        self.offset_spin_box.valueChanged.connect(self.update_parameters)

        self.clip_box = QDoubleSpinBox()
        self.clip_box.setToolTip('Clip boundary 0..10; 1 is default; >1 allows higher amplitude setting')
        self.clip_box.setRange(0, 10)
        self.clip_box.setSingleStep(0.05)
        self.clip_box.setValue(clip)
        self.clip_box.valueChanged.connect(self.update_parameters)

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
        self.addWidget(self.operator_combo_box)
        self.addWidget(self.amplitude_spin_box)
        self.addWidget(self.combo_box)
        self.addWidget(self.file_select_button)
        self.addWidget(self.file_show_path_line_edit)
        self.addWidget(self.frequency_spin_box)
        self.addWidget(self.offset_spin_box)
        self.addWidget(self.clip_box)
        if signal == 'file':
            self.amplitude_spin_box.setVisible(False)
            self.frequency_spin_box.setVisible(False)
            self.offset_spin_box.setVisible(False)
            self.clip_box.setVisible(False)
        else:
            self.file_select_button.setVisible(False)
            self.file_show_path_line_edit.setVisible(False)
        self.addWidget(self.delete_button)
        self.addSpacerItem(self.end_spacer)

        self.update_spacer()

    def add_parameters(self, _parameters: dict) -> None:
        self.parameters = _parameters

    def update_parameters(self) -> None:
        operator = self.operator_combo_box.currentText()
        amplitude = round(self.amplitude_spin_box.value(), 2)
        signal = self.combo_box.currentText()
        frequency = self.frequency_spin_box.value()
        offset = round(self.offset_spin_box.value(), 2)
        clip = round(self.clip_box.value(), 2)
        file = self.file

        self.parameter = parameter(operator=operator, amplitude=amplitude,
                                   function=signal, frequency=frequency,
                                   offset=offset, clip=clip, side=self.side,
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
        if self.combo_box.currentText() == 'file':
            self.amplitude_spin_box.setVisible(False)
            self.frequency_spin_box.setVisible(False)
            self.offset_spin_box.setVisible(False)
            self.clip_box.setVisible(False)

            self.file_select_button.setVisible(True)
            self.file_show_path_line_edit.setVisible(True)
        else:
            self.file_select_button.setVisible(False)
            self.file_show_path_line_edit.setVisible(False)

            self.amplitude_spin_box.setVisible(True)
            self.frequency_spin_box.setVisible(True)
            self.offset_spin_box.setVisible(True)
            self.clip_box.setVisible(True)
        self.frequency_spin_box.setVisible(self.combo_box.currentText() != 'comb')

    def select_file(self) -> None:
        path = show_load_file_pop_up(_filter="WAV File (*.wav)")
        if os.path.isfile(path):
            # TODO save relative path not absolute
            self.file = path
            self.file_show_path_line_edit.setText(path)
            self.file_show_path_line_edit.setToolTip('Selected filepath ' + self.file)

            self.amplitude_spin_box.setValue(1.0)  # default will prevent calc
            self.offset_spin_box.setValue(0.0)  # default will prevent calc
            self.clip_box.setValue(1.0)  # default will prevent calc
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

        self.operator_combo_box.deleteLater()
        self.amplitude_spin_box.deleteLater()
        self.combo_box.deleteLater()
        self.file_select_button.deleteLater()
        self.file_show_path_line_edit.deleteLater()
        self.frequency_spin_box.deleteLater()
        self.offset_spin_box.deleteLater()
        self.clip_box.deleteLater()
        self.delete_button.deleteLater()
        del self.parameters[self.uu]


class MorphSelector(QHBoxLayout):
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu: str, side: str, signal=None, level: int = 0,
              hierarchy=None) -> None:
        self.uu = uu
        self.side = side

        self.level = level
        if hierarchy:
            self.hierarchy = hierarchy
        else:
            hierarchies_on_this_side = [_.hierarchy for _ in self.parameters.values()]
            if hierarchies_on_this_side:
                self.hierarchy = max(hierarchies_on_this_side) + 1
            else:
                self.hierarchy = 0

        self.spacer = QSpacerItem(0, 0, vPolicy=QSizePolicy.Maximum)

        self.level_buttons = LevelButtons()
        self.level_buttons.build(self.uu)

        self.file_1_button = QPushButton()
        self.file_1_button.setText('Select File 1')
        self.file_2_button = QPushButton()
        self.file_2_button.setText('Select File 2')

        self.delete_button = QPushButton('x')
        self.delete_button.setToolTip('delete line')
        self.delete_button.setMaximumWidth(15)
        self.delete_button.clicked.connect(self.delete)

        self.end_spacer = QSpacerItem(0, 0, hPolicy=QSizePolicy.Expanding)

        self.update_merges()

        self.addSpacerItem(self.spacer)
        self.addLayout(self.level_buttons)
        self.addWidget(self.file_1_button)
        self.addWidget(self.file_2_button)
        self.addWidget(self.delete_button)
        self.addSpacerItem(self.end_spacer)

        self.update_spacer()

    def update_merges(self) -> None:
        self.merge = parameter(operator=1, amplitude=1,
                                   function=1, frequency=1,
                                   offset=1, clip=clip, side=None,
                                   level=self.level, hierarchy=self.hierarchy)
        merges[self.uu] = self.merge

    def update_spacer(self) -> None:
        self.level = merges[self.uu].level
        self.spacer.changeSize(50 * self.level, 0)
        self.invalidate()

    def delete(self) -> None:
        """Implements functionality of the delete button"""
        number_selectors = len([None for p in merges.values()])
        if number_selectors > 1:
            self.remove()
            self.hierarchy_changed.emit((self.uu, 999, 999))
        else:
            show_error_message('Deletion error', 'Can\'t delete only remaining selector')

    def remove(self) -> None:
        self.level_buttons.remove()
        self.level_buttons.deleteLater()
        self.file_1_button.deleteLater()
        self.file_2_button.deleteLater()

        self.delete_button.deleteLater()
        del merges[self.uu]


gen_sig = {'sin': gen_sin, 'cos': gen_cos, 'saw': gen_sawtooth,
           'tri': gen_triangle, 'rec': gen_rectangle, 'x^f': gen_x_over_y}


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


def combine(signal_list: list[tuple[list, str]], parameters: dict) -> list[float]:
    """ Combine list of signals
    :param signal_list: [(signal, uuid)]
    :param parameters: parameter dict of one side
    :return: combined signal
    """
    t_signal = signal_list[0][0]  # Load first signal
    for signal, _uu in signal_list[1:]:
        operator = parameters[_uu].operator
        if operator == '+':
            t_signal = add(signal, t_signal)
        elif operator == '*':
            t_signal = multiply(signal, t_signal)
    return t_signal


def calc_comb(param: parameter, signal: list) -> list[float]:
    # Amplitude
    if param.amplitude != 1.0:
        signal = scale(signal, param.amplitude)
    # Offset
    if param.offset != 0:
        signal = offset(signal, param.offset)
    # Clip
    if param.clip != 1.0:
        signal = clip(signal, param.clip)
    return signal


def valid_tree(tree: list[tuple[str, int, int]], parameters: dict) -> str:
    tree_len = len(tree)
    side = parameters[tree[0][0]].side
    if tree_len == 1:
        uu = tree[0][0]
        if parameters[uu].function == 'comb':
            return f'First selector in {side} can\'t be comb.'
    else:
        first_level = tree[0][2]
        if first_level != 0:
            return 'First selector needs to be on top level.'
        for i_h in range(1, tree_len):
            previous_uu = tree[i_h - 1][0]
            previous_l = tree[i_h - 1][2]
            current_l = tree[i_h][2]
            if previous_l >= current_l:  # End of comb / On same level
                continue
            elif previous_l + 1 == current_l:  # Correct comb
                if parameters[previous_uu].function != 'comb':
                    return f'Selector {i_h} in {side} should be comb.'
                continue
            elif previous_l < current_l:  #
                return f'Selector {i_h + 1} in {side} on wrong level / level to low.'
            else:
                print(f'pre{previous_l}, cur{current_l}')
                raise NotImplementedError

        # Check if gen selector is below comb selector
        for i_h in range(0, tree_len-1):
            current_uu = tree[i_h][0]
            current_l = tree[i_h][2]
            next_l = tree[i_h + 1][2]
            if (parameters[current_uu].function == 'comb'
                    and next_l != current_l + 1):
                return f'Comb on level {current_l + 1} in {side} can\'t be empty. Add a selector on lower level or remove comb.'


def calc_signal_one_level(level: int, tree: list[tuple[str, int, int]], parameters: dict):
    h_list = [(None, None, None)] * len(tree)
    target_comb_index = None
    for (i, (uu, h, l)) in enumerate(tree):
        if l != level:
            target_comb_index = None
            continue
        else:
            if target_comb_index == None:  # New part to combine
                target_comb_index = i - 1
        param = parameters[uu]
        if param.function != 'comb':
            signal = calc_signal(param)
            h_list[h] = (signal, uu, target_comb_index)
        else:
            h_list[h] = (None, uu, target_comb_index)
    return h_list


def combine_one_level(level: int, tree: list[tuple[str, int, int]], h_list, parameters: dict):
    c_list = [None] * len(tree)
    combinations_on_level = [(uu, h, l) for (uu, h, l) in tree
                             if parameters[uu].function == 'comb'
                             and l == level]
    for comb_uu, h, _level in combinations_on_level:
        signals_with_comb_index = [(signal, uu) for (signal, uu, target_comb_index)
                                   in h_list if target_comb_index == h]
        if len(signals_with_comb_index) > 0:
            combined_signal = combine(signals_with_comb_index, parameters)
            c_list[h] = calc_comb(parameters[comb_uu], combined_signal)
    return c_list


def collapse_tree(level: int, tree: list[tuple[str, int, int]], c_list) -> tuple[list[tuple[str, int, int]], list]:
    # Chop of highest tree level
    t_tree = []
    t_signals = []
    for i, ((uu, h_index, l_index), c_signal) in enumerate(zip(tree, c_list)):
        if l_index < level:
            t_tree.append((uu, len(t_tree), l_index))
            t_signals.append(c_signal)
            continue
    return t_tree, t_signals


def calc_selector_tree(tree: list[tuple[str, int, int]], parameters: dict):
    t_tree = tree
    t_signals = None
    for _l in reversed(range(1, 1 + max([_l for (_, _, _l) in tree]))):
        hierarchy_list = calc_signal_one_level(_l, t_tree, parameters)
        if t_signals:  # Merge signals with last combination
            hierarchy_list = [h_l if h_l[0] is not None else (t_s, h_l[1], h_l[2])  # h_l[0] == signal
                              for (h_l, t_s) in zip(hierarchy_list, t_signals)]
        combined_list = combine_one_level(_l - 1, t_tree, hierarchy_list, parameters)
        t_tree, t_signals = collapse_tree(_l, t_tree, combined_list)

    # Top Level
    hierarchy_list = calc_signal_one_level(0, t_tree, parameters)
    if t_signals:
        t_signals = [h_l if h_l[0] is not None else (t_s, h_l[1], h_l[2])
                     for (h_l, t_s) in zip(hierarchy_list, t_signals)]
    else:  # No combs in tree
        t_signals = hierarchy_list
    top = combine([(signal, uu) for (signal, uu, _l) in t_signals], parameters)
    return top


class XYLayout(QVBoxLayout):
    def build(self, side, _parameters, default_selector=True) -> None:
        self.side = side

        add_button = QPushButton()
        add_button.setText(f'Add {side} signal')
        add_button.clicked.connect(self.add_selector)

        self.addWidget(add_button)
        self.setAlignment(Qt.AlignTop)
        self.parameters = _parameters
        if default_selector:
            self.add_selector()

    def add_selector(self, selector=None) -> None:
        if selector:
            _selector = selector
        else:
            _selector = Selector()
            uu = str(uuid4())
            _selector.add_parameters(self.parameters)
            _selector.build(uu, side=self.side)
        _selector.hierarchy_changed.connect(self.update_order)
        self.addLayout(_selector)

    def remove_selector(self, uu: str) -> None:
        for child in self.findChildren(Selector):
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

            s = Selector()
            s.add_parameters(self.parameters)
            s.build(uu=uu, side=side, signal=signal, operator=operator,
                    amplitude=amplitude, frequency=frequency, offset=offset,
                    clip=clip, file=file, level=level, hierarchy=hierarchy)

            self.add_selector(selector=s)


class MergeLayout(XYLayout):
    def build(self, default_selector=True) -> None:

        add_button = QPushButton()
        add_button.setText(f'Add Merge')
        add_button.clicked.connect(self.add_selector)

        self.addWidget(add_button)
        self.setAlignment(Qt.AlignTop)
        if default_selector:
            self.add_selector()

    def add_selector(self, selector=None) -> None:
        if selector:
            _selector = selector
        else:
            _selector = MorphSelector()
            uu = str(uuid4())
            _selector.build(uu, side=None)
        _selector.hierarchy_changed.connect(self.update_order)
        self.addLayout(_selector)


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


class ImageDisplay(QCheckBox):
    def __init__(self, parent=None):
        super(ImageDisplay, self).__init__(parent)
        self.clicked.connect(self.on_click)

        self.image = QLabel()
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        self.ffmpeg_available = os.system('ffmpeg') == 1
        self.ffmpeg_available = 0  # disabled until the video_widget correctly displays content

        self.w = self.build_window()
        self.w.setWindowTitle('OMT-GUI Audio as Image')

    def on_click(self, clicked: bool) -> None:
        if clicked:
            self.w.show()
        else:
            self.w.hide()

    def build_window(self) -> QWidget:
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        if self.ffmpeg_available:
            main_layout.addWidget(self.video_widget)
        else:
            main_layout.addWidget(self.image)
        main_widget.setLayout(main_layout)
        return main_widget

    def refresh_image(self, x_samples: list, y_samples: list) -> None:
        if not self.ffmpeg_available:
            file_name = 'temp_' + str(uuid4()) + '.png'
            omt_image_utils.convert_audio_to_image((x_samples, y_samples), file_name=file_name)
            self.image.setPixmap(QPixmap(file_name))
            try:
                remove(file_name)
            except (PermissionError, FileNotFoundError):
                pass
            self.update()
        else:
            fps = 30
            chunk_size = int(SAMPLE_RATE / fps)
            files = []

            # Write files
            uu = uuid4()
            for i in range(fps):
                file_name = f'temp_{uu}_{i:02d}.png'
                files.append(file_name)
                start = i * chunk_size
                end = start + chunk_size
                omt_image_utils.convert_audio_to_image((x_samples[start:end],
                                                        y_samples[start:end]),
                                                       file_name=file_name)
                self.image.setPixmap(QPixmap(file_name))

            # Combine files
            output_file_name = 'animation.mp4'
            os.system(f'ffmpeg -y -i temp_{uu}_%02d.png -c:v libx264 -vf fps={fps} -pix_fmt yuv420p {output_file_name}')

            # Remove files
            for file_name in files:
                try:
                    remove(file_name)
                except (PermissionError, FileNotFoundError):
                    pass

            # Update Widget
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(output_file_name)))
            self.player.play()


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
        sample_rate.currentTextChanged.connect(set_sample_rate)

        duration = QComboBox()
        duration.setToolTip('Duration to render')
        duration.addItems(('5s', '10s', '1s', 'calc lcm'))
        duration.currentTextChanged.connect(set_samples)

        start_button = QPushButton()
        start_button.setToolTip('Start rendering and save wav')
        start_button.setText('Start')
        start_button.clicked.connect(self.start)

        control_layout = QHBoxLayout()
        control_layout.addWidget(load_button)
        control_layout.addWidget(save_button)
        if not matplotlib_missing:
            wav_img = ImageDisplay()
            wav_img.setText('Show wav as image')
            control_layout.addWidget(wav_img)
        control_layout.addWidget(sample_rate)
        control_layout.addWidget(duration)
        control_layout.addWidget(start_button)

        x_y_widget = self._build_x_y_widget()

        # main layout
        self.main_h_box = QVBoxLayout()
        self.main_h_box.addWidget(x_y_widget)
        self.main_h_box.addLayout(control_layout)
        self.setLayout(self.main_h_box)

    def _build_x_y_widget(self) -> QWidget:
        # Add layouts to main widget
        self.x_layout = XYLayout()
        self.x_parameters = OrderedDict()
        self.x_layout.build('x', self.x_parameters)
        self.y_layout = XYLayout()
        self.y_parameters = OrderedDict()
        self.y_layout.build('y', self.y_parameters)
        x_y_layout = QHBoxLayout()
        x_y_layout.addLayout(self.x_layout)
        x_y_layout.addLayout(self.y_layout)

        x_y_widget = QWidget()
        x_y_widget.setLayout(x_y_layout)
        return x_y_widget

    def load(self, path: str = '') -> None:
        if not os.path.isfile(path):
            file_path = show_load_file_pop_up()
            if not file_path:
                return None
        else:
            file_path = path

        # clear x and y layout
        for selector in self.x_layout.findChildren(Selector):
            self.x_layout.remove_selector(selector.uu)
        for selector in self.y_layout.findChildren(Selector):
            self.y_layout.remove_selector(selector.uu)

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

                s = Selector()
                if side == 'x':
                    s.add_parameters(self.x_parameters)
                elif side == 'y':
                    s.add_parameters(self.y_parameters)
                s.build(uu=uu, side=side, signal=signal, operator=operator,
                        amplitude=amplitude, frequency=frequency, offset=offset,
                        clip=clip, file=file, level=level, hierarchy=hierarchy)

                if description['side'] == 'x':
                    self.x_layout.add_selector(selector=s)
                elif description['side'] == 'y':
                    self.y_layout.add_selector(selector=s)

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
            for uu, p in self.y_parameters.items():
                data = [p.operator, p.amplitude, p.function, p.frequency,
                        p.offset, p.clip, p.side, p.file, p.level, p.hierarchy, uu]
                writer.writerow(data)

    def start(self) -> None:
        print('calc start')
        x_tree = [(uu, p.hierarchy, p.level) for (uu, p) in self.x_parameters.items()]
        y_tree = [(uu, p.hierarchy, p.level) for (uu, p) in self.y_parameters.items()]
        valid_x = valid_tree(x_tree, self.x_parameters)
        valid_y = valid_tree(y_tree, self.y_parameters)

        if valid_x != None:
            show_error_message('Configuration error', valid_x)
            return None
        if valid_y != None:
            show_error_message('Configuration error', valid_y)
            return None

        x_samples = calc_selector_tree(x_tree, self.x_parameters)
        y_samples = calc_selector_tree(y_tree, self.y_parameters)
        print('calc done')

        # Check for for to large values
        if max(x_samples) > 1.0 or min(x_samples) < -1.0:
            warnings.warn('X samples to big to convert, samples will be rerendered with clipping')
            x_samples = clip(x_samples, 1.0)
        if max(y_samples) > 1.0 or min(y_samples) < -1.0:
            warnings.warn('Y samples to big to convert, samples will be rerendered with clipping')
            y_samples = clip(y_samples, 1.0)
        write(x_samples, y_samples, sample_rate=SAMPLE_RATE)

        if not matplotlib_missing:
            wav_img = self.findChild(ImageDisplay)
            if wav_img.isChecked():
                wav_img.refresh_image(x_samples, y_samples)
        del x_samples
        del y_samples


def create_gui(*, path: str = ''):
    root = GUI()
    root.build()
    if path != '':
        root.load(path)
    root.setWindowTitle('OMT-GUI ' + path)
    return root
