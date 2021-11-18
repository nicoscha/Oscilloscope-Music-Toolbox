import math
from datetime import date as dt_date
from functools import partial
from PyQt5.QtCore import QDate, Qt, pyqtSignal
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QComboBox, QCheckBox, QDateEdit, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSpinBox, QSizePolicy, QSpacerItem, QStackedWidget, QTabWidget, QVBoxLayout, QWidget
from typing import Callable, List, Union
from collections import namedtuple
import csv

from uuid import uuid4
from omt_utils import add, gen_sin, gen_cos, gen_sawtooth, gen_triangle, gen_rectangle, offset, write, scale, multiply

SAMPLE_RATE = 192000
SAMPLES = SAMPLE_RATE*5
parameter = namedtuple('parameter', 'operator amplitude function frequency offset side level hierarchy')
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


class HierarchyButtons(QVBoxLayout):
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu):
        self.uu = uu
        self.setSpacing(0)

        self.hierarchy_up = QPushButton('⯅')
        self.hierarchy_up.setMaximumWidth(15)
        self.hierarchy_up.setMaximumHeight(15)
        self.hierarchy_up.clicked.connect(self.up_clicked)
        self.hierarchy_down = QPushButton('⯆')
        self.hierarchy_down.setMaximumWidth(15)
        self.hierarchy_down.setMaximumHeight(15)
        self.hierarchy_down.clicked.connect(self.down_clicked)

        self.addWidget(self.hierarchy_up)
        self.addWidget(self.hierarchy_down)

    def up_clicked(self):
        hierarchy = parameters[self.uu].hierarchy
        if hierarchy > 0:
            parameters[self.uu] = parameters[self.uu]._replace(hierarchy=None)
            self.hierarchy_changed.emit((self.uu, hierarchy, -1))

    def down_clicked(self):
        hierarchy = parameters[self.uu].hierarchy
        own_side = parameters[self.uu].side
        current_max = max([p.hierarchy for p in parameters.values() if p.side == own_side])
        if hierarchy < current_max:  # Arbitrary limit
            parameters[self.uu] = parameters[self.uu]._replace(hierarchy=None)
            self.hierarchy_changed.emit((self.uu, hierarchy, +1))

    def remove(self):
        self.hierarchy_up.deleteLater()
        self.hierarchy_down.deleteLater()


class LevelButtons(QHBoxLayout):
    level_changed = pyqtSignal()
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu):
        self.uu = uu
        self.setSpacing(0)

        self.level_up = QPushButton('⯇')
        self.level_up.setMaximumWidth(15)
        self.level_up.clicked.connect(self.up_clicked)
        self.level_down= QPushButton('⯈')
        self.level_down.setMaximumWidth(15)
        self.level_down.clicked.connect(self.down_clicked)
        self.hierarchy_buttons = HierarchyButtons()
        self.hierarchy_buttons.build(self.uu)

        self.addWidget(self.level_up)
        self.addLayout(self.hierarchy_buttons)
        self.addWidget(self.level_down)

        self.hierarchy_buttons.hierarchy_changed.connect(self.hierarchy_changed.emit)

    def up_clicked(self):
        if parameters[self.uu].level > 0:
            level = parameters[self.uu].level
            parameters[self.uu] = parameters[self.uu]._replace(level=level - 1)
            self.level_changed.emit()

    def down_clicked(self):
        if parameters[self.uu].level <= 5:  # Arbitrary limit
            level = parameters[self.uu].level
            parameters[self.uu] = parameters[self.uu]._replace(level=level + 1)
            self.level_changed.emit()

    def remove(self):
        self.level_up.deleteLater()

        self.hierarchy_buttons.remove()
        self.hierarchy_buttons.deleteLater()

        self.level_down.deleteLater()


class Selector(QHBoxLayout):
    hierarchy_changed = pyqtSignal(tuple)

    def build(self, uu: str, side: str, signal=None, operator='*', amplitude: float = 1.0,
              frequency: float = 400, offset: float = 0.0, level: int = 0, hierarchy=None) -> None:
        self.uu = uu
        self.side = side

        self.level = level
        if hierarchy:
            self.hierarchy = hierarchy
        else:
            hierarchies_on_this_side = [_.hierarchy for _ in parameters.values() if _.side == self.side]
            if hierarchies_on_this_side:
                self.hierarchy = max(hierarchies_on_this_side) + 1
            else:
                self.hierarchy = 0

        self.spacer = QSpacerItem(0, 0, vPolicy=QSizePolicy.Maximum)

        self.level_buttons = LevelButtons()
        self.level_buttons.build(self.uu)

        self.operator_combo_box = QComboBox()
        self.operator_combo_box.addItems(['*', '+'])
        self.operator_combo_box.setCurrentText(operator)
        self.operator_combo_box.currentTextChanged.connect(self.update_parameters)

        self.amplitude_spin_box = QDoubleSpinBox()
        self.amplitude_spin_box.setMaximum(10)
        self.amplitude_spin_box.setSingleStep(0.1)
        self.amplitude_spin_box.setValue(amplitude)
        self.amplitude_spin_box.valueChanged.connect(self.update_parameters)

        self.combo_box = QComboBox()
        self.combo_box.addItems(('sin', 'cos', 'saw', 'tri', 'rec', 'comb'))
        if side == 'x' and signal == None:
            self.combo_box.setCurrentText('cos')
        else:
            self.combo_box.setCurrentText(signal)
        self.combo_box.currentTextChanged.connect(self.update_parameters)

        self.frequency_spin_box = QDoubleSpinBox()
        self.frequency_spin_box.setRange(0, SAMPLE_RATE/2)
        self.frequency_spin_box.setSingleStep(0.02)
        self.frequency_spin_box.setValue(frequency)
        self.frequency_spin_box.valueChanged.connect(self.update_parameters)
        if signal == 'comb':
            self.frequency_spin_box.setEnabled(False)

        self.offset_spin_box = QDoubleSpinBox()
        self.offset_spin_box.setRange(-1, 1)
        self.offset_spin_box.setSingleStep(0.02)
        self.offset_spin_box.setValue(offset)
        self.offset_spin_box.valueChanged.connect(self.update_parameters)

        self.update_parameters()
        self.level_buttons.level_changed.connect(self.update_spacer)
        self.level_buttons.hierarchy_changed.connect(self.hierarchy_changed.emit)

        self.addSpacerItem(self.spacer)
        self.addLayout(self.level_buttons)
        self.addWidget(self.operator_combo_box)
        self.addWidget(self.amplitude_spin_box)
        self.addWidget(self.combo_box)
        self.addWidget(self.frequency_spin_box)
        self.addWidget(self.offset_spin_box)

        self.update_spacer()

    def update_parameters(self) -> None:
        operator = self.operator_combo_box.currentText()
        amplitude = round(self.amplitude_spin_box.value(), 2)
        signal = self.combo_box.currentText()
        frequency = self.frequency_spin_box.value()
        offset = round(self.offset_spin_box.value(), 2)

        if signal == 'comb':
            self.frequency_spin_box.setEnabled(False)
        else:
            self.frequency_spin_box.setEnabled(True)

        self.parameter = parameter(operator=operator, amplitude=amplitude, function=signal,
                                   frequency=frequency, offset=offset, side=self.side,
                                   level=self.level, hierarchy=self.hierarchy)
        parameters[self.uu] = self.parameter
        #print(parameters[self.uu])

    def update_spacer(self):
        self.level = parameters[self.uu].level
        self.spacer.changeSize(50 * self.level, 0)
        self.invalidate()

    def remove(self) -> None:
        self.level_buttons.remove()
        self.level_buttons.deleteLater()

        self.operator_combo_box.deleteLater()
        self.amplitude_spin_box.deleteLater()
        self.combo_box.deleteLater()
        self.frequency_spin_box.deleteLater()
        self.offset_spin_box.deleteLater()
        del parameters[self.uu]


gen_sig = {'sin': gen_sin, 'cos': gen_cos, 'saw': gen_sawtooth,
           'tri': gen_triangle, 'rec': gen_rectangle}


def calc():
    print('calc start')
    x_samples = []
    y_samples = []

    raw_x_signals = {}
    raw_y_signals = {}
    raw_signals = {}
    for uu, param in parameters.items():
        if param.function == 'comb':
            continue
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
        if param.side == 'x':
            raw_x_signals[uu] = samples
        elif param.side == 'y':
            raw_y_signals[uu] = samples
        raw_signals[uu] = samples

    for side in ['x', 'y']:
        params_values_filtered = [v for v in parameters.values() if v.side == side]
        params_items_filtered = [i for i in parameters.items() if i[1].side == side]
        max_level = max([param.level for param in params_values_filtered])
        max_hierarchy = max([param.hierarchy for param in params_values_filtered])
        sorted_uu_params = sorted(params_items_filtered, key=lambda _: _[1].hierarchy)
        for level in range(max_level - 1, -1, -1):  # level -1 because the lowest level can not contain comb signals(doesn't make sens if it could)
            # print('Level:', level)
            params_with_same_level = [p for p in sorted_uu_params if p[1].level == level]
            for (uu, param) in params_with_same_level:
                # print(f'   Signal/Hierarchy/Level: {param.function:<4}', str(param.hierarchy), str(param.level))
                if param.function == 'comb':
                    last_signal_in_comb_tree = None

                    range_hierarchy_til_end = range(param.hierarchy + 1, max_hierarchy + 1)
                    for t_hierarchy in range_hierarchy_til_end:
                        # print('      ', t_hierarchy)
                        if sorted_uu_params[t_hierarchy][1].level >= (level + 1):  # (param.level + 1)
                            last_signal_in_comb_tree = t_hierarchy
                        else:
                            # Next Path
                            # print('break on', t_hierarchy)
                            break
                        print('         Last sig', last_signal_in_comb_tree)

                    if last_signal_in_comb_tree == None:
                        # print(f'Warning: hierarchy {param.hierarchy} empty')
                        raw_signals[uu] = [1]  # evade calc error
                        continue

                    # Next signal has different level -> calc this path and save to comb then break
                    path_parameters = sorted_uu_params[param.hierarchy + 1:last_signal_in_comb_tree + 1]
                    if len(path_parameters) == 1:
                        uu_signal_in_path = path_parameters[0][0]
                        raw_signals[uu] = raw_signals[uu_signal_in_path]
                    else:
                        first_uu_in_list = path_parameters[0][0]
                        t_signal = raw_signals[first_uu_in_list]
                        for _uu, operator in [(_uu, p.operator) for _uu, p in path_parameters[1:]]:
                            signal_1 = raw_signals[_uu]
                            if operator == '+':
                                t_signal = add(signal_1, t_signal)
                            elif operator == '*':
                                t_signal = multiply(signal_1, t_signal)
                        raw_signals[uu] = t_signal
                    # print(f'         sigs in path: {path_parameters}')
                    # Amplitude
                    if param.amplitude != 1.0:
                        raw_signals[uu] = scale(raw_signals[uu], param.amplitude)
                    # Offset
                    if param.offset != 0:
                        raw_signals[uu] = offset(raw_signals[uu], param.offset)

                    break
        del params_values_filtered
        del params_items_filtered

        # Add/Multiply on level 0
        level_0 = [_ for _ in sorted_uu_params if _[1].level == 0]
        print(level_0)
        t_signal = raw_signals[level_0[0][0]]
        for _uu, operator in [(_uu, p.operator) for _uu, p in level_0[1:]]:
            print('First level')
            signal_1 = raw_signals[_uu]
            if operator == '+':
                t_signal = add(signal_1, t_signal)
            elif operator == '*':
                t_signal = multiply(signal_1, t_signal)

        if side == 'x':
            x_samples = t_signal
        elif side == 'y':
            y_samples = t_signal

    for uu, sig in raw_signals.items():
        print(uu, len(sig))

    try:
        write(x_samples, y_samples, sample_rate=SAMPLE_RATE)
    except Exception as E:
        print(E)
    print('calc done')


class XYLayout(QVBoxLayout):
    def build(self, side, default_selector=True) -> None:
        self.side = side

        add_button = QPushButton()
        add_button.setText('Add signal')
        add_button.clicked.connect(self.add_selector)

        label = QLabel()
        label.setText('\tOperation  Amplitue  Function       f in Hz            Offset')

        self.addWidget(add_button)
        self.addWidget(label)
        if default_selector:
            self.add_selector()

    def add_selector(self, selector=None) -> None:
        if selector:
            _selector = selector
        else:
            _selector = Selector()
            uu = str(uuid4())
            _selector.build(uu, side=self.side)
        _selector.hierarchy_changed.connect(self.update_order)
        self.addLayout(_selector)

    def remove_selector(self, uu) -> None:
        for child in self.findChildren(Selector):
            if child.uu == uu:
                child.remove()
                child.deleteLater()

    def update_order(self, changed_selector: tuple[str, int, int]):
        c_parameters = parameters.copy()
        # Remove all selectors on this side
        _parameters = []
        for uu in c_parameters:
            if c_parameters[uu].side == self.side:
                _parameters.append((uu, parameters[uu]))
                self.remove_selector(uu)

        # Order selectors
        changed_selector_uu = changed_selector[0]
        changed_selector_hierarchy = changed_selector[1]
        change = changed_selector[2]  # +1 or -1
        new_index = changed_selector_hierarchy + change

        new_h = [None] * len(_parameters)
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
            (operator, amplitude, signal, frequency, offset, side, level, hierarchy) = param

            s = Selector()
            s.build(uu=uu, side=side, signal=signal, operator=operator,
                    amplitude=amplitude, frequency=frequency, offset=offset,
                    level=level, hierarchy=hierarchy)

            self.add_selector(selector=s)


def set_sample_rate(sample_rate_str: str) -> None:
    global SAMPLE_RATE
    SAMPLE_RATE = int(float(sample_rate_str.replace('k Hz', '')) * 1000)


def set_samples(samples_str: str) -> None:
    global SAMPLES
    if samples_str == 'calc lcm':
        SAMPLES = int(SAMPLE_RATE / math.lcm(*[int(p.frequency) for p in parameters.values()]))
        print('lcm of', SAMPLES, 'samples')
    else:
        SAMPLES = int(samples_str.replace('s', '')) * SAMPLE_RATE


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

        sample_rate = QComboBox()
        sample_rate.addItems(('192k Hz', '96k Hz', '48k Hz', '44.1k Hz'))
        sample_rate.currentTextChanged.connect(set_sample_rate)

        duration = QComboBox()
        duration.addItems(('5s', '10s', '1s', 'calc lcm'))
        duration.currentTextChanged.connect(set_samples)

        start_button = QPushButton()
        start_button.setText('Start')
        start_button.clicked.connect(calc)

        control_layout = QHBoxLayout()
        control_layout.addWidget(load_button)
        control_layout.addWidget(save_button)
        control_layout.addWidget(sample_rate)
        control_layout.addWidget(duration)
        control_layout.addWidget(start_button)

        # Add layouts to main widget
        self.main_h_box = QVBoxLayout()
        self.x_layout = XYLayout()
        self.x_layout.build('x')
        self.y_layout = XYLayout()
        self.y_layout.build('y')
        self.x_y_layout = QHBoxLayout()
        self.x_y_layout.addLayout(self.x_layout)
        self.x_y_layout.addLayout(self.y_layout)
        self.main_h_box.addLayout(self.x_y_layout)
        self.main_h_box.addLayout(control_layout)

        self.setLayout(self.main_h_box)

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
                level = int(description['level'])
                hierarchy = int(description['hierarchy'])

                s = Selector()
                s.build(uu=uu, side=side, signal=signal, operator=operator,
                        amplitude=amplitude, frequency=frequency, offset=offset,
                        level=level, hierarchy=hierarchy)

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
            writer.writerow(('operator', 'amplitude', 'function', 'frequency', 'offset', 'side', 'level', 'hierarchy', 'uu'))
            for uu, p in parameters.items():
                data = [p.operator, p.amplitude, p.function, p.frequency, p.offset, p.side,
                        p.level, p.hierarchy, uu]
                writer.writerow(data)
