import logging
from math import sin, tau
from cmath import rect, polar, phase
from typing import Final
#logging.basicConfig(level=logging.DEBUG)

FRAMERATE: Final = 48000
CHANNELS: Final = 2
SAMPLEWIDTH: Final = 2


def _limit(frame, clip=None):
    """
    Limit the input frame to the highest/lowest value allowed in a wav file
    :param frame:
    :return: limited frame
    :rtype: int
    """
    if clip is None:
        UPPER_XY_LIMIT = 32767
        LOWER_XY_LIMIT = -32767
    else:
        UPPER_XY_LIMIT = int(32767 * clip)
        LOWER_XY_LIMIT = int(-32767 * clip)
    if frame > UPPER_XY_LIMIT:
        return UPPER_XY_LIMIT
    elif frame < LOWER_XY_LIMIT:
        return LOWER_XY_LIMIT
    return frame


class Modifier:
    __slots__ = ('frequency', 'phi', 'magnitude', 'offset', 'affected_sample',
                 'duration_left', 'swap', 'clip')

    def __init__(self, frequency=None, phi=None, magnitude=None, offset=None,
                 start=None, duration=None, swap=False, clip=None, t_modifier=1.0):
        if duration and swap:
            raise NotImplementedError

        if duration == -1:  # Affect all samples
            divisor = FRAMERATE
            self.duration_left = -1
        elif duration is None:  # One sample
            divisor = 1
            self.duration_left = 1
        else:  # Given amount of seconds in samples
            divisor = duration * FRAMERATE
            self.duration_left = duration * FRAMERATE

        if start is not None:
            self.affected_sample = start * FRAMERATE
        else:
            self.affected_sample = 0

        if swap:
            self.frequency = frequency
            self.phi = phi
            self.magnitude = magnitude
            self.offset = offset
        else:
            self.frequency = frequency / divisor if frequency else None
            self.phi = phi / divisor if phi else None
            self.magnitude = magnitude / divisor if magnitude else None
            self.offset = offset / divisor if offset else None

        if clip is not None:
            if clip > 1.0 or clip < 0.0:
                raise ValueError
        self.clip = clip
        self.swap = swap

    def __repr__(self):
        return (f'frequency={self.frequency}, phi={self.phi}, '
                f'magnitude={self.magnitude}, offset={self.offset}, '
                f'self.duration_left={self.duration_left}, '
                f'affected_sample={self.affected_sample}, swap={self.swap}, '
                f'clip={self.clip}')


class BasicWave(object):
    def __init__(self, frequency, phi=0.0, magnitude=1.0, offset=0.0,
                 list_of_modifiers=None, t_modifier=1):
        """

        :param frequency: Hz
        :param phi: sin = phi=0.0; cos = phi=math.pi/2
        :param magnitude:
        :param offset: DC offset
        """
        logging.debug(f'Creating BasicWave f={frequency}Hz, p={phi}Phi, '
                      f'a={magnitude}, offset={offset}, t_modifier={t_modifier}')

        if type(frequency) is not int and type(frequency) is not float:
            raise ValueError('Frequency must be int or float not '
                             f'{type(frequency)}')
        self.frequency = frequency

        if type(phi) is not int and type(phi) is not float:
            raise ValueError('Phi must be int or float not '
                             f'{type(phi)}={str(phi)}. Phi set to 0.0')
        self.phi = phi

        if type(magnitude) is not int and type(magnitude) is not float:
            raise ValueError('Magnitude must be int or float not '
                             f'{type(magnitude)}={str(magnitude)}. ')
        if magnitude > 1.0 or magnitude < -1.0:
            logging.warning('Magnitude must be between '
                            f'-1.0 and 1.0 is {magnitude} '
                            'magnitude set to the closer one.')
            self.magnitude = 1.0 if magnitude > 1.0 else -1.0
        else:
            self.magnitude = magnitude

        if type(offset) is not int and type(offset) is not float:
            raise ValueError('Offset must be int or float not '
                             f'{type(offset)}={str(offset)}. ')
        if offset > 1.0 or offset < -1.0:
            logging.warning('Offset must be between '
                            f'-1.0 and 1.0 is {offset} '
                            'offset set to 1.0')
            self.offset = 0.0
        else:
            self.offset = offset

        self.t = 0
        self.t_modifier = t_modifier
        self.modifiers = set()
        self.clip = None
        if list_of_modifiers:
            if type(list_of_modifiers) is not list:
                raise ValueError('list_of_modifiers must be list not '
                                 f'{type(list_of_modifiers)}='
                                 f'{str(list_of_modifiers)}. ')
            for mod in list_of_modifiers:
                self.add_modifier(mod)
        logging.debug(f'Created BasicWave {str(self)}')

        phi_per_t = (tau / FRAMERATE) * self.frequency * t_modifier
        self.c = rect(magnitude, phi)
        self.c_t = rect(1, phi_per_t)

    def __str__(self):
        for f, p, a, o, mods ,tm in self._wave_description():
            return f'f={f}Hz, p={p}Phi, a={a}, offset={o}; mods={mods}; t_modifier={tm};'

    def __add__(self, other):
        if isinstance(other, BasicWave) or isinstance(other, Wave):
            return Wave(wave_description=(self._wave_description()
                                          + other._wave_description()))
        else:
            raise NotImplementedError

    def add_modifier(self, modifier):
        self.modifiers.add(modifier)

    def _modify(self, t):
        for mod in self.modifiers:
            affect = t >= mod.affected_sample and mod.duration_left > 0
            if affect or mod.duration_left == -1:
                if mod.swap:
                    if mod.frequency is not None:
                        self.frequency = mod.frequency
                        phi_per_t = (tau / FRAMERATE) * self.frequency * self.t_modifier
                        print(polar(self.c_t))
                        self.c_t = rect(1, phi_per_t)
                        mod.frequency = None

                    if mod.phi is not None:
                        self.phi = mod.phi
                        self.c = rect(abs(self.c), mod.phi)
                        mod.phi = None
                    if mod.magnitude is not None:
                        self.magnitude = mod.magnitude
                        self.c = rect(self.magnitude, phase(self.c))
                        mod.magnitude = None
                    if mod.offset is not None:
                        self.offset = mod.offset
                        mod.offset = None
                    mod.swap = False
                else:
                    if mod.frequency:
                        self.frequency += mod.frequency
                        phi_per_t = (tau / FRAMERATE) * self.frequency * self.t_modifier
                        self.c_t = rect(1, phi_per_t)
                    if mod.phi:
                        self.phi += mod.phi
                        self.c = rect(abs(self.c), mod.phi)
                    if mod.magnitude:
                        self.magnitude += mod.magnitude
                        self.c = rect(self.magnitude, phase(self.c))
                    if mod.offset:
                        self.offset += mod.offset
                if mod.clip is not None:
                    self.clip = mod.clip
                mod.duration_left -= 1
                # TODO Remove mod with duration_left == 0

    def _wave_description(self):
        """
        Returns a list of tuples. Each tuple describes a frequency
        :return: wave description
        :rtype: list
        """
        wave_description = [(self.frequency, self.phi, self.magnitude,
                             self.offset, [], self.t_modifier)]
        return wave_description

    def calculate_frame(self, t):
        """
        Calculate one frame without self.offset as integer
        :return: frame
        :rtype: int
        """
        frame_ = self.c * self.c_t
        self.c = frame_
        self._modify(t)
        return frame_.real

    def play(self, in_bytes=True):
        """
        Returns a wav data frame
        :param in_bytes:
        :return: frame
        :rtype: bytes, int
        """
        # Formula x = a*sin(w(t)+p) * scaling + offset
        frame = (self.calculate_frame(self.t)
                 * 32767.0 + (32767.0 * self.offset))
        self.t += 1  # TODO reset self.t to keep small to save memory
        b = _limit(int(frame), self.clip).to_bytes(2, byteorder='little', signed=True)
        return b if in_bytes else frame


class Wave(object):
    @staticmethod
    def _sort_wave_description(wave_description):
        return sorted(wave_description, key=lambda tup: tup[0], reverse=True)

    def __init__(self, wave_description, t=None):
        """
        :param wave_description: tuple of f and p
        :param t:
        """
        logging.debug(f'Creating Wave wave_description={wave_description}')
        self.frequencies = []
        if type(wave_description) is not list:
            raise NotImplementedError
        if type(wave_description[0]) is not tuple:
            raise NotImplementedError
        if type(wave_description[0]) is BasicWave:
            # TODO order by frequency; add test
            for wave in wave_description:
                self.frequencies.append(wave)
        else:
            descriptions = self._sort_wave_description(wave_description)
            for description in descriptions:
                frequency = description[0]
                phi = description[1]
                magnitude = description[2]
                offset = description[3]
                modifiers = description[4] if len(description) >= 5 else None
                t_modifier = description[5] if len(description) >= 6 else 1.0
                if magnitude == 0.0:
                    logging.info(f'{frequency=}Hz with magnitude of 0.0')
                self.frequencies.append(BasicWave(frequency=frequency,
                                                  phi=phi,
                                                  magnitude=magnitude,
                                                  offset=offset,
                                                  t_modifier=t_modifier))
                if modifiers:
                    for mod in modifiers:
                        self.frequencies[-1].add_modifier(mod)

        self.t = t if t else 0
        logging.debug('Created Wave wave_description='
                      f'{self._wave_description()}')

    def __str__(self):
        s = ''
        for wave in self.frequencies:
            s += str(wave)
        return s

    def __add__(self, other):
        return Wave(wave_description=(self._wave_description() +
                                      other._wave_description()),
                    t=self.t)

    @classmethod
    def from_desc(cls, frequency=0.0, phi=0.0, magnitude=1.0, offset=0.0,
                  t=None):
        return cls([(frequency, phi, magnitude, offset)], t)

    def _wave_description(self):
        """
        Returns a list of tuples. Each tuple describes a frequency
        :return: wave description
        :rtype: list
        """
        wave_description = []
        for wave in self.frequencies:
            wave_description += wave._wave_description()
        return wave_description

    def calculate_frame(self):
        frame = 0
        for basic_wave in self.frequencies:
            frame += basic_wave.calculate_frame(t=self.t)
        offset = 0.0
        for basic_wave in self.frequencies:
            offset += basic_wave.offset
        frame = frame * 32767.0 + offset * 32767.0
        self.t += 1
        return frame

    def play(self):
        """
        Returns a wav data frame
        :return: frame
        :rtype: bytes
        """
        frame = self.calculate_frame()
        return _limit(int(frame)).to_bytes(2, byteorder='little', signed=True)
