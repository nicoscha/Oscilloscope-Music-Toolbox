import logging
from math import tau
from cmath import rect, phase
from typing import Final, List, Optional, Tuple, Union, Set
# logging.basicConfig(level=logging.DEBUG)

FRAMERATE: Final = 44100
CHANNELS: Final = 2
SAMPLEWIDTH: Final = 2
TAU_PER_FRAME = (tau / FRAMERATE)


def _limit(frame, clip: Optional[float] = 1) -> int:
    """
    Limit the input frame to the highest/lowest value allowed in a wav file
    :param frame:
    :return: limited frame
    :rtype: int
    """
    if clip == 1:
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

    def __init__(self, frequency: Optional[float] = None,
                 phi: Optional[float] = None, magnitude: Optional[float] = None,
                 offset: Optional[float] = None, start: Optional[float] = None,
                 duration: Optional[float] = None, swap: Optional[float] = False,
                 clip: Optional[float] = None, t_modifier: Optional[float] = 1):
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
            if clip > 1 or clip < 0:
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
    def __init__(self, frequency: float, phi: Optional[float] = 0,
                 magnitude: Optional[float] = 1, offset: Optional[float] = 0,
                 modifiers: Optional[List] = None, t_modifier: Optional[float] = 1):
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
        self.frequency: float = frequency

        if type(phi) is not int and type(phi) is not float:
            raise ValueError('Phi must be int or float not '
                             f'{type(phi)}={str(phi)}. Phi set to 0.0')
        self.phi: float = phi

        if type(magnitude) is not int and type(magnitude) is not float:
            raise ValueError('Magnitude must be int or float not '
                             f'{type(magnitude)}={str(magnitude)}. ')
        if magnitude > 1 or magnitude < -1:
            logging.warning('Magnitude must be between '
                            f'-1.0 and 1.0 is {magnitude} '
                            'magnitude set to the closer one.')
            self.magnitude: float = 1 if magnitude > 1 else -1
        else:
            self.magnitude: float = magnitude

        if type(offset) is not int and type(offset) is not float:
            raise ValueError('Offset must be int or float not '
                             f'{type(offset)}={str(offset)}. ')
        if offset > 1 or offset < -1:
            logging.warning('Offset must be between '
                            f'-1.0 and 1.0 is {offset} '
                            'offset set to 1.0')
            self.offset: float = 0
        else:
            self.offset: float = offset

        self.t: int = 0
        self.t_modifier: float = t_modifier
        self.modifiers: Set = set()
        self.clip: float = 1
        if modifiers:
            if type(modifiers) is not list:
                raise ValueError('modifiers must be list not '
                                 f'{type(modifiers)}={str(modifiers)}. ')
            for mod in modifiers:
                self.add_modifier(mod)
        logging.debug(f'Created BasicWave {str(self)}')

        phi_per_t = (tau / FRAMERATE) * self.frequency * t_modifier
        self.c: complex = rect(magnitude, phi)
        self.c_t: complex = rect(1, phi_per_t)

    def __str__(self):
        for f, p, a, o, mods, tm in self._wave_description():
            return f'f={f}Hz, p={p}rad, a={a}, offset={o}; mods={mods}; t_modifier={tm};'

    def __add__(self, other):
        if isinstance(other, BasicWave) or isinstance(other, Wave):
            return Wave(wave_description=(self._wave_description()
                                          + other._wave_description()))
        else:
            raise NotImplementedError

    def add_modifier(self, modifier: Modifier):
        self.modifiers.add(modifier)

    def _modify(self, t: int):
        drop_modifier: bool = False
        for mod in self.modifiers:
            affect: bool = t >= mod.affected_sample and mod.duration_left > 0
            if affect or mod.duration_left == -1:
                if mod.swap:
                    if mod.frequency is not None:
                        self.frequency = mod.frequency
                        phi_per_t = TAU_PER_FRAME * self.frequency * self.t_modifier
                        self.c_t = rect(1, phi_per_t)
                        mod.frequency = None
                    if mod.phi is not None:
                        self.phi = mod.phi
                        self.c = rect(self.c.real, mod.phi)
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
                        phi_per_t = TAU_PER_FRAME * self.frequency * self.t_modifier
                        self.c_t = rect(1, phi_per_t)
                    if mod.phi:
                        self.phi += mod.phi
                        self.c = rect(self.c.real, mod.phi)
                    if mod.magnitude:
                        self.magnitude += mod.magnitude
                        self.c = rect(self.magnitude, phase(self.c))
                    if mod.offset:
                        self.offset += mod.offset
                if mod.clip is not None:
                    self.clip = mod.clip

                mod.duration_left -= 1

                if mod.duration_left == 0:
                    drop_modifier = True

        if drop_modifier:
            self.modifiers = {x for x in self.modifiers if x.duration_left != 0}

    def _wave_description(self) -> List[Tuple]:
        """
        Returns a list of tuples. Each tuple describes a frequency
        """
        wave_description = [(self.frequency, self.phi, self.magnitude,
                             self.offset, [], self.t_modifier)]
        return wave_description

    def calculate_frame(self, t: int) -> float:
        """
        Calculate one frame without self.offset
        """
        self.c *= self.c_t
        self._modify(t)
        return self.c.real

    def play(self, in_bytes: bool = True) -> Union[int, bytes]:
        """
        Returns a wav data frame as int or bytes
        """
        # Formula x = a*sin(w(t)+p) * scaling + offset
        frame = int(self.calculate_frame(self.t)
                    * 32767 + (32767 * self.offset))
        self.t += 1  # TODO reset self.t to keep small to save memory
        b = _limit(frame, self.clip).to_bytes(2, byteorder='little', signed=True)
        return b if in_bytes else frame


class Wave(object):
    @staticmethod
    def _sort_wave_description(wave_description: Tuple) -> List:
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
                if magnitude == 0:
                    logging.info(f'{frequency=}Hz with magnitude of 0 dropped')
                    continue
                offset = description[3]
                modifiers = description[4] if len(description) >= 5 else None
                t_modifier = description[5] if len(description) >= 6 else 1

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
    def from_desc(cls, frequency=0, phi=0, magnitude=1, offset=0,
                  t=None):
        return cls([(frequency, phi, magnitude, offset)], t)

    def _wave_description(self) -> List:
        """
        Returns a list of tuples. Each tuple describes a frequency
        :return: wave description
        :rtype: list
        """
        wave_description = []
        for wave in self.frequencies:
            wave_description += wave._wave_description()
        return wave_description

    def calculate_frame(self) -> float:
        frame = 0
        for basic_wave in self.frequencies:
            frame += basic_wave.calculate_frame(t=self.t)
        offset = 0
        for basic_wave in self.frequencies:
            offset += basic_wave.offset
        frame = frame * 32767 + offset * 32767
        self.t += 1
        return frame

    def play(self) -> bytes:
        """
        Returns a wav data frame
        :return: frame
        :rtype: bytes
        """
        frame = self.calculate_frame()
        return _limit(int(frame)).to_bytes(2, byteorder='little', signed=True)
