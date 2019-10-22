import logging
from math import sin, tau

logging.basicConfig(level=logging.DEBUG)
FRAMERATE = 48000


def _limit(signal):
    UPPER_XY_LIMIT = 32767
    LOWER_XY_LIMIT = -32767
    if signal > UPPER_XY_LIMIT:
        return UPPER_XY_LIMIT
    elif signal < LOWER_XY_LIMIT:
        return LOWER_XY_LIMIT
    return signal

class BasicWave(object):
    def __init__(self, frequency, phi=0.0, amplitude=1.0, offset=0.0):
        """

        :param frequency: Hz
        :param phi: sin = phi=0.0; cos = phi=math.pi/2
        :param amplitude:
        :param offset: DC offset
        """
        logging.debug(f'Creating BasicWave f={frequency}Hz, p={phi}Phi, '
                      f'a={amplitude}, offset={offset}')

        if type(frequency) is int or type(frequency) is float:
            self.frequency = frequency
        else:
            raise ValueError('Frequency must be int or float not '
                             'f{type(frequency)}')

        if type(phi) is int or type(phi) is float:
            self.phi = phi
        else:
            raise ValueError('Phi must be int or float not '
                             f'{type(phi)}={str(phi)}. Phi has been set to 0.0.'
                             )

        if type(amplitude) is int or type(amplitude) is float:
            if amplitude > 1.0 or amplitude < 0.0:
                logging.warning('Amplitude must be between '
                                f'0.0 and 1.0 is {amplitude}'
                                'amplitude has been set to 1.0.')
                self.amplitude = 1.0
            else:
                self.amplitude = amplitude
        else:
            raise ValueError('Amplitude must be int or float not '
                             f'{type(amplitude)}={str(amplitude)}. ')

        if type(offset) is int or type(offset) is float:
            if offset > 1.0 or offset < -1.0:
                logging.warning('Offset must be between '
                                f'-1.0 and 1.0 is {offset} '
                                'offset has been set to 1.0.')
                self.offset = 0.0
            else:
                self.offset = offset
        else:
            self.offset = 0.0
            raise ValueError('Offset must be int or float not '
                             f'{type(offset)}={str(offset)}. '
                             'offset has been set to 0.0. '
                             f'State: {self}')
        self.t = 1
        logging.debug(f'Created BasicWave {str(self)}')

    def __str__(self):
        for f, p, a, o in self._wave_description():
            return f'f={f}Hz, p={p}Phi, a={a}, offset={o}; '

    def __add__(self, other):
        if isinstance(other, BasicWave) or isinstance(other, Wave):
            return Wave(wave_description=(self._wave_description()
                                          + other._wave_description()))
        else:
            raise NotImplementedError

    def _wave_description(self):
        """
        Returns a list of tuples. Each tuple describes a frequency
        :return:
        """
        wave_description = [(self.frequency, self.phi, self.amplitude,
                             self.offset)]
        return wave_description

    def play(self, in_bytes=True):
        """
        Returns a wav data frame
        :param in_bytes:
        :return:
        """
        # Formula x = a*sin(w(t)+p) * scaling + offset
        frame = (self.amplitude
                 * sin(((tau * self.frequency / FRAMERATE) * self.t) + self.phi)
                 * 32767.0 + (32767.0 * self.offset))

        self.t += 1  # TODO reset self.t to keep small to save memory
        b = _limit(int(frame)).to_bytes(2, byteorder='little', signed=True)
        return b if in_bytes else frame


class Wave(BasicWave):
    def __init__(self, wave_description):
        """
        :param wave_description: tuple of f and p
        """
        logging.debug(f'Creating Wave wave_description={wave_description}')
        self.frequencies = []
        if type(wave_description) is list:
            if type(wave_description[0]) is tuple:
                for frequency, phi, amplitude, offset in wave_description:
                    self.frequencies.append(BasicWave(frequency=frequency,
                                                      phi=phi,
                                                      amplitude=amplitude,
                                                      offset=offset))
            elif type(wave_description[0] is BasicWave):
                for wave in wave_description:
                    self.frequencies.append(wave)
        logging.debug(f'Created Wave wave_description={str(self)}')

    def __str__(self):
        s = ''
        for wave in self.frequencies:
            s += str(wave)
        return s

    def __add__(self, other):
        return Wave(wave_description=(self._wave_description() +
                                      other._wave_description()))

    def _wave_description(self):
        """
        Returns a list of tuples. Each tuple describes a frequency
        :return:
        """
        wave_description = []
        for wave in self.frequencies:
            wave_description += wave._wave_description()
        return wave_description

    def play(self):
        """
        Returns a wav data frame
        :return:
        """
        frame = 0
        fraction_of_frequencies = 1/len(self.frequencies)
        for basic_wave in self.frequencies:
            frame += basic_wave.play(in_bytes=False) * fraction_of_frequencies
        return int(frame).to_bytes(2, byteorder='big', signed=True)
