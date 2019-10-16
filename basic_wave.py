import logging
from math import sin, tau

logging.basicConfig(level=logging.DEBUG)
FRAMERATE = 48000


class BasicWave(object):
    def __init__(self, frequency, phi=0.0, amplitude=1.0, offset=0.0):
        """

        :param frequency:
        :param phi: sin = phi=0.0; cos = phi=math.pi/2
        :param amplitude:
        :param offset: DC offset
        """
        logging.debug(f'Creating BasicWave f={frequency}, p={phi}, '
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
        self.t = 0
        logging.debug(f'Created BasicWave f={self.frequency}, p={self.phi}, '
                      f'a={self.amplitude}, offset={self.offset}')

    def __str__(self):
        return (f'f={self.frequency}Hz, p={self.phi}Phi, '
                f'a={self.amplitude}, offset={self.offset}')

    def __add__(self, other):
        return Wave(wave_description=[(self.frequency, self.phi,
                                       self.amplitude, self.offset),
                                      (other.frequency, other.phi,
                                       other.amplitude, other.offset)])

    def play(self, in_bytes=True):
        """
        Returns a wav data frame
        :param in_bytes:
        :return:
        """
        # Formula x = a*sin(w(t)+p) * scaling + offset
        frame = (self.amplitude
             * sin(((tau * self.frequency / FRAMERATE) * self.t) + self.phi)
             * 125.0 + (125.0 * self.offset))

        self.t += 1
        if self.t > FRAMERATE:  # Keep self.t small to save memory
            self.t = 0
        b = int(frame).to_bytes(2, byteorder='big', signed=True)
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
        logging.debug(f'Created Wave wave_description={self}')

    def __str__(self):
        s = ''
        for f, p, a, o in self._wave_description():
            s += f'f={f}Hz, p={p}Phi, a={a}, offset={o}; '
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
            wave_description.append((wave.frequency, wave.phi, wave.amplitude,
                                     wave.offset))
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
