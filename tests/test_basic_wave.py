import unittest
import basic_wave
from math import pi, tau, sin


class Limit(unittest.TestCase):
    def test_limit(self):
        self.assertEqual(32767, basic_wave._limit(4815162342))
        self.assertEqual(-32767, basic_wave._limit(-4815162342))
        self.assertEqual(440, basic_wave._limit(440))


class BasicWave(unittest.TestCase):
    def test_init_frequency_ValueError(self):
        with self.assertRaises(ValueError):
            a = basic_wave.BasicWave('Not a int/float')

    def test_init_frequency_value(self):
        self.assertEqual(440, basic_wave.BasicWave(440).frequency)

    def test_init_phi_ValueError(self):
        with self.assertRaises(ValueError):
            a = basic_wave.BasicWave(frequency=440, phi='Not a int/float')

    def test_init_phi_value(self):
        self.assertEqual(pi, basic_wave.BasicWave(440, phi=pi).phi)

    def test_init_magnitude_ValueError(self):
        with self.assertRaises(ValueError):
            a = basic_wave.BasicWave(frequency=440, magnitude='Not a int/float')

    def test_init_magnitude_value(self):
        self.assertEqual(0.749, basic_wave.BasicWave(440,
                                                     magnitude=0.749).magnitude)

    def test_init_magnitude_value_range(self):
        self.assertEqual(1.0, basic_wave.BasicWave(440,
                                                   magnitude=2.5).magnitude)
        self.assertEqual(-1.0, basic_wave.BasicWave(440,
                                                    magnitude=-5.2).magnitude)

    def test_init_offset_ValueError(self):
        with self.assertRaises(ValueError):
            a = basic_wave.BasicWave(frequency=440, offset='Not a int/float')

    def test_init_offset_value(self):
        self.assertEqual(0.749, basic_wave.BasicWave(440,
                                                     offset=0.749).offset)
        self.assertEqual(-0.361, basic_wave.BasicWave(440,
                                                      offset=-0.361).offset)

    def test_init_offset_value_range(self):
        self.assertEqual(0.0, basic_wave.BasicWave(440,
                                                   offset=1.5).offset)
        self.assertEqual(0.0, basic_wave.BasicWave(440,
                                                   offset=-5.1).offset)

    def test_add_return(self):
        a_w = basic_wave.BasicWave(440)
        b_w = basic_wave.BasicWave(493.88)
        x_w = basic_wave.Wave([(770, 0.0, 1.0, 0.0)])
        self.assertIsInstance(a_w + b_w , basic_wave.Wave)
        self.assertIsInstance(a_w + x_w, basic_wave.Wave)

    def test_play_in_bytes(self):
        a_w = basic_wave.BasicWave(440)
        self.assertIsInstance(a_w.play(), bytes)
        a_w = basic_wave.BasicWave(440)
        self.assertIsInstance(a_w.play(in_bytes=False), float)

    def test_wave_description(self):
        a_w_description = (440, pi/2, 0.63, 0.87)
        frequency = a_w_description[0]
        phi = a_w_description[1]
        magnitude = a_w_description[2]
        offset = a_w_description[3]
        a_w = basic_wave.BasicWave(frequency, phi=phi, magnitude=magnitude,
                                   offset=offset)

        self.assertEqual([a_w_description], a_w._wave_description())

    def test_calculate_frame(self):
        a_w_description = (440, pi / 2, 0.63, 0.87)
        frequency = a_w_description[0]
        phi = a_w_description[1]
        magnitude = a_w_description[2]
        offset = a_w_description[3]
        a_w = basic_wave.BasicWave(frequency, phi=phi, magnitude=magnitude,
                                   offset=offset)

        for t in range(1, int(basic_wave.FRAMERATE/frequency+1)):
            frame = (magnitude
                     * sin(((tau * frequency / basic_wave.FRAMERATE) * t) + phi)
                     + (32767.0 * offset))
            self.assertEqual(frame, a_w.calculate_frame())

    def test_play(self):
        a_w_description = (440, pi / 2, 0.63, 0.87)
        frequency = a_w_description[0]
        phi = a_w_description[1]
        magnitude = a_w_description[2]
        offset = a_w_description[3]

        # As integer
        a_w = basic_wave.BasicWave(frequency, phi=phi, magnitude=magnitude,
                                   offset=offset)
        for t in range(1, int(basic_wave.FRAMERATE / frequency + 1)):
            frame = (magnitude
                     * sin(((tau * frequency / basic_wave.FRAMERATE) * t) + phi)
                     * 32767.0 + (32767.0 * offset))
            self.assertEqual(frame, a_w.play(in_bytes=False))

        # As bytes
        a_w = basic_wave.BasicWave(frequency, phi=phi, magnitude=magnitude,
                                   offset=offset)
        for t in range(1, int(basic_wave.FRAMERATE / frequency + 1)):
            frame = (magnitude
                     * sin(((tau * frequency / basic_wave.FRAMERATE) * t) + phi)
                     * 32767.0 + (32767.0 * offset))
            frame = basic_wave._limit(int(frame)).to_bytes(2,
                                                           byteorder='little',
                                                           signed=True)
            self.assertEqual(frame, a_w.play(in_bytes=True))


if __name__ == '__main__':
    unittest.main()
