import unittest
import basic_wave
from math import pi

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

    def test_init_amplitude_ValueError(self):
        with self.assertRaises(ValueError):
            a = basic_wave.BasicWave(frequency=440, amplitude='Not a int/float')

    def test_init_amplitude_value(self):
        self.assertEqual(0.749, basic_wave.BasicWave(440,
                                                     amplitude=0.749).amplitude)

    def test_init_amplitude_value_range(self):
        self.assertEqual(1.0, basic_wave.BasicWave(440,
                                                     amplitude=1.5).amplitude)
        self.assertEqual(1.0, basic_wave.BasicWave(440,
                                                   amplitude=-5.1).amplitude)

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


if __name__ == '__main__':
    unittest.main()
