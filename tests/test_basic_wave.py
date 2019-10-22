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

    def test_init_magnitude_ValueError(self):
        with self.assertRaises(ValueError):
            a = basic_wave.BasicWave(frequency=440, magnitude='Not a int/float')

    def test_init_magnitude_value(self):
        self.assertEqual(0.749, basic_wave.BasicWave(440,
                                                     magnitude=0.749).magnitude)

    def test_init_magnitude_value_range(self):
        self.assertEqual(1.0, basic_wave.BasicWave(440,
                                                   magnitude=1.5).magnitude)
        self.assertEqual(-1.0, basic_wave.BasicWave(440,
                                                    magnitude=-5.1).magnitude)

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
        a_w = basic_wave.BasicWave(400)
        b_w = basic_wave.BasicWave(493.88)
        x_w = basic_wave.Wave([(770, 0.0, 1.0, 0.0)])
        self.assertIsInstance(a_w + b_w , basic_wave.Wave)
        self.assertIsInstance(a_w + x_w, basic_wave.Wave)

    def test_play_in_bytes(self):
        a_w = basic_wave.BasicWave(400)
        self.assertIsInstance(a_w.play(), bytes)
        a_w = basic_wave.BasicWave(400)
        self.assertIsInstance(a_w.play(in_bytes=False), float)


if __name__ == '__main__':
    unittest.main()
