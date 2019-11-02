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
                     )
            self.assertEqual(frame, a_w.calculate_frame(t))

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


class Wave(unittest.TestCase):
    def test_sort_wave_description(self):
        expected_list = [(880, 0.0, 0.0, 0.0),(440, 0.0, 0.0, 0.0)]
        sorted_list = basic_wave.Wave._sort_wave_description(
            [(440, 0.0, 0.0, 0.0),(880, 0.0, 0.0, 0.0)])
        self.assertEqual(expected_list, sorted_list)

    def test_init_wav_desc_not_a_list_NotImplemented(self):
        with self.assertRaises(NotImplementedError):
            basic_wave.Wave('No wave desc')

    def test_init_wav_desc_does_not_contain_tuble_NotImplemented(self):
        with self.assertRaises(NotImplementedError):
            basic_wave.Wave(['No wave desc'])

    def test_init_wav_desc(self):
        ab_desc = [(493.88, 0.3, 0.5, 0.1), (440, 2.0, 0.3, 0.2)]
        ab = basic_wave.Wave(ab_desc)
        # Ignore order of frequencies
        self.assertIn(*ab.frequencies[0]._wave_description(), ab_desc)
        self.assertIn(*ab.frequencies[1]._wave_description(), ab_desc)

    def test_init_t_value(self):
        ab_desc = [(493.88, 0.3, 0.5, 0.1), (440, 2.0, 0.3, 0.2)]
        ab = basic_wave.Wave(ab_desc, t=100)
        self.assertEqual(ab.t, 100)
        ab = basic_wave.Wave(ab_desc)
        self.assertEqual(1, ab.t)

    def test_add_t_value(self):
        a_desc = [(440, 2.0, 0.3, 0.2)]
        a = basic_wave.Wave(a_desc, t=100)
        b_desc = [(493.88, 0.3, 0.5, 0.1)]
        b = basic_wave.Wave(b_desc)
        ab = a + b
        self.assertEqual(100, ab.t)
        ba = b + a
        self.assertEqual(1, ba.t)
        a += b
        self.assertEqual(100, a.t)

    def test_wave_description(self):
        ab_desc = [(493.88, 0.0, 0.5, 0.0), (440, 0.0, 0.7, 0.0)]
        ab = basic_wave.Wave(ab_desc)
        self.assertEqual(ab._wave_description(), ab_desc)

    def test_calculate(self):
        ab_desc = [(493.88, 0.3, 0.5, 0.1), (440, 2.0, 0.3, 0.2)]
        ab = basic_wave.Wave(ab_desc)
        ab_copy = basic_wave.Wave(ab_desc)

        for t in range(1, int(basic_wave.FRAMERATE / ab_desc[0][0] + 1)):
            frame = 0
            for b_wave in ab_copy.frequencies:
                frame += b_wave.calculate_frame(t)
            offset = 0.0
            for b_wave in ab_copy.frequencies:
                offset += b_wave.offset
            expected_frame = frame * 32767.0 + offset * 32767.0
            self.assertEqual(ab.calculate_frame(), expected_frame)

    def test_play(self):
        ab_desc = [(493.88, 0.3, 0.5, 0.1), (440, 2.0, 0.3, 0.2)]
        ab = basic_wave.Wave(ab_desc)
        ab_copy = basic_wave.Wave(ab_desc)

        for t in range(1, int(basic_wave.FRAMERATE / ab_desc[0][0] + 1)):
            expected_play_frame = ab_copy.calculate_frame()
            epf_bytes = basic_wave._limit(int(expected_play_frame)
                                          ).to_bytes(2, byteorder='little',
                                                     signed=True)
            self.assertEqual(ab.play(), epf_bytes)


if __name__ == '__main__':
    unittest.main()
