import unittest
import btm_to_array


class Centered(unittest.TestCase):
    def test_max_min_x_y_in_positive_px_list(self):
        px_list = [(50, 40), (50, 80), (80, 90), (80, 40), (70, 50)]
        (max_x_p, max_y_p,
         min_x_p, min_y_p) = btm_to_array._max_min_x_y_in_px_list(px_list)
        self.assertEqual(80, max_x_p)
        self.assertEqual(90, max_y_p)
        self.assertEqual(50, min_x_p)
        self.assertEqual(40, min_y_p)

    def test_max_min_x_y_in_negative_px_list(self):
        px_list = [(-50, -40), (-50, -80), (-80, -90), (-80, -40), (-70, -50)]
        (max_x_p, max_y_p,
         min_x_p, min_y_p) = btm_to_array._max_min_x_y_in_px_list(px_list)
        self.assertEqual(-50, max_x_p)
        self.assertEqual(-40, max_y_p)
        self.assertEqual(-80, min_x_p)
        self.assertEqual(-90, min_y_p)

    def test_centered_positive_x_y(self):
        px_list = [(50, 40), (50, 80), (80, 80), (80, 40), (70, 50)]
        centered_list = [(-15.0, -20.0), (-15.0, 20.0), (15.0, 20.0),
                         (15.0, -20.0), (5, -10.0)]
        self.assertEqual(centered_list, btm_to_array._centered(px_list))

    def test_centered_positive_x_negative_y(self):
        px_list = [(50, -40), (50, -80), (80, -80), (80, -40), (70, -50)]
        centered_list = [(-15.0, -20.0), (-15.0, 20.0), (15.0, 20.0),
                         (15.0, -20.0), (5, -10.0)]
        self.assertEqual(centered_list, btm_to_array._centered(px_list))

    def test_centered_negative_x_positive_y(self):
        px_list = [(-50, 40), (-50, 80), (-80, 80), (-80, 40), (-70, 50)]
        centered_list = [(-15.0, -20.0), (-15.0, 20.0), (15.0, 20.0),
                         (15.0, -20.0), (5, -10.0)]
        self.assertEqual(centered_list, btm_to_array._centered(px_list))

    def test_centered_negative_x_y(self):
        px_list = [(-50, -40), (-50, -80), (-80, -80), (-80, -40), (-70, -50)]
        centered_list = [(-15.0, -20.0), (-15.0, 20.0), (15.0, 20.0),
                         (15.0, -20.0), (5, -10.0)]
        self.assertEqual(centered_list, btm_to_array._centered(px_list))

    def test_centered_positive_and_negative_x_y(self):
        px_list = [(-16, -21), (-16, 19), (14, 19), (14, -21), (4, -11)]
        centered_list = [(-15.0, -20.0), (-15.0, 20.0), (15.0, 20.0),
                         (15.0, -20.0), (5, -10.0)]
        self.assertEqual(centered_list, btm_to_array._centered(px_list))


if __name__ == '__main__':
    unittest.main()
