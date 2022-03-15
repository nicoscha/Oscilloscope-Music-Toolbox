from typing import List
import matplotlib.image as mat_img


def convert_audio_to_image(data: tuple[list, list], image_size: int = 640,
                           file_name: str = 'a_to_i.png') -> None:
    """
    :param data: 2 channel audio (100%=32767)
    :param image_size: Image edge length
    :param file_name: Str name and extension optional with path.
                      Empty file_name only returns but does not save image
    :return: Image
    """
    if len(data) != 2:
        raise ValueError(f'Stereo audio needed {len(data)} channel provided')

    image = [[0 for _ in range(image_size)] for _ in range(image_size)]
    scaling_factor = (image_size - 2) / 2
    offset = int(image_size / 2)
    _y, _x = data
    for (x, y) in zip(_x, _y):
        x_i = int(x * scaling_factor) + offset
        y_i = int(y * scaling_factor) + offset
        image[x_i][y_i] = 128
    if file_name:
        mat_img.imsave(file_name, image, cmap='gray')
