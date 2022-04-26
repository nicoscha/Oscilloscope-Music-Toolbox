from collections import deque
from typing import List, Union
import matplotlib.image as mat_img
import random
import math
import uuid
import numpy as np


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


class CrossPoint:
    def __init__(self):
        self.ways = []
        self.pos = (-1, -1)
        self.block = (-1, -1, -1, -1)

    def set_block(self, y1, x1, y2, x2):
        self.block = (y1, x1, y2, x2)

    def add_way(self, way: deque):
        self.ways.append(way)

    def __repr__(self):
        return f"\nways length: {[len(w) for w in self.ways]} \npos:{self.pos} \nblock: {self.block}"


def find_crosspoints(image, crosspoints):
    image_size = len(image)
    grid_size = 16
    for i_x in range(0, image_size, grid_size):
        for i_y in range(0, image_size, grid_size):
            # loop intern grid
            count = 0
            r = random.randint(3, 25) * 10
            for ii_x in range(grid_size):
                for ii_y in range(grid_size):
                    if image[i_y + ii_y, i_x + ii_x]:
                        count += 1
            if count >= grid_size + 2:
                no_crosspoint = False
                t_count = 0
                for ii_x in range(grid_size - 2):
                    for ii_y in range(grid_size - 2):
                        if image[i_y + 1 + ii_y, i_x + 1 + ii_x]:
                            t_count += 1
                if count - t_count <= 2:
                    no_crosspoint = True
                if not no_crosspoint:
                    # paint cross point
                    for ii_x in range(grid_size):
                        for ii_y in range(grid_size):
                            pass  # orig[i_y+ii_y, i_x+ii_x] = 150 # r
                    # add crosspoint
                    cp = CrossPoint()
                    cp.set_block(i_y, i_x, i_y + grid_size, i_x + grid_size)
                    crosspoints[uuid.uuid4()] = cp


def expanding_grid_search(image, start_size, end_size, start_y, start_x, stuck, way, path_color, path_found, image_size):
    t_start_y = start_y
    t_start_x = start_x
    ##print(t_start_y, t_start_x)
    stuck = False
    b = False
    for grid_size in range(start_size, end_size + 1, 2):
        distances = np.array([np.zeros(grid_size) for _ in range(grid_size)])
        grid_index = grid_size // 2
        # Calculate distances to unused paths
        for i_x in range(-grid_index, grid_index + 1):
            for i_y in range(-grid_index, grid_index + 1):
                try:
                    if i_x - grid_index == 0 and i_y - grid_index == 0:
                        distances[i_x, i_y] = -8  # High distance to own pos
                    else:
                        tile_color = image[t_start_y + i_y, t_start_x + i_x]
                        if tile_color == path_color:  # path_color guarantees path hasn't been used jet
                            euc_dist = math.sqrt(abs(i_y) ** 2 + abs(i_x) ** 2)
                            dist = round(euc_dist, 2)
                            distances[i_x + grid_index, i_y + grid_index] = dist
                            image[t_start_y + i_y, t_start_x + i_x] += 1
                            ##print(i_x + grid_index, i_y + grid_index, round(euc_dist, 2))
                        else:
                            distances[i_x + grid_index, i_y + grid_index] = -1  # No path pixel or already visted
                except IndexError:
                    distances[i_x + grid_index, i_y + grid_index] = -9  # High distance to pos over the edge
                    # stuck = False
        ##print(distances)

        # Find closes unused path
        distances_flatt = distances.flatten()
        # print(distances_flatt)
        indexes = np.argsort(distances_flatt)
        # print(indexes)
        indexes = [_ for _ in indexes if distances_flatt[_] > 0]  # Filter for unused paths
        ##print(indexes)
        if len(indexes) >= 1:
            # TODO random
            next_path_index = indexes[0]#random.randint(0, min(len(indexes) - 1, 1))]
            t_x = next_path_index % grid_size
            t_y = next_path_index // grid_size
            ##print(next_path_index, 't_y', t_y, 't_x', t_x)
            image[t_start_y + t_x - grid_index, t_start_x + t_y - grid_index] = path_found
            t_start_y = t_start_y + t_x - grid_index
            t_start_x = t_start_x + t_y - grid_index
            assert t_start_x < (image_size + 1)
            assert t_start_y < (image_size + 1)
            way.append((t_start_y, t_start_x))
            stuck = False

            break
        else:  # Check stuck
            if grid_size >= end_size:
                stuck = True
                #image[t_start_y + grid_index, t_start_x + grid_index] = 200 + grid_index
    return t_start_y, t_start_x, stuck, way


def span_path(image, cp_uu, crosspoints, path_color, path_found, debug=False):
    image_size = len(image)
    cp = crosspoints[cp_uu]
    b = False
    # print(cp_uu)
    block_size = 16
    start_x = None
    start_y = None
    # find start
    for i_x in range(block_size):
        for i_y in range(block_size):
            if image[cp.block[0] + i_y, cp.block[1] + i_x] == path_color:
                start_y = cp.block[0] + i_y
                start_x = cp.block[1] + i_x
                image[cp.block[0] + i_y, cp.block[1] + i_x] = path_found
                b = True
                break
        if b:
            break
    if not b:  # No start found, just return.
        return None

    # find path
    way = deque()
    for __ in range(2):  # wege von cosspoint TODO
        stuck = False
        b = False
        t_start_x = start_x
        t_start_y = start_y

        for _ in range(25000):
            # print('new')
            if True:
                lower_grid_size = 3  # 3 or above
                upper_grid_size = 27
                # for gs in range(3, 11+1, 2):
                t_start_y, t_start_x, stuck, way = expanding_grid_search(image,
                    lower_grid_size, upper_grid_size, t_start_y, t_start_x,
                    stuck, way, path_color, path_found, image_size)

                # Got stuck
                if stuck:
                    if len(way) > 0:
                        #crosspoints[cp_uu].add_way(way)
                        pass
                    if debug:
                        image[t_start_y, t_start_x] = 250
                    b = True
                    break
            if b:  ### TODO ändern für mehre wege
                break
        way.reverse()
    print('EOL', len(way))
    if len(way) > 5:
        crosspoints[cp_uu].add_way(way)


def find_closest_points_1(path_1: list, path_2: list) -> Union[tuple[tuple[int, int], tuple[int, int]], None]:
    closest_euclidean_distance = 9_999_999
    closet_point_path_1: tuple = (None, None)
    closet_point_path_2: tuple = (None, None)
    for y_p_1, x_p_1 in path_1:
        for y_p_2, x_p_2 in path_2:
            euclidean_distance = math.sqrt(abs(y_p_1 - y_p_2) ** 2
                                           + abs(x_p_1 - x_p_2) ** 2)
            if euclidean_distance < closest_euclidean_distance:  # TODO could be multiple with same/similar values but better angel
                closest_euclidean_distance = euclidean_distance
                closet_point_path_1 = (y_p_1, x_p_1)
                closet_point_path_2 = (y_p_2, x_p_2)

    if closet_point_path_1 and closet_point_path_2:
        return closet_point_path_1, closet_point_path_2
    else:
        return None


def find_closest_points(point: list, paths: list) -> tuple[int, list]:
    closest_euclidean_distance = 9_999_999
    y_p_1, x_p_1 = point

    for i, path in enumerate(paths):
        # Test first element
        y_p_2, x_p_2 = path[1]
        euclidean_distance = math.sqrt(abs(y_p_1 - y_p_2) ** 2
                                       + abs(x_p_1 - x_p_2) ** 2)
        if euclidean_distance < closest_euclidean_distance:  # TODO could be multiple with same/similar values but better angel
            closest_euclidean_distance = euclidean_distance
            path_to_join_with = path
            best_index = i

        # Test last element
        y_p_2, x_p_2 = path[-1]
        euclidean_distance = math.sqrt(abs(y_p_1 - y_p_2) ** 2
                                       + abs(x_p_1 - x_p_2) ** 2)
        if euclidean_distance < closest_euclidean_distance:  # TODO could be multiple with same/similar values but better angel
            closest_euclidean_distance = euclidean_distance
            path_to_join_with = list(reversed(path))
            best_index = i

    return best_index, path_to_join_with


def join_close_paths_(main_path: list, path: list, main_join_point, join_point) -> list[tuple[int, int]]:
    new_path = []
    for point in main_path:
        if point[0] == main_join_point[0] and point[1] == main_join_point[1]:
            buffer = []
            for point_2 in path:
                if point_2[0] == join_point[0] and point_2[1] == join_point[1]:
                    # From point to end
                    new_path.append(path[len(buffer):])
                    # From end back to point_2
                    new_path.append(reversed(path[len(buffer):]))
                    # From point_2 to top
                    new_path.append(reversed(buffer))
                    # From top to point_2
                    new_path.append(buffer)
                else:
                    buffer.append(point_2)
        else:
            new_path.append(point)
    return new_path


def transform_to_wav_cord(way: List, image_size: int) -> tuple[List, List]:
    half_image_size = int(image_size / 2)
    t_x = []
    t_y = []
    for (y, x) in way:  # y and x are swapped in the image array!
        t_x.append((x - half_image_size) / half_image_size)
        t_y.append((y - half_image_size) / half_image_size)
    if max(np.abs(t_y)) > 1 or max(np.abs(t_y)) > 1:
        raise ValueError
    return t_x, t_y


def join(crosspoints):
    ways = [_cp.ways[0] for _cp in crosspoints.values() if len(_cp.ways) > 0]

    if len(ways) < 2:
        return ways[0]
    elif len(ways) == 2:
        return ways[0] + ways[1]  # TODO join by distance

    joined_path = ways[0]
    leftover_paths = ways[1:]
    while len(leftover_paths) > 0:
        index_best_path, best_path = find_closest_points(joined_path[-1],
                                                         leftover_paths)
        joined_path += best_path
        leftover_paths.remove(leftover_paths[index_best_path])
    return joined_path


def trim_convolved(filter_length, samples):
    return samples[filter_length-1:-filter_length-2]


def convolve(_filter: np.array, x: np.array, y: np.array) -> tuple[np.array, np.array]:
    """Convolve x and y with _filter and trim edges to original length"""
    filter_length = len(_filter)
    convolved_x = np.convolve(x, _filter)
    m_x = list(trim_convolved(filter_length, convolved_x))
    convolved_y = np.convolve(y, _filter)
    m_y = list(trim_convolved(filter_length, convolved_y))
    return m_x, m_y


def average_filter(x: list, y: list, filter_length: int = 14):
    if filter_length < 3:
        filter_length = 3
    avg_filter = np.divide(np.ones(filter_length), filter_length)

    m_x, m_y = convolve(avg_filter, x, y)
    return m_x, m_y


def binominal_filter(x: list, y: list, filter_length: int = 32):
    filter = (np.poly1d([0.5, 0.5]) ** (filter_length-1)).coeffs
    m_x, m_y = convolve(filter, x, y)
    return m_x, m_y


def mirror_wav(_m_x, _m_y):
    _m_x += reversed(_m_x)
    _m_y += reversed(_m_y)
    return _m_x, _m_y


def repeat(m_x, m_y, seconds, sample_rate: int = 48000):
    _m_x, _m_y = m_x, m_y
    samples = seconds * sample_rate
    repeats = int(samples / len(m_x)) - 1
    for _ in range(repeats):
        _m_x += m_x
        _m_y += m_y
    return m_x, m_y


def convert_image_to_audio(file_name: str) -> tuple[List, List]:
    path_color = 27
    path_found = path_color + 48
    crosspoints = dict()
    debug = False

    # Load image and drop redundant data
    image = mat_img.imread(file_name)
    image = image[:, :, 0]
    # Reduce color
    image_size = len(image)
    for i_x in range(image_size):
        for i_y in range(image_size):
            if image[i_y, i_x] > 0:
                image[i_y, i_x] = path_color

    # Process
    find_crosspoints(image, crosspoints)
    if crosspoints.keys():
        pass
    else:
        b = False
        for i_x in range(image_size):
            for i_y in range(image_size):
                if image[i_y, i_x] > 0:
                    cp = CrossPoint()
                    cp.set_block(i_y, i_x, i_y + 16, i_x + 16)
                    crosspoints[uuid.uuid4()] = cp
                    b = True
                    break
            if b:
                break
    for cp_uuid in crosspoints.keys():
        span_path(image, cp_uuid, crosspoints, path_color, path_found, debug=debug)

    joined_path = join(crosspoints)
    x, y = transform_to_wav_cord(joined_path, image_size)

    # print()
    # print([v for (k, v) in crosspoints.items() if len(v.ways) > 0])
    import matplotlib.pyplot as plt
    # plt.imshow(image, cmap='gray')
    # plt.show()

    return x, y


def gen_low_pass(k, vg):
    t_bk = [0] * k
    for k_i in range(k):
        if k_i == 0:
            t_bk[0] = 2 * vg
        else:
            t_bk[k_i] = 2 * ((np.sin(2 * np.pi * k_i * vg)) / (2 * np.pi * k_i *vg))
    bk = t_bk[:0:-1] + t_bk
    return bk


def scale_audio_to_fps(x_samples: np.array, y_samples: np.array, fps: int = 30, sample_rate: int = 48000) -> tuple[np.array, np.array]:
    samples_per_frame = int(sample_rate / fps)
    x_target_samples = np.zeros(samples_per_frame)
    y_target_samples = np.zeros(samples_per_frame)
    if len(x_samples) < samples_per_frame:  # Up sampling
        print('up')
        if False:
            """
            factor = samples_per_frame / len(x_samples)
            t_x = []
            t_y = []
            for i in range(int(factor)):
                if i % 2 == 0:
                    t_x.extend(x_samples)
                    t_y.extend(y_samples)
                else:
                    t_x.extend(x_samples[::-1])
                    t_y.extend(y_samples[::-1])
            #if len(t_x) < samples_per_frame:
            #    t_x[len(t_x):] = x_samples[:samples_per_frame - len(t_x)]
            #    t_y[len(t_y):] = y_samples[:samples_per_frame - len(t_y)]
            x_target_samples = t_x
            y_target_samples = t_y
            """
            #"""
            factor = samples_per_frame / len(x_samples)
            for i in range(1, samples_per_frame):
                try:
                    _i = int(round(i / factor))
                    x_target_samples[i] = x_samples[_i]
                    y_target_samples[i] = y_samples[_i]
                except IndexError as E:
                    print('IndexError at i:', i, '_i:', _i)
                    break


            k = 21
            lp_filter = gen_low_pass(k, 0.25)
            '''
            for i in range(8):
                x_target_samples = np.convolve(x_target_samples, lp_filter)
                y_target_samples = np.convolve(y_target_samples, lp_filter)
            for i in range(8):
                x_target_samples = trim_convolved(k, x_target_samples)
                y_target_samples = trim_convolved(k, y_target_samples)
            '''
            for i in range(2):
                x_target_samples, y_target_samples = average_filter(x_target_samples, y_target_samples, k)

        x_target_samples = np.interp(x_target_samples,
                                     np.linspace(0, len(x_samples)),
                                     x_samples)
        y_target_samples = np.interp(x_target_samples,
                                     np.linspace(0, len(x_samples)),
                                     y_samples)
        scale_factor = max(np.max(np.abs(x_target_samples)),
                           np.max(np.abs(y_target_samples)))
        x_target_samples = np.divide(x_target_samples, scale_factor)
        y_target_samples = np.divide(y_target_samples, scale_factor)
        #"""

    elif len(x_samples) > samples_per_frame:  # Down sampling
        factor = 1 / (samples_per_frame / len(x_samples)) + 1
        k = 21

        lp_filter = gen_low_pass(k, 0.01)
        x_filtered_samples = np.convolve(x_samples, lp_filter)
        y_filtered_samples = np.convolve(y_samples, lp_filter)
        x_trimmed_samples = x_samples#trim_convolved(k, x_filtered_samples)
        y_trimmed_samples = y_samples#trim_convolved(k, y_filtered_samples)

        x_target_samples = x_trimmed_samples[::int(factor)]
        y_target_samples = y_trimmed_samples[::int(factor)]

        scale_factor = max(np.max(np.abs(x_target_samples)),
                           np.max(np.abs(y_target_samples)))
        x_target_samples = np.divide(x_target_samples, scale_factor)
        y_target_samples = np.divide(y_target_samples, scale_factor)
    else:
        x_target_samples = x_samples
        y_target_samples = y_samples
    return x_target_samples, y_target_samples
