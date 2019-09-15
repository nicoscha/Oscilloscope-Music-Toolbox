from PIL import Image
import numpy
import math
import wave

from omt_utils import in_bytes


def _closed_neighbour(neighbour_list, starting_point):
    starting_point_x, starting_point_y = starting_point
    closed_neighbour = starting_point
    distance = math.inf
    for compare_point_x, compare_point_y in neighbour_list:
        temp_distance = (math.sqrt(math.pow(compare_point_x -
                                            starting_point_x, 2) +
                                   math.pow(compare_point_y -
                                            starting_point_y, 2)))
        if temp_distance == distance and temp_distance < 1.5:
            # TODO multiple pixels with same distance
            # closed neighbour should be the pixel in the same direction
            print(compare_point_x, compare_point_y, starting_point_x,
                  starting_point_y, closed_neighbour)
        if 0 < temp_distance < distance:
            closed_neighbour = (compare_point_x, compare_point_y)
            distance = temp_distance
    return closed_neighbour


def _order_black_px_list(black_px_list, starting_point=None):
    if starting_point not in black_px_list:
        starting_point = None
        print(f'starting point {starting_point} not in black_px_list')
    if starting_point is None:
        starting_point = black_px_list[0]
    sorted_px_list = [starting_point]
    black_px_list.remove(starting_point)
    unvisited_px_list = black_px_list

    len_list = len(unvisited_px_list)
    for i in range(len_list):
        closed_neighbour = _closed_neighbour(unvisited_px_list,
                                             starting_point)
        # print(sorted_px_list, unvisited_px_list, closed_neighbour,
        #      starting_point)
        sorted_px_list.append(closed_neighbour)
        unvisited_px_list.remove(closed_neighbour)
        starting_point = closed_neighbour
    return sorted_px_list


def write_black_px_list_to_wav(path, black_px_list):
    CHANNELS = 2
    SAMPLEWIDTH = 2
    FRAMERATE = 48000
    NFRAMES = 1 * FRAMERATE
    with wave.open(path.replace('.bmp', '.wav'), 'wb') as wav:
        wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                       'NONE', 'not compressed'))
        for i in range(10*len(black_px_list)):
            for x, y in black_px_list:
                wav.writeframes(in_bytes(x + 32760))
                wav.writeframes(in_bytes(y + 32760))


def _max_min_x_y_in_px_list(px_list):
    max_x_p, max_y_p = px_list[0]
    min_x_p, min_y_p = px_list[0]
    for x, y in px_list:
        if x > max_x_p:
            max_x_p = x
        if y > max_y_p:
            max_y_p = y
        if x < min_x_p:
            min_x_p = x
        if y < min_y_p:
            min_y_p = y
    return max_x_p, max_y_p, min_x_p, min_y_p


def _centered(black_px_list):
    max_x_p, max_y_p, min_x_p, min_y_p = _max_min_x_y_in_px_list(black_px_list)
    if max_x_p > 0 and min_x_p > 0:
        size_x = max_x_p - min_x_p
    else:  # Both negative
        size_x = min_x_p - max_x_p
    if max_y_p > 0 and min_y_p > 0:
        size_y = max_y_p - min_y_p
    else:  # Both negative
        size_y = min_y_p - max_y_p

    centered_black_px_list = []
    for x, y in black_px_list:
        dist_to_old_max_x = max_x_p - x
        if min_x_p > 0 and max_x_p > 0:  # All points positive
            new_x = size_x / 2 - dist_to_old_max_x
        elif min_x_p < 0 < max_x_p:  # Points positive and negative
            new_x = -(size_x / 2 - (- dist_to_old_max_x))
        else:  # All points negative
            new_x = size_x / 2 - (- dist_to_old_max_x)

        dist_to_old_max_y = max_y_p - y
        if min_y_p > 0 and max_y_p > 0:  # All points positive
            new_y = size_y / 2 - dist_to_old_max_y
        elif min_y_p < 0 < max_y_p:  # Points positive and negative
            new_y = -(size_y / 2 - (- dist_to_old_max_y))
        else:  # All points negative
            new_y = size_y / 2 - (- dist_to_old_max_y)

        centered_black_px_list.append((new_x, new_y))
    return centered_black_px_list


def _borders(px):
    y_range = 400  # Rows to be scanned
    x_range = 400  # Rows to be scanned
    for x in range(x_range):
        try:
            px[x,1]  # Access obj to find border
        except IndexError:
            x_range = x - 1
            break
    for y in range(y_range):
        try:
            px[1,y]  # Access obj to find border
        except IndexError:
            y_range = y - 1
            break
    return x, y


def main(path, starting_point=None):
    if path[-4:] != '.bmp':
        raise ValueError('File is not a .bmp')
    img = Image.open(path)
    px = img.load()
    y_range, x_range = _borders(px)
    black_px_list = []
    for x in range(x_range):
        for y in range(y_range):
            if px[x,y] == 0:
                black_px_list.append((x, y))
    black_px_list = _order_black_px_list(black_px_list, starting_point)
    black_px_list = black_px_list + list(reversed(black_px_list))
    black_px_list = _centered(black_px_list)
    write_black_px_list_to_wav(path, black_px_list)
    #print(black_px_list)


if __name__ == '__main__':
    main('test_bmp.bmp', (1, 49))
