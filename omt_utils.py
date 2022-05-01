from typing import Union, List
from math import pi
import numpy as np
import warnings
import wave


def _limit(signal):
    UPPER_XY_LIMIT = 32767
    LOWER_XY_LIMIT = -32767
    if signal > UPPER_XY_LIMIT:
        return UPPER_XY_LIMIT
    elif signal < LOWER_XY_LIMIT:
        return LOWER_XY_LIMIT
    return signal


def in_bytes(signal):
    return int(_limit(signal)).to_bytes(2, byteorder='little', signed=True)


def scale(signal: np.array, factor: float, to_int: bool = False) -> np.array:
    if to_int:
        return np.int16(np.multiply(signal, factor))
    else:
        return np.multiply(signal, factor)


def add(signal_1: np.array, signal_2: np.array, factor_1: float = 1.0, factor_2: float = 1.0, to_int: bool = False) -> np.array:
    if len(signal_1) != len(signal_2):
        warnings.warn('Signals have different length')
    if to_int:
        if factor_1 == 1.0 or factor_2 == 1.0:
            sig = np.int16(np.add(signal_1, signal_2))
        else:
            sig = np.int16(np.add(np.multiply(signal_1, factor_1), np.multiply(signal_2, factor_2)))
    else:
        if factor_1 == 1.0 or factor_2 == 1.0:
            sig = np.add(signal_1, signal_2)
        else:
            sig = np.add(np.multiply(signal_1, factor_1), np.multiply(signal_2, factor_2))
    return sig


def offset(signal: np.array, offset_value: float) -> np.array:
    return np.add(signal, offset_value)


def clip(signal:  np.array, limit: float) -> np.array:
    return np.clip(signal, -limit, limit)


def multiply(signal_1: np.array, signal_2: np.array, to_int: bool = False) -> np.array:
    if len(signal_1) != len(signal_2):
        warnings.warn('Signals have different length')
    if to_int:
        return np.int16(np.multiply(signal_1, signal_2))
    else:
        return np.multiply(signal_1, signal_2)


def gen_sin(frequency: int, sample_rate: int, duration: int) -> np.array:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: numpy.array of samples
    """
    samples = np.sin(np.multiply(2 * pi * frequency, np.divide(np.arange(0, duration), sample_rate)))
    return samples


def gen_cos(frequency: int, sample_rate: int, duration: int) -> np.array:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: numpy.array of samples
    """
    samples = np.cos(np.multiply(2 * pi * frequency, np.divide(np.arange(0, duration), sample_rate)))
    return samples


def gen_sawtooth(frequency: int, sample_rate: int, duration: int) -> np.array:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: numpy.array of samples
    """
    len_wave = (1 / frequency) * sample_rate
    samples = np.divide(np.mod(np.arange(0, duration), len_wave), len_wave)
    samples = np.subtract(np.multiply(samples, 2), 1)  # Scale from 0..1 to 1..-1
    return samples


def gen_triangle(frequency: int, sample_rate: int, duration: int) -> np.array:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: numpy.array of samples
    """
    range_samples = range(duration)
    len_wave = (1 / frequency) * sample_rate
    samples = []
    for i in range_samples:
        j = i % len_wave
        if j >= len_wave / 2:
            sample = 2 * (1 - j / len_wave)
        else:
            sample = 2 * (j / len_wave)
        samples.append(sample)
    samples = np.subtract(np.multiply(samples, 2), 1)  # Scale from 0..1 to 1..-1
    return samples


def gen_rectangle(frequency: int, sample_rate: int, duration: int) -> np.array:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: numpy.array of samples
    """
    len_wave = (1 / frequency) * sample_rate
    ones = np.ones(int(len_wave / 2))
    minus_ones = np.multiply(ones, -1)
    rec_wave = np.concatenate((ones, minus_ones))
    repeated_wave = np.array([])
    for i in range(int(duration // len_wave)):
        repeated_wave = np.concatenate((repeated_wave, rec_wave))
    missing = duration - len(repeated_wave)
    if missing != 0:
        repeated_wave = np.concatenate((repeated_wave, rec_wave[0: missing]))
    return repeated_wave


def gen_x_over_y(y: int, sample_rate: int, duration: int) -> np.array:
    """
    :param y: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: List of samples
    """
    half_duration = int(duration / 2)
    range_samples = range(-half_duration, half_duration)
    samples = []
    for x in range_samples:
        p = x ** y
        if isinstance(p, complex):
            p = abs(p)
        samples.append(p)
    max_sample = max([abs(min(samples)), abs(max(samples))])
    samples = np.divide(samples, max_sample)  # Scale from 0..1
    return samples


def gen_morph(signal_1: tuple[np.array, np.array],
              signal_2: tuple[np.array, np.array],
              sample_rate: int, duration: int) -> np.array:
    #signal_1 = (np.add(signal_1[0], 64000), np.add(signal_1[1], 64000))
    signal_2 = (signal_2[0][:len(signal_1[0])], signal_2[1][:len(signal_1[0])])
    difference_per_sample_x = np.divide(np.subtract(signal_2[0], signal_1[0]),
                                        duration/(len(signal_1[0]) * 1))
    difference_per_sample_y = np.divide(np.subtract(signal_2[1], signal_1[1]),
                                        duration/(len(signal_1[0]) * 1))
    len_part = len(signal_2[0])
    morph_signal_x = new_increment_x = signal_1[0][0:len_part]
    morph_signal_y = new_increment_y = signal_1[1][0:len_part]
    for i in range(len_part, duration + len_part, len_part):
        last = i - len_part
        new_increment_x = np.add(new_increment_x, difference_per_sample_x)
        new_increment_y = np.add(new_increment_y, difference_per_sample_y)
        morph_signal_x = np.concatenate((morph_signal_x, new_increment_x))
        morph_signal_y = np.concatenate((morph_signal_y, new_increment_y))
    #for i in range(int(len(signal_1[0])/duration)):
    #    morph_signal_x = np.concatenate((morph_signal_x, signal_2[0]))
    #    morph_signal_y = np.concatenate((morph_signal_y, signal_2[1]))
    # Rest
    #morph_signal_x = np.subtract(morph_signal_x, 64000)
    #morph_signal_y = np.subtract(morph_signal_y, 64000)
    return morph_signal_x, morph_signal_y


def gen_morph_fft(signal_1: tuple[np.array, np.array],
                  signal_2: tuple[np.array, np.array],
                  sample_rate: int, duration: int) -> np.array:
    signal_2 = (signal_2[0][:len(signal_1[0])], signal_2[1][:len(signal_1[0])])


    assert len(signal_1[0]) == len(signal_2[0])
    fft_1 = np.fft.fft(signal_1)
    fft_2 = np.fft.fft(signal_2)
    difference_per_sample_x = np.divide(np.subtract(fft_2[0], fft_1[0]),
                                        duration/(len(fft_1[0]) * 1))
    difference_per_sample_y = np.divide(np.subtract(fft_2[1], fft_1[1]),
                                        duration/(len(fft_1[0]) * 1))
    len_part = len(fft_2[0])
    morph_fft_x = new_increment_x = fft_1[0]
    morph_fft_y = new_increment_y = fft_1[1]
    morph_signal_x = []
    morph_signal_y = []
    for i in range(len_part, duration + len_part, len_part):
        new_increment_x = np.add(new_increment_x, difference_per_sample_x)
        new_increment_y = np.add(new_increment_y, difference_per_sample_y)
        morph_signal_x = np.concatenate((morph_signal_x, np.real(np.fft.ifft(new_increment_x))))
        morph_signal_y = np.concatenate((morph_signal_y, np.real(np.fft.ifft(new_increment_y))))
        morph_signal_x = np.concatenate(
            (morph_signal_x, np.flip(np.real(np.fft.ifft(new_increment_x)))))
        morph_signal_y = np.concatenate(
            (morph_signal_y, np.flip(np.real(np.fft.ifft(new_increment_y)))))
    #for i in range(int(len(signal_1[0])/duration)):
    #    morph_signal_x = np.concatenate((morph_signal_x, signal_2[0]))
    #    morph_signal_y = np.concatenate((morph_signal_y, signal_2[1]))
    # Rest
    #morph_signal_x = np.subtract(morph_signal_x, 64000)
    #morph_signal_y = np.subtract(morph_signal_y, 64000)
    return morph_signal_x, morph_signal_y


def write(x_frames: List, y_frames: List, channels: int = 2, sample_width: int = 2, sample_rate: int = 48000, file_name: str = 'gen2.wav'):
    if channels > 2 or channels < 1:
        raise ValueError('Number of channels below 1 or above 2 are not allowed')
    print(f'len x {len(x_frames)} len y {len(y_frames)}')
    n_frames = len(x_frames)
    frames = []
    for i in range(n_frames):
        x_frame = 32767 * x_frames[i]
        frames.append(int(x_frame).to_bytes(sample_width, byteorder='little', signed=True))
        if channels == 1:
            continue
        y_frame = 32767 * y_frames[i]
        frames.append(int(y_frame).to_bytes(sample_width, byteorder='little', signed=True))

    write_wav_file(frames, channels=channels, sample_width=sample_width, sample_rate=sample_rate, file_name=file_name)


def write_4ch(x_frames: List, y_frames: List, x_frames_2: List, y_frames_2: List, sample_rate: int = 48000, file_name: str = 'gen2.wav'):
    print(f'len x {len(x_frames)} len y {len(y_frames)} len x2 {len(x_frames_2)} len y2 {len(y_frames_2)}')
    n_frames = len(x_frames)
    frames = []
    for i in range(n_frames):
        x_frame = 32767 * x_frames[i]
        frames.append(int(x_frame).to_bytes(2, byteorder='little', signed=True))
        y_frame = 32767 * y_frames[i]
        frames.append(int(y_frame).to_bytes(2, byteorder='little', signed=True))
        x_frame_2 = 32767 * x_frames_2[i]
        frames.append(int(x_frame_2).to_bytes(2, byteorder='little', signed=True))
        y_frame_2 = 32767 * y_frames_2[i]
        frames.append(int(y_frame_2).to_bytes(2, byteorder='little', signed=True))

    write_wav_file(frames, channels=4, sample_width=2, sample_rate=sample_rate, file_name=file_name)


def write_wav_file(frames: List, channels: int = 2, sample_width: int = 2, sample_rate: int = 48000, file_name: str = 'gen2.wav'):
    n_frames = len(frames)
    with wave.open(file_name, 'wb') as wav:
        wav.setparams((channels, sample_width, sample_rate,
                       n_frames, 'NONE', 'not compressed'))
        wav.writeframes(b''.join(frames))


def read(file_path: str, norm: bool = False) -> Union[List, tuple[List, List]]:
    # one/two channel, two bytes sample width
    with wave.open(file_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        raw_data = wav.readframes(n_frames)
        # Convert bytes to integers
        data = [int.from_bytes(raw_data[i:i + 2], byteorder='little', signed=True) for i in range(0, len(raw_data), 2)]
        del raw_data
        if n_channels == 2:
            l = [data[i] for i in range(0, len(data), 2)]
            r = [data[i] for i in range(1, len(data), 2)]
            data = (l, r)
        if norm:
            data = np.divide(data, 32767)
        return sample_rate, data


def normalize_signal(signal: Union[List, tuple]):
    if isinstance(signal, tuple):  # 2 channels
        max_1 = max(signal[0])
        max_2 = max(signal[1])
        _max = max(max_1, max_2)
        factor = 1 / _max
        return scale(signal[0], factor=factor), scale(signal[1], factor=factor)
    elif isinstance(signal, List):
        _max = max(signal)
        factor = 1 / _max
        return scale(signal, factor=factor)
