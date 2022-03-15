from typing import Union, List
from math import cos, sin, pi
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


def scale(signal: List[float], factor: float, to_int: bool = False) -> List[float]:
    if to_int:
        return [int(_ * factor) for _ in signal]
    else:
        return [_ * factor for _ in signal]


def add(signal_1: List[float], signal_2: List[float], factor_1: float = 1.0, factor_2: float = 1.0, to_int: bool = False) -> List[float]:
    if len(signal_1) != len(signal_2):
        warnings.warn('Signals have different length')
    if to_int:
        if factor_1 == 1.0 or factor_2 == 1.0:
            sig = [int(_1 + _2) for (_1, _2) in zip(signal_1, signal_2)]
        else:
            sig = [int(_1 * factor_1 + _2 * factor_2) for (_1, _2) in zip(signal_1, signal_2)]
    else:
        if factor_1 == 1.0 or factor_2 == 1.0:
            sig = [_1 + _2 for (_1, _2) in zip(signal_1, signal_2)]
        else:
            sig = [_1 * factor_1 + _2 * factor_2 for (_1, _2) in zip(signal_1, signal_2)]
    return sig


def offset(signal: List[float], offset_value: float) -> List:
    return [_ + offset_value for _ in signal]


def multiply(signal_1: List[float], signal_2: List[float], to_int: bool = False) -> List[float]:
    if len(signal_1) != len(signal_2):
        warnings.warn('Signals have different length')
    if to_int:
        return [int(_1 * _2) for (_1, _2) in zip(signal_1, signal_2)]
    else:
        return [_1 * _2 for (_1, _2) in zip(signal_1, signal_2)]


def gen_sin(frequency: int, sample_rate: int, duration: int) -> List:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: List of samples
    """
    range_samples = range(duration)
    samples = [sin(2 * pi * frequency * (i / sample_rate)) for i in range_samples]
    return samples


def gen_cos(frequency: int, sample_rate: int, duration: int) -> List:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: List of samples
    """
    range_samples = range(duration)
    samples = [cos(2 * pi * frequency * (i / sample_rate)) for i in range_samples]
    return samples


def gen_sawtooth(frequency: int, sample_rate: int, duration: int) -> List:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: List of samples
    """
    range_samples = range(duration)
    len_wave = (1 / frequency) * sample_rate
    samples = [(i % len_wave) / len_wave for i in range_samples]
    samples = [2 * s - 1 for s in samples]  # Scale from 0..1 to 1..-1
    return samples


def gen_triangle(frequency: int, sample_rate: int, duration: int) -> List:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: List of samples
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
    samples = [2 * s - 1 for s in samples]  # Scale from 0..1 to 1..-1
    return samples


def gen_rectangle(frequency: int, sample_rate: int, duration: int) -> List:
    """
    :param frequency: Frequency in Hz
    :param sample_rate: Samples pre second
    :param duration: Duration in samples
    :return: List of samples
    """
    range_samples = range(duration)
    len_wave = (1 / frequency) * sample_rate
    samples = [1 if (i % len_wave) >= len_wave / 2 else - 1 for i in range_samples]
    return samples


def gen_x_over_y(y: int, sample_rate: int, duration: int) -> List:
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
    samples = [s / max_sample for s in samples]  # Scale from 0..1
    return samples


def write(x_frames: List, y_frames: List, channels: int = 2, sample_width: int = 2, sample_rate: int = 48000):
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

    write_wav_file(frames, channels=channels, sample_width=sample_width, sample_rate=sample_rate)


def write_4ch(x_frames: List, y_frames: List, x_frames_2: List, y_frames_2: List, sample_rate: int = 48000):
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

    write_wav_file(frames, channels=4, sample_width=2, sample_rate=sample_rate)


def write_wav_file(frames: List, channels: int = 2, sample_width: int = 2, sample_rate: int = 48000):
    n_frames = len(frames)
    with wave.open('gen2.wav', 'wb') as wav:
        wav.setparams((channels, sample_width, sample_rate,
                       n_frames, 'NONE', 'not compressed'))
        wav.writeframes(b''.join(frames))


def read(file_path: str,) -> Union[List, tuple[List, List]]:
    # one/two channel, two bytes sample width
    with wave.open(file_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        raw_data = wav.readframes(n_frames)
        # Convert bytes to integers
        data = [int.from_bytes(raw_data[i:i + 2], byteorder='little', signed=True) for i in range(0, len(raw_data), 2)]
        if n_channels == 2:
            l = [data[i] for i in range(len(data)) if i % 2 == 0]
            r = [data[i] for i in range(len(data)) if i % 2 == 1]
            data = (l, r)
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
