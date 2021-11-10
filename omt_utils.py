from typing import List
from math import cos, sin, pi
import warnings


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


def scale(signal: List[float], factor: float) -> List[float]:
    return [_ * factor for _ in signal]


def add(signal_1: List[float], signal_2: List[float]) -> List[float]:
    if len(signal_1) != len(signal_2):
        warnings.warn('Signals have different length')
    return [_1 + _2 for (_1, _2) in zip(signal_1, signal_2)]


def multiply(signal_1: List[float], signal_2: List[float]) -> List[float]:
    if len(signal_1) != len(signal_2):
        warnings.warn('Signals have different length')
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
