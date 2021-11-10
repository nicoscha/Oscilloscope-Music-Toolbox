from typing import List
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
