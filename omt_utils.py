def _limit(signal):
    UPPER_XY_LIMIT = 65520
    LOWER_XY_LIMIT = 0
    if signal > UPPER_XY_LIMIT:
        return UPPER_XY_LIMIT
    elif signal < LOWER_XY_LIMIT:
        return LOWER_XY_LIMIT
    return signal


def in_bytes(signal):
    return int(_limit(signal)).to_bytes(2, byteorder='little', signed=True)