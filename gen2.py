from numpy import fft
from typing import List, Final
from math import cos, sin, pi, sqrt
from omt_utils import gen_cos, gen_sin, gen_sawtooth, add, scale, multiply

from time import  time
t1 = time()
SAMPLE_RATE: Final = 192000#48000
CHANNELS: Final = 2
SAMPLEWIDTH: Final = 2





def write(x_frames, y_frames):
    n_frames = len(x_frames)
    frames = []
    for i in range(n_frames):
        frames.append(x_frames[i])
        frames.append(y_frames[i])

    import wave
    with wave.open('gen2.wav', 'wb') as wav:
        wav.setparams((CHANNELS, SAMPLEWIDTH, SAMPLE_RATE,
                       n_frames, 'NONE', 'not compressed'))
        wav.writeframes(b''.join(frames))


N_SAMPLES = 480000*5
#####
signal_1 = gen_sin(1000, SAMPLE_RATE, int(N_SAMPLES/2))
signal_1_1 = [int(32767 * (i * 0.3 - 0.5)) for i in signal_1]
signal_1_2 = [int(32767 * (i * 0.3 + 0.5)) for i in signal_1]
#signal_1_ = [s1 if i % (1/1000* (SAMPLE_RATE * 2)) >= (1/1000* SAMPLE_RATE) else s2 for (i, (s1, s2)) in enumerate(zip(signal_1_1, signal_1_2))]
saw_1 = gen_sin(50.5, SAMPLE_RATE, N_SAMPLES)
signal_1_ = [int(x1*x2) for (x1, x2) in zip(signal_1_1, saw_1)]

signal_1_ = [int(_ * ((i % SAMPLE_RATE)/SAMPLE_RATE)) for (i, _) in enumerate(signal_1_)]
b_signal_1 = [i.to_bytes(2, byteorder='little', signed=True) for i in signal_1_]

sinx = gen_sin(8000, SAMPLE_RATE, N_SAMPLES)
cosy = gen_cos(1000, SAMPLE_RATE, N_SAMPLES)
saw = gen_sawtooth(int(50), SAMPLE_RATE, N_SAMPLES)
signal_2 = add(cosy, saw, 0.3, 0.7)
signal_2 = scale(signal_2, 32767, True)
#signal_2 = [int(32767 * (1*x2)) for (x1, x2) in zip(sinx, signal_2)]

signal_2 = [int(_ * ((i % SAMPLE_RATE)/SAMPLE_RATE)) for (i, _) in enumerate(signal_2)]
b_signal_2 = [i.to_bytes(2, byteorder='little', signed=True) for i in signal_2]
write(b_signal_1, b_signal_2)
print(time()-t1)
exit()
###
coeff = fft.fft(signal_1)
print(coeff, len(coeff))
freqs = fft.fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
print('freqs ', freqs)
coeff2 = [(i, c) for (i, c) in enumerate(coeff) if abs(c) > 0.02]
print(coeff2, len(coeff2))
freqs_and_coeff = [(freqs[i], c) for (i, c) in coeff2]
for (f, c) in freqs_and_coeff:
    print(f'f: {str(f).rjust(8, " ")}Hz c:  {c}')


