from math import pi
import wave
from _old_py_scripts import basic_wave as bw

E4 = 329.63
B3 = 246.94
G3 = 196.00
D3 = 146.83
A2 = 110.00
E2 = 82.41
AS2 = 116.54
G2 = 98.00

CHANNELS = 2
SAMPLEWIDTH = 2
FRAMERATE = 44100
NFRAMES = 5*FRAMERATE

a = bw.BasicWave(440)
a_p = bw.BasicWave(440, phi=pi/2)

with wave.open('a.wav', 'wb') as wav:
    wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                   'NONE', 'not compressed'))
    frames = []
    for i in range(NFRAMES):
        frames.append(a.play())
        frames.append(a_p.play())
    wav.writeframes(b''.join(frames))

bc = bw.Wave([(493.88, 0.0, 1.0, 0.0), (523.25, 0.0, 1.0, 0.0)])
bc_p = bw.Wave([(493.88, pi/2, 1.0, 0.0),
                        (523.25, pi/2, 1.0, 0.0)])

with wave.open('bc.wav', 'wb') as wav:
    wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                   'NONE', 'not compressed'))
    frames = []
    for i in range(NFRAMES):
        frames.append(bc.play())
        frames.append(bc_p.play())
    wav.writeframes(b''.join(frames))

x_wave = bw.Wave([(AS2, 0.0, 0.5, 0.0,
                   [bw.Modifier(frequency=G3, start=0.25, swap=True),
                    bw.Modifier(frequency=AS2, start=0.50, swap=True),
                    bw.Modifier(frequency=G3, start=0.75, swap=True),
                    bw.Modifier(frequency=AS2, start=1.00, swap=True),
                    bw.Modifier(frequency=G3, start=1.25, swap=True),
                    bw.Modifier(frequency=AS2, start=1.50, swap=True),
                    bw.Modifier(frequency=G3, start=1.75, swap=True),
                    bw.Modifier(frequency=AS2, start=2.00, swap=True),
                    bw.Modifier(frequency=G3, start=2.25, swap=True),
                    bw.Modifier(frequency=AS2, start=2.50, swap=True),
                    bw.Modifier(frequency=G3, start=2.75, swap=True),
                    bw.Modifier(frequency=AS2, start=3.00, swap=True),
                    bw.Modifier(frequency=G3, start=3.25, swap=True),
                    ]),
                 (A2, 0.0, 0.0, 0.0,
                   [bw.Modifier(magnitude=0.5, start=1.00, swap=True),
                    bw.Modifier(magnitude=0.0, start=1.25, swap=True), ])
                  ])
y_wave = bw.Wave([(AS2, pi / 2, 0.5, 0.0,
                   [bw.Modifier(frequency=G3, start=0.25, swap=True),
                    bw.Modifier(frequency=AS2, start=0.50, swap=True),
                    bw.Modifier(frequency=G3, start=0.75, swap=True),
                    bw.Modifier(frequency=AS2, start=1.00, swap=True),
                    bw.Modifier(frequency=G3, start=1.25, swap=True),
                    bw.Modifier(frequency=AS2, start=1.50, swap=True),
                    bw.Modifier(frequency=G3, start=1.75, swap=True),
                    bw.Modifier(frequency=AS2, start=2.00, swap=True),
                    bw.Modifier(frequency=G3, start=2.25, swap=True),
                    bw.Modifier(frequency=AS2, start=2.50, swap=True),
                    bw.Modifier(frequency=G3, start=2.75, swap=True),
                    bw.Modifier(frequency=AS2, start=3.00, swap=True),
                    bw.Modifier(frequency=G3, start=3.25, swap=True),
                    ]),
                  (A2, 0.0, 0.0, 0.0,
                   [bw.Modifier(magnitude=0.5, start=1.00, swap=True),
                    bw.Modifier(magnitude=0.0, start=1.25, swap=True), ])
                  ])

with wave.open('../../demo/circle_of_notes.wav', 'wb') as wav:
    wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                   'NONE', 'not compressed'))
    frames = []
    for i in range(NFRAMES):
        frames.append(x_wave.play())
        frames.append(y_wave.play())
    wav.writeframes(b''.join(frames))

x_wave_1 = bw.Wave([(300, 0.0, 0.5, 0.0, [bw.Modifier(frequency=150),
                                          bw.Modifier(frequency=0.1,
                                                      start=2)])])
y_wave_1 = bw.Wave([(300, 0.0, 0.5, 0.0)])

with wave.open('../../demo/gamma.wav', 'wb') as wav:
    wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                   'NONE', 'not compressed'))
    frames = []
    for i in range(NFRAMES):
        frames.append(x_wave_1.play())
        frames.append(y_wave_1.play())
    wav.writeframes(b''.join(frames))
