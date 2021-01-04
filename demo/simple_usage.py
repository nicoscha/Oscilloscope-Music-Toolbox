import math
import wave
import basic_wave

CHANNELS = 2
SAMPLEWIDTH = 2
FRAMERATE = 44100
NFRAMES = 5*FRAMERATE

a = basic_wave.BasicWave(440)
a_p = basic_wave.BasicWave(440, phi=math.pi/2)

with wave.open('a.wav', 'wb') as wav:
    wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                   'NONE', 'not compressed'))
    frames = []
    for i in range(NFRAMES):
        frames.append(a.play())
        frames.append(a_p.play())
    wav.writeframes(b''.join(frames))

bc = basic_wave.Wave([(493.88, 0.0, 1.0, 0.0), (523.25, 0.0, 1.0, 0.0)])
bc_p = basic_wave.Wave([(493.88, math.pi/2, 1.0, 0.0),
                        (523.25, math.pi/2, 1.0, 0.0)])

with wave.open('bc.wav', 'wb') as wav:
    wav.setparams((CHANNELS, SAMPLEWIDTH, FRAMERATE, NFRAMES,
                   'NONE', 'not compressed'))
    frames = []
    for i in range(NFRAMES):
        frames.append(bc.play())
        frames.append(bc_p.play())
    wav.writeframes(b''.join(frames))
