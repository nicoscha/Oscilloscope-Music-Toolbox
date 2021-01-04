import basic_wave as bw
import logging
import simpleaudio
import time
import wave


class Player:
    def __init__(self, x_wave, y_wave, loop_length, save_to_disk=False,
                 filename='test.wav'):
        self.x_wave = x_wave
        self.y_wave = y_wave
        self.buffer = []
        self.chunk_length = int(bw.FRAMERATE * loop_length)
        self.stop_playing = False
        self.write_to_disk = save_to_disk
        self.filename = filename

    def _calculate_frames(self):
        frames = []
        for i in range(self.chunk_length):
            frames.append(self.x_wave.play())
            frames.append(self.y_wave.play())
        return b''.join(frames)

    def play(self):
        t1 = time.time()
        self.buffer = self._calculate_frames()
        WO = simpleaudio.WaveObject(self.buffer, sample_rate=bw.FRAMERATE)
        t2 = time.time()
        logging.debug('_calculate_frames took ', t2 - t1, ' seconds')
        while not self.stop_playing:
            play_buffer = WO.play()
            if self.write_to_disk:
                self.write_to_disk = False
                with wave.open(self.filename, 'wb') as wav:
                    wav.setparams((bw.CHANNELS, bw.SAMPLEWIDTH, bw.FRAMERATE,
                                   self.chunk_length, 'NONE', 'not compressed'))
                    wav.writeframes(self.buffer)
            play_buffer.wait_done()
