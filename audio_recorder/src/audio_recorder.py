import pyaudio
import threading

class AudioRecorder:
    def __init__(self):
        p = pyaudio.PyAudio()
        # try to use virtual cable device
        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and 'cable' in info['name'].lower():
                device_index = i
                break
        if device_index is None:
            device_index = p.get_default_input_device_info()['index']
        self.input_device_index = device_index
        self.audio = p
        self.frames = []
        self.recording = False
        self.rate = 44100
        self.chunk = 1024
        self.channels = 2
        self.fmt = pyaudio.paInt16

    def start_recording(self):
        if self.recording:
            return
        self.frames = []
        self.recording = True
        threading.Thread(target=self._record, daemon=True).start()

    def _record(self):
        stream = self.audio.open(format=self.fmt,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 input_device_index=self.input_device_index,
                                 frames_per_buffer=self.chunk)
        while self.recording:
            self.frames.append(stream.read(self.chunk, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()

    def stop_recording(self):
        self.recording = False

    def play_recording(self):
        if not self.frames:
            return
        stream = self.audio.open(format=self.fmt,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=self.chunk)
        for f in self.frames:
            stream.write(f)
        stream.stop_stream()
        stream.close()

    def get_recording_status(self):
        return self.recording

    def cleanup(self):
        self.audio.terminate() 