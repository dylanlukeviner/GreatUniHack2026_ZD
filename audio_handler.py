import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os

class AudioRecorder:
    def __init__(self, samplerate=16000):
        # 16000 Hz is the native sample rate for Whisper models
        self.samplerate = samplerate
        self.recording = []
        self.stream = None
        self.temp_file = os.path.join(tempfile.gettempdir(), "tutor_recording.wav")

    def start_recording(self):
        """Starts the microphone stream."""
        self.recording = []
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        """Appends audio chunks to the recording array."""
        self.recording.append(indata.copy())

    def stop_recording(self):
        """Stops the stream and saves the audio to a temp WAV file using soundfile."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if not self.recording:
            return None

        # Flatten the array and save as WAV using soundfile
        audio_data = np.concatenate(self.recording, axis=0)
        sf.write(self.temp_file, audio_data, self.samplerate)
        
        return self.temp_file