import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os

class AudioRecorder:
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.recording = []
        self.stream = None
        self.temp_file = os.path.join(tempfile.gettempdir(), "tutor_recording.wav")
        self.last_recorded_data = None

    @staticmethod
    def get_input_devices():
        """Returns a list of available microphones."""
        devices = sd.query_devices()
        input_devices = []
        try:
            default_device = sd.default.device[0]
        except Exception:
            default_device = -1

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0: # Only get devices that can record
                input_devices.append({
                    "index": i,
                    "name": dev['name'],
                    "is_default": (i == default_device)
                })
        return input_devices

    def start_recording(self, device_index=None):
        """Starts the microphone stream using a specific device."""
        self.recording = []
        print(f"\n[MIC] Attempting to start stream on device index: {device_index}")
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate, 
                channels=1, 
                device=device_index,
                callback=self.audio_callback
            )
            self.stream.start()
            print("[MIC] Stream started successfully.")
        except Exception as e:
            print(f"[MIC ERROR] Could not open stream: {e}")

    def audio_callback(self, indata, frames, time, status):
        """Appends audio chunks and prints volume levels to terminal."""
        if status:
            print(f"[MIC STATUS WARNING] {status}")
        
        self.recording.append(indata.copy())
        
        # Calculate volume level
        volume_norm = np.linalg.norm(indata) * 10
        
        # Print actual numbers so we don't get blank lines if it's 0
        print(f"Vol: {volume_norm:05.2f} |" + "=" * int(volume_norm)) 

    def stop_recording(self):
        """Stops the stream and saves the audio to a temp WAV file."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[MIC] Stream stopped.")
        
        if not self.recording:
            print("[MIC] Error: No audio data captured.")
            return None

        self.last_recorded_data = np.concatenate(self.recording, axis=0)
        
        # Check if the recording is completely silent
        if np.max(np.abs(self.last_recorded_data)) == 0:
            print("[MIC WARNING] The recorded audio is completely silent (all zeros).")
            
        sf.write(self.temp_file, self.last_recorded_data, self.samplerate)
        return self.temp_file

    def play_last_recording(self):
        """Plays back the last recorded audio."""
        if self.last_recorded_data is not None:
            print("[MIC] Playing back recorded audio...")
            sd.play(self.last_recorded_data, self.samplerate)
            sd.wait()
        else:
            print("[MIC] Nothing to play.")