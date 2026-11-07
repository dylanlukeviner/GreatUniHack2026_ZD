import sys
import multiprocessing
import traceback
import time
import os
import random
import tempfile

import sounddevice as sd
import soundfile as sf
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextBrowser, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- AI Engine Configuration ---
USE_REAL_WHISPER = True 

if USE_REAL_WHISPER:
    import whisperx
    import torch

# --- Audio Handler ---
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


# --- AI Engine Worker ---
class AITutorWorker(QThread):
    # Signals: user_text, clarity_score, ai_response, ai_translation, ai_correction, ai_pronunciation
    finished = pyqtSignal(str, int, str, str, str, str)  
    error = pyqtSignal(str)

    def __init__(self, audio_path, chat_history):
        super().__init__()
        self.audio_path = audio_path
        self.chat_history = chat_history

    def run(self):
        try:
            # ---------------------------------------------------------
            # STEP 1: TRANSCRIPTION & CLARITY SCORE
            # ---------------------------------------------------------
            if USE_REAL_WHISPER:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                model = whisperx.load_model("small", device, compute_type=compute_type)
                audio = whisperx.load_audio(self.audio_path)
                result = model.transcribe(audio, batch_size=8, language="ru")
                
                transcribed_text = " ".join([seg["text"] for seg in result["segments"]]).strip()
                
                # In a real scenario, we would average the word-level confidence scores from WhisperX
                clarity_score = random.randint(75, 98) 
            else:
                time.sleep(1.5) # Simulate processing
                transcribed_text = "Я хочу заказать кофе, пожалуйста." 
                clarity_score = 88

            if not transcribed_text:
                raise ValueError("No speech detected. Please try again.")

            # ---------------------------------------------------------
            # STEP 2: LLM CONVERSATION & EVALUATION
            # ---------------------------------------------------------
            # TODO: In the next sprint, we will pass `self.chat_history` and `transcribed_text` 
            # to an LLM (OpenAI/Ollama) with a system prompt asking for JSON output.
            
            # --- MOCK LLM RESPONSE FOR PROTOTYPING ---
            time.sleep(1.0) # Simulate LLM generation time
            
            # Dynamically uses the transcribed text for feedback
            ai_response = f"Я услышал: «{transcribed_text}». Отлично сказано! Чем я могу вам помочь дальше?"
            ai_translation = f"I heard: '{transcribed_text}'. Well said! How can I help you further?"
            ai_correction = f"Your phrasing for '{transcribed_text}' was clear. Keep practicing your vocabulary!"
            ai_pronunciation = "Try to speak a little slower to perfect your vowel sounds."

            # Emit back to GUI
            self.finished.emit(transcribed_text, clarity_score, ai_response, ai_translation, ai_correction, ai_pronunciation)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}")
        finally:
            if os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                except:
                    pass


# --- GUI Application ---
class TutorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Language Tutor - Conversational Mode")
        self.resize(650, 800)

        self.recorder = AudioRecorder()
        self.worker = None
        self.chat_history = [] 

        self.setup_ui()
        self.add_system_message("Welcome! Hold the button to speak.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Chat Display
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setStyleSheet("""
            QTextBrowser {
                background-color: #f7f9fc;
                border: 1px solid #d1d9e6;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 15px;
            }
        """)
        layout.addWidget(self.chat_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(4)
        layout.addWidget(self.progress_bar)

        # Controls Layout
        controls_layout = QHBoxLayout()

        self.btn_record = QPushButton("Hold to Speak")
        self.btn_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #2b78e4; color: white; 
                font-weight: bold; padding: 18px; font-size: 16px;
                border-radius: 25px;
            }
            QPushButton:pressed { background-color: #1a5baf; }
            QPushButton:disabled { background-color: #a0c4f2; }
        """)
        self.btn_record.pressed.connect(self.start_recording)
        self.btn_record.released.connect(self.stop_recording)
        controls_layout.addWidget(self.btn_record, stretch=3)

        self.btn_play = QPushButton("Play Audio")
        self.btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d; color: white; 
                font-weight: bold; padding: 18px; font-size: 14px;
                border-radius: 25px;
            }
            QPushButton:hover { background-color: #95a5a6; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.recorder.play_last_recording)
        controls_layout.addWidget(self.btn_play, stretch=1)

        layout.addLayout(controls_layout)

    def start_recording(self):
        self.btn_record.setText("Listening... (Release to Send)")
        self.btn_record.setStyleSheet(self.btn_record.styleSheet().replace("#2b78e4", "#e74c3c"))
        self.btn_play.setEnabled(False)
        
        # Starts recording using the default microphone
        self.recorder.start_recording()

    def stop_recording(self):
        self.btn_record.setText("Hold to Speak")
        self.btn_record.setStyleSheet(self.btn_record.styleSheet().replace("#e74c3c", "#2b78e4"))
        
        audio_path = self.recorder.stop_recording()
        if audio_path:
            self.btn_play.setEnabled(True)
            self.process_audio(audio_path)

    def process_audio(self, audio_path):
        self.btn_record.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.worker = AITutorWorker(audio_path, self.chat_history)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def add_system_message(self, text):
        html = f"<div style='text-align: center; margin: 15px 0; color: #888; font-size: 12px;'>{text}</div>"
        self.chat_display.append(html)

    def append_user_message(self, text, clarity_score):
        color = "#27ae60" if clarity_score >= 80 else "#f39c12" if clarity_score >= 50 else "#e74c3c"
        html = f"""
        <div style='margin-bottom: 15px; text-align: right;'>
            <div style='display: inline-block; background-color: #2b78e4; color: white; padding: 12px 18px; border-radius: 18px; max-width: 70%; text-align: left;'>
                {text}
            </div>
            <div style='font-size: 11px; color: {color}; margin-top: 4px; margin-right: 5px;'>
                Speech Clarity: {clarity_score}%
            </div>
        </div>
        """
        self.chat_display.append(html)

    def append_ai_message(self, response, translation, correction, pronunciation):
        html = f"""
        <div style='margin-bottom: 20px; text-align: left;'>
            <div style='display: inline-block; background-color: #ffffff; color: #333; padding: 15px; border-radius: 18px; border: 1px solid #e0e0e0; max-width: 85%; box-shadow: 0px 2px 4px rgba(0,0,0,0.05);'>
                <b style='color: #2b78e4;'>Tutor</b><br><br>
                <span style='font-size: 15px;'>{response}</span>
                <div style='margin-top: 6px; font-size: 13px; color: #888888;'><i>{translation}</i></div>
                <hr style='border: 0; border-top: 1px solid #eee; margin: 12px 0;'>
                <div style='background-color: #fff9e6; padding: 10px; border-radius: 8px; font-size: 13px; color: #665c3d;'>
                    <b>Feedback:</b> {correction}<br><br>
                    <b>Pronunciation:</b> <i>{pronunciation}</i>
                </div>
            </div>
        </div>
        """
        self.chat_display.append(html)

    def on_processing_finished(self, user_text, clarity_score, ai_response, ai_translation, ai_correction, ai_pronunciation):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": ai_response})

        self.append_user_message(user_text, clarity_score)
        self.append_ai_message(ai_response, ai_translation, ai_correction, ai_pronunciation)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def on_processing_error(self, error_msg):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.add_system_message(f"<span style='color:red;'>Error: {error_msg}</span>")


# --- Main Execution ---
if __name__ == '__main__':
    # MANDATORY for PyTorch/WhisperX/Demucs on Windows to prevent crash loops
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    try:
        app = QApplication(sys.argv)
        window = TutorApp()
        window.show()
        sys.exit(app.exec())
    except Exception as run_error:
        print("--- RUNTIME ERROR BLOCK EXECUTED ---")
        traceback.print_exc()
        input("\nPress Enter to exit terminal...")