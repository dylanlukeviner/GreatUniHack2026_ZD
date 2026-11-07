import sys
import multiprocessing
import time
import os
import random
import tempfile

import sounddevice as sd
import soundfile as sf
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextBrowser, QProgressBar, QComboBox, QLabel)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- AI Engine Configuration ---
USE_REAL_WHISPER = False # Set to True to re-enable local Whisper

if USE_REAL_WHISPER:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import whisperx
    import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Audio Handler ---
class AudioRecorder:
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.recording = []
        self.stream = None
        self.temp_file = os.path.join(tempfile.gettempdir(), "tutor_recording.wav")
        self.last_recorded_data = None

    def start_recording(self, device_index=None):
        self.recording = []
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate, 
                channels=1, 
                device=device_index,
                callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"[MIC ERROR] {e}")

    def audio_callback(self, indata, frames, time, status):
        self.recording.append(indata.copy())

    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if not self.recording:
            return None

        self.last_recorded_data = np.concatenate(self.recording, axis=0)
        sf.write(self.temp_file, self.last_recorded_data, self.samplerate)
        return self.temp_file

    def play_last_recording(self):
        if self.last_recorded_data is not None:
            sd.play(self.last_recorded_data, self.samplerate)
            sd.wait()

# --- AI Engine Worker ---
class AITutorWorker(QThread):
    finished = pyqtSignal(str, int, str)  
    error = pyqtSignal(str)

    def __init__(self, audio_path, mode, model, tokenizer):
        super().__init__()
        self.audio_path = audio_path
        self.mode = mode
        self.model = model
        self.tokenizer = tokenizer

    def call_local_transformers(self, user_text):
        if self.mode == "Conversational Mode":
            system_prompt = "You are a friendly Russian tutor. The user is practicing Russian. Reply naturally in Russian. Then, provide the English translation of your reply, followed by brief feedback on their grammar or pronunciation in English."
        else:
            system_prompt = "You are a Russian tutor. Do not continue the conversation. Just correct the user's Russian grammar, provide the English translation of what they meant, and offer brief feedback."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            # Format chat for the Qwen model
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt")
            
            # Generate response locally
            outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.7)
            
            # Decode only the newly generated tokens
            input_length = inputs.input_ids.shape[1]
            response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            return response_text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def run(self):
        try:
            if USE_REAL_WHISPER:
                device = "cpu"
                compute_type = "int8"
                whisper_model = whisperx.load_model("small", device, compute_type=compute_type)
                audio = whisperx.load_audio(self.audio_path)
                result = whisper_model.transcribe(audio, batch_size=8, language="ru")
                transcribed_text = " ".join([seg["text"] for seg in result["segments"]]).strip()
                clarity_score = random.randint(75, 98) 
            else:
                time.sleep(1.0)
                transcribed_text = "Я хочу заказать кофе, пожалуйста." 
                clarity_score = 88

            if not transcribed_text:
                raise ValueError("No speech detected.")

            ai_response = self.call_local_transformers(transcribed_text)
            self.finished.emit(transcribed_text, clarity_score, ai_response)

        except Exception as e:
            self.error.emit(f"{str(e)}")
        finally:
            if os.path.exists(self.audio_path):
                try: os.remove(self.audio_path)
                except: pass

# --- GUI Application ---
class TutorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Language Tutor (Pure Local PyTorch)")
        self.resize(650, 800)
        self.recorder = AudioRecorder()
        self.setup_ui()
        
        self.add_system_message("Downloading/Loading the 0.5B AI model. This takes a moment...")
        QApplication.processEvents() 
        
        try:
            # Automatically downloads Qwen 0.5B on first run, loads from cache afterwards
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.add_system_message("Ready! Select your mode and hold the button to speak.")
        except Exception as e:
            self.add_system_message(f"<span style='color:red;'>Failed to load model: {str(e)}</span>")
            self.btn_record.setEnabled(False)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["Conversational Mode", "Review Mode"])
        layout.addWidget(self.mode_dropdown)

        self.chat_display = QTextBrowser()
        self.chat_display.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.chat_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.btn_record = QPushButton("Hold to Speak")
        self.btn_record.setStyleSheet("padding: 15px; font-weight: bold; background-color: #2b78e4; color: white; border-radius: 10px;")
        self.btn_record.pressed.connect(self.start_recording)
        self.btn_record.released.connect(self.stop_recording)
        layout.addWidget(self.btn_record)

    def start_recording(self):
        self.btn_record.setText("Listening...")
        self.btn_record.setStyleSheet("padding: 15px; font-weight: bold; background-color: #e74c3c; color: white; border-radius: 10px;")
        self.recorder.start_recording()

    def stop_recording(self):
        self.btn_record.setText("Hold to Speak")
        self.btn_record.setStyleSheet("padding: 15px; font-weight: bold; background-color: #2b78e4; color: white; border-radius: 10px;")
        audio_path = self.recorder.stop_recording()
        if audio_path:
            self.process_audio(audio_path)

    def process_audio(self, audio_path):
        self.btn_record.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker = AITutorWorker(audio_path, self.mode_dropdown.currentText(), self.model, self.tokenizer)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def add_system_message(self, text):
        self.chat_display.append(f"<div style='text-align: center; color: #888;'><i>{text}</i></div><br>")

    def on_finished(self, user_text, clarity, ai_response):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Format output gracefully using basic HTML
        ai_response_html = ai_response.replace('\n', '<br>')
        
        self.chat_display.append(f"<div style='text-align: right; color: #2b78e4;'><b>You:</b> {user_text}</div>")
        self.chat_display.append(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 8px;'><b>Tutor:</b><br>{ai_response_html}</div><br>")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def on_error(self, error_msg):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.chat_display.append(f"<b style='color:red;'>Error:</b> {error_msg}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass
    app = QApplication(sys.argv)
    window = TutorApp()
    window.show()
    sys.exit(app.exec())