import sys
import multiprocessing
import traceback
import time
import os
import random
import tempfile
import json

import sounddevice as sd
import soundfile as sf
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextBrowser, QProgressBar, QComboBox, QLabel)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- AI Engine Configuration ---
USE_REAL_WHISPER = True 
GGUF_MODEL_PATH = "qwen2.5-1.5b-instruct-q4_k_m.gguf" # Ensure this file is in your folder

if USE_REAL_WHISPER:
    import whisperx
    import torch

from llama_cpp import Llama

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
        devices = sd.query_devices()
        input_devices = []
        try:
            default_device = sd.default.device[0]
        except Exception:
            default_device = -1

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    "index": i,
                    "name": dev['name'],
                    "is_default": (i == default_device)
                })
        return input_devices

    def start_recording(self, device_index=None):
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
        if status:
            print(f"[MIC STATUS WARNING] {status}")
        
        self.recording.append(indata.copy())
        volume_norm = np.linalg.norm(indata) * 10
        print(f"Vol: {volume_norm:05.2f} |" + "=" * int(volume_norm)) 

    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[MIC] Stream stopped.")
        
        if not self.recording:
            print("[MIC] Error: No audio data captured.")
            return None

        self.last_recorded_data = np.concatenate(self.recording, axis=0)
        
        if np.max(np.abs(self.last_recorded_data)) == 0:
            print("[MIC WARNING] The recorded audio is completely silent.")
            
        sf.write(self.temp_file, self.last_recorded_data, self.samplerate)
        return self.temp_file

    def play_last_recording(self):
        if self.last_recorded_data is not None:
            print("[MIC] Playing back recorded audio...")
            sd.play(self.last_recorded_data, self.samplerate)
            sd.wait()
        else:
            print("[MIC] Nothing to play.")


# --- AI Engine Worker ---
class AITutorWorker(QThread):
    finished = pyqtSignal(str, int, str, str, str, str)  
    error = pyqtSignal(str)

    def __init__(self, audio_path, chat_history, mode, llm):
        super().__init__()
        self.audio_path = audio_path
        self.chat_history = chat_history
        self.mode = mode
        self.llm = llm # Reference to the globally loaded model

    def call_local_llm(self, user_text):
        if self.mode == "Conversational Mode":
            system_prompt = """You are a friendly Russian language tutor. The user is practicing Russian.
Reply in Russian to continue the conversation naturally."""
        else:
            system_prompt = """You are a Russian language tutor. The user is practicing Russian.
Do not continue the conversation. Review what they said and correct any obvious grammatical errors."""

        # Define strictly enforced JSON schema for the output
        json_schema = {
            "type": "object",
            "properties": {
                "russian_response": {"type": "string"},
                "english_translation": {"type": "string"},
                "feedback": {"type": "string"},
                "pronunciation_tip": {"type": "string"}
            },
            "required": ["russian_response", "english_translation", "feedback", "pronunciation_tip"]
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        try:
            print("[LLM] Generating response via llama-cpp-python...")
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={
                    "type": "json_object",
                    "schema": json_schema
                },
                temperature=0.7,
                max_tokens=300
            )
            
            # The model is constrained by the schema, so this will safely parse
            result_str = response['choices'][0]['message']['content']
            result = json.loads(result_str)
            
            return (
                result.get("russian_response", "Извините, произошла ошибка."),
                result.get("english_translation", "Sorry, an error occurred."),
                result.get("feedback", "No feedback provided."),
                result.get("pronunciation_tip", "No pronunciation tip provided.")
            )
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return (
                "Произошла ошибка при генерации ответа.", 
                "An error occurred while generating the response.", 
                str(e), 
                "N/A"
            )

    def run(self):
        try:
            if USE_REAL_WHISPER:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                model = whisperx.load_model("small", device, compute_type=compute_type)
                audio = whisperx.load_audio(self.audio_path)
                result = model.transcribe(audio, batch_size=8, language="ru")
                
                transcribed_text = " ".join([seg["text"] for seg in result["segments"]]).strip()
                clarity_score = random.randint(75, 98) 
            else:
                time.sleep(1.5)
                transcribed_text = "Я хочу заказать кофе, пожалуйста." 
                clarity_score = 88

            if not transcribed_text:
                raise ValueError("No speech detected. Please try again.")

            ai_response, ai_translation, ai_correction, ai_pronunciation = self.call_local_llm(transcribed_text)
            self.finished.emit(transcribed_text, clarity_score, ai_response, ai_translation, ai_correction, ai_pronunciation)

        except Exception as e:
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
        self.setWindowTitle("AI Language Tutor (GGUF CPU Edition)")
        self.resize(650, 800)

        self.recorder = AudioRecorder()
        self.worker = None
        self.chat_history = [] 

        self.setup_ui()
        
        # Load the model directly into the app state once
        self.add_system_message("Loading AI Model into memory. Please wait...")
        QApplication.processEvents() # Force UI update before blocking thread
        
        try:
            print(f"[INIT] Loading {GGUF_MODEL_PATH}...")
            # Automatically calculate threads, leaving 1 free for the OS/UI
            cpu_threads = max(1, multiprocessing.cpu_count() - 1)
            
            self.llm = Llama(
                model_path=GGUF_MODEL_PATH,
                n_ctx=2048,
                n_threads=cpu_threads,
                verbose=False # Set to True if you want to see detailed token stats
            )
            self.add_system_message("Ready! Select your mode and hold the button to speak.")
        except Exception as e:
            self.add_