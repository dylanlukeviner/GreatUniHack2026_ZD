from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import random

USE_REAL_WHISPER = False 

if USE_REAL_WHISPER:
    import whisperx
    import torch

class AITutorWorker(QThread):
    # Signals: user_text, clarity_score, ai_response, ai_correction, ai_pronunciation
    finished = pyqtSignal(str, int, str, str, str)  
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
            
            ai_response = "Отличный выбор! Какой кофе вы предпочитаете? Эспрессо, капучино или латте?"
            ai_correction = "Your grammar is perfect! However, make sure to stress the 'о' in 'кофе'."
            ai_pronunciation = "Ya kho-CHOO za-ka-ZAT' KO-fye, po-ZHA-loo-sta."

            # Emit back to GUI
            self.finished.emit(transcribed_text, clarity_score, ai_response, ai_correction, ai_pronunciation)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}")
        finally:
            if os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                except:
                    pass