from PyQt6.QtCore import QThread, pyqtSignal
from difflib import SequenceMatcher
import time
import os

# Toggle this to True when you are ready to use the real AI
USE_REAL_WHISPER = False 

if USE_REAL_WHISPER:
    import whisperx
    import torch

class AITutorWorker(QThread):
    finished = pyqtSignal(str, float, str)  
    error = pyqtSignal(str)

    def __init__(self, audio_path, target_phrase):
        super().__init__()
        self.audio_path = audio_path
        self.target_phrase = target_phrase

    def run(self):
        try:
            # ---------------------------------------------------------
            # STEP 1: TRANSCRIPTION
            # ---------------------------------------------------------
            if USE_REAL_WHISPER:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                # Load model (Consider moving this to a global state later so it doesn't load every time)
                model = whisperx.load_model("small", device, compute_type=compute_type)
                audio = whisperx.load_audio(self.audio_path)
                result = model.transcribe(audio, batch_size=8, language="ru")
                
                # Extract text from segments
                transcribed_text = " ".join([seg["text"] for seg in result["segments"]]).strip()
            else:
                # Mock processing for rapid UI testing
                time.sleep(1.5) 
                transcribed_text = "Привет, как дела" 

            # ---------------------------------------------------------
            # STEP 2: SIMILARITY SCORING (Using standard difflib/cdifflib)
            # ---------------------------------------------------------
            clean_target = self.target_phrase.lower().strip()
            clean_transcribed = transcribed_text.lower().strip()
            
            score = SequenceMatcher(None, clean_target, clean_transcribed).ratio()

            # ---------------------------------------------------------
            # STEP 3: CHATBOT RESPONSE (Mocked for now)
            # ---------------------------------------------------------
            # TODO: Replace with OpenAI / Ollama API call in the next sprint
            if score >= 0.8:
                feedback = "Excellent pronunciation! You sounded just like a native speaker."
            elif score >= 0.5:
                feedback = f"You are close! I heard '{transcribed_text}'. Pay attention to your vowels."
            else:
                feedback = f"Not quite. I heard '{transcribed_text}'. Let's try saying it slower."

            self.finished.emit(transcribed_text, score, feedback)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
        finally:
            # Cleanup temp audio file
            if os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                except:
                    pass