from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QTextEdit, QLineEdit, QLabel, QProgressBar)
from audio_handler import AudioRecorder
from ai_engine import AITutorWorker

class TutorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Language Tutor Prototype")
        self.resize(600, 700)

        self.recorder = AudioRecorder()
        self.worker = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        layout.addWidget(QLabel("<b>Target Phrase (What you are trying to say):</b>"))
        self.target_input = QLineEdit("Привет, как дела")
        self.target_input.setStyleSheet("font-size: 16px; padding: 5px;")
        layout.addWidget(self.target_input)

        layout.addWidget(QLabel("<b>Chat & Feedback:</b>"))
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: #1e1e1e; color: #ffffff; font-size: 14px; padding: 10px;")
        layout.addWidget(self.chat_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.btn_record = QPushButton("🎤 Hold to Record")
        self.btn_record.setStyleSheet("background-color: #2b78e4; color: white; font-weight: bold; padding: 15px; font-size: 16px;")
        
        self.btn_record.pressed.connect(self.start_recording)
        self.btn_record.released.connect(self.stop_recording)
        
        layout.addWidget(self.btn_record)

    def start_recording(self):
        self.btn_record.setText("🔴 Recording... (Release to Stop)")
        self.btn_record.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 15px; font-size: 16px;")
        self.recorder.start_recording()

    def stop_recording(self):
        self.btn_record.setText("🎤 Hold to Record")
        self.btn_record.setStyleSheet("background-color: #2b78e4; color: white; font-weight: bold; padding: 15px; font-size: 16px;")
        
        audio_path = self.recorder.stop_recording()
        target_phrase = self.target_input.text()

        if audio_path and target_phrase:
            self.process_audio(audio_path, target_phrase)

    def process_audio(self, audio_path, target_phrase):
        self.chat_display.append("\n<i>🎙️ Audio captured. AI is analyzing...</i>")
        self.btn_record.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.worker = AITutorWorker(audio_path, target_phrase)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_finished(self, transcribed, score, feedback):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)

        accuracy_percent = int(score * 100)
        color = "#00ff00" if accuracy_percent >= 80 else "#ffa500" if accuracy_percent >= 50 else "#ff4444"

        self.chat_display.append(f"<b>You said:</b> {transcribed}")
        self.chat_display.append(f"<b>Accuracy:</b> <span style='color:{color};'>{accuracy_percent}%</span>")
        self.chat_display.append(f"<b>🤖 Tutor:</b> {feedback}")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def on_processing_error(self, error_msg):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.chat_display.append(f"<b style='color:#ff4444;'>Error:</b> {error_msg}")