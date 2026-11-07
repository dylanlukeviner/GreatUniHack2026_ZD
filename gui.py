from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextBrowser, QProgressBar, QComboBox, QLabel)
from PyQt6.QtCore import Qt
from audio_handler import AudioRecorder
from ai_engine import AITutorWorker

class TutorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Language Tutor - Conversational Mode")
        self.resize(650, 800)

        self.recorder = AudioRecorder()
        self.worker = None
        self.chat_history = [] 

        self.setup_ui()
        self.populate_mic_dropdown()
        self.add_system_message("Welcome! Please select your microphone from the dropdown above, then hold the button to speak.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- NEW: Microphone Selector ---
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("<b>Microphone:</b>"))
        self.mic_dropdown = QComboBox()
        self.mic_dropdown.setStyleSheet("padding: 5px; font-size: 14px;")
        mic_layout.addWidget(self.mic_dropdown, stretch=1)
        layout.addLayout(mic_layout)

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

        self.btn_record = QPushButton("🎤 Hold to Speak")
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

        self.btn_play = QPushButton("🔊 Play Audio")
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

    def populate_mic_dropdown(self):
        """Fills the dropdown with available microphones."""
        devices = self.recorder.get_input_devices()
        for dev in devices:
            # Add item with the device name, and store the device index as hidden data
            self.mic_dropdown.addItem(dev["name"], dev["index"])
            if dev["is_default"]:
                self.mic_dropdown.setCurrentIndex(self.mic_dropdown.count() - 1)

    def start_recording(self):
        self.btn_record.setText("🔴 Listening... (Release to Send)")
        self.btn_record.setStyleSheet(self.btn_record.styleSheet().replace("#2b78e4", "#e74c3c"))
        self.btn_play.setEnabled(False)
        
        # Get the selected microphone index from the dropdown
        selected_mic_index = self.mic_dropdown.currentData()
        self.recorder.start_recording(device_index=selected_mic_index)

    def stop_recording(self):
        self.btn_record.setText("🎤 Hold to Speak")
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

    def append_ai_message(self, response, correction, pronunciation):
        html = f"""
        <div style='margin-bottom: 20px; text-align: left;'>
            <div style='display: inline-block; background-color: #ffffff; color: #333; padding: 15px; border-radius: 18px; border: 1px solid #e0e0e0; max-width: 85%; box-shadow: 0px 2px 4px rgba(0,0,0,0.05);'>
                <b style='color: #2b78e4;'>🤖 Tutor</b><br><br>
                <span style='font-size: 15px;'>{response}</span>
                <hr style='border: 0; border-top: 1px solid #eee; margin: 12px 0;'>
                <div style='background-color: #fff9e6; padding: 10px; border-radius: 8px; font-size: 13px; color: #665c3d;'>
                    <b>📝 Feedback:</b> {correction}<br><br>
                    <b>🗣️ Pronunciation:</b> <i>{pronunciation}</i>
                </div>
            </div>
        </div>
        """
        self.chat_display.append(html)

    def on_processing_finished(self, user_text, clarity_score, ai_response, ai_correction, ai_pronunciation):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": ai_response})

        self.append_user_message(user_text, clarity_score)
        self.append_ai_message(ai_response, ai_correction, ai_pronunciation)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def on_processing_error(self, error_msg):
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.add_system_message(f"<span style='color:red;'>Error: {error_msg}</span>")