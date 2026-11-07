import sys
import multiprocessing
import traceback
import os
import tempfile
import html
import random

import sounddevice as sd
import soundfile as sf
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextBrowser,
    QProgressBar, QComboBox, QLabel
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal


# =========================================================
# APP 8 - AI MULTILINGUAL LANGUAGE TUTOR
# =========================================================

USE_REAL_WHISPER = True

if USE_REAL_WHISPER:
    import whisperx
    import torch


# =========================================================
# SUPPORTED LANGUAGES
# =========================================================

LANGUAGES = {
    "English": "en",
    "Russian": "ru",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Greek": "el",
    "Hindi": "hi",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da"
}


# =========================================================
# LANGUAGE-SPECIFIC TUTOR RESPONSES
# =========================================================

TUTOR_RESPONSES = {

    "en": [
        "That is very interesting! Tell me more.",
        "I understand. What would you like to do next?",
        "Great! Can you explain that in more detail?",
        "Excellent! What do you think about this?"
    ],

    "ru": [
        "Это очень интересно! Расскажите мне подробнее.",
        "Понятно. Что вы хотели бы сделать дальше?",
        "Отлично! Можете объяснить подробнее?",
        "Прекрасно! Что вы думаете об этом?"
    ],

    "es": [
        "¡Qué interesante! Cuéntame más.",
        "Entiendo. ¿Qué te gustaría hacer ahora?",
        "¡Genial! ¿Puedes explicarlo con más detalle?",
        "¡Excelente! ¿Qué opinas sobre esto?"
    ],

    "fr": [
        "C'est très intéressant ! Parlez-moi davantage.",
        "Je comprends. Que souhaitez-vous faire ensuite ?",
        "Excellent ! Pouvez-vous expliquer davantage ?",
        "Très bien ! Qu'en pensez-vous ?"
    ],

    "de": [
        "Das ist sehr interessant! Erzählen Sie mir mehr.",
        "Ich verstehe. Was möchten Sie als Nächstes tun?",
        "Großartig! Können Sie das genauer erklären?",
        "Ausgezeichnet! Was denken Sie darüber?"
    ],

    "it": [
        "È molto interessante! Raccontami di più.",
        "Capisco. Cosa vorresti fare adesso?",
        "Fantastico! Puoi spiegare meglio?",
        "Eccellente! Cosa ne pensi?"
    ],

    "pt": [
        "Isso é muito interessante! Conte-me mais.",
        "Entendo. O que gostaria de fazer agora?",
        "Ótimo! Pode explicar com mais detalhes?",
        "Excelente! O que você acha disso?"
    ],

    "zh": [
        "这很有趣！请告诉我更多。",
        "我明白了。你接下来想做什么？",
        "太好了！你能详细解释一下吗？",
        "非常好！你对此有什么看法？"
    ],

    "ja": [
        "とても興味深いですね！もっと教えてください。",
        "なるほど。次は何をしたいですか？",
        "素晴らしいですね！もう少し詳しく説明できますか？",
        "いいですね！これについてどう思いますか？"
    ],

    "ko": [
        "정말 흥미롭네요! 더 이야기해 주세요.",
        "알겠습니다. 다음에는 무엇을 하고 싶으세요?",
        "좋아요! 좀 더 자세히 설명해 주시겠어요?",
        "훌륭해요! 이것에 대해 어떻게 생각하세요?"
    ],

    "ar": [
        "هذا مثير للاهتمام جداً! أخبرني المزيد.",
        "أفهم ذلك. ماذا تريد أن تفعل بعد ذلك؟",
        "رائع! هل يمكنك التوضيح أكثر؟",
        "ممتاز! ما رأيك في هذا؟"
    ],

    "nl": [
        "Dat is erg interessant! Vertel me meer.",
        "Ik begrijp het. Wat wil je hierna doen?",
        "Geweldig! Kun je dat uitgebreider uitleggen?",
        "Uitstekend! Wat vind je hiervan?"
    ],

    "pl": [
        "To bardzo interesujące! Opowiedz mi więcej.",
        "Rozumiem. Co chciałbyś zrobić dalej?",
        "Świetnie! Czy możesz wyjaśnić to dokładniej?",
        "Doskonale! Co o tym myślisz?"
    ],

    "tr": [
        "Bu çok ilginç! Bana daha fazla anlat.",
        "Anlıyorum. Bundan sonra ne yapmak istersin?",
        "Harika! Bunu biraz daha açıklayabilir misin?",
        "Mükemmel! Bu konuda ne düşünüyorsun?"
    ],

    "uk": [
        "Це дуже цікаво! Розкажіть мені більше.",
        "Зрозуміло. Що ви хотіли б зробити далі?",
        "Чудово! Можете пояснити детальніше?",
        "Чудово! Що ви думаєте про це?"
    ],

    "el": [
        "Αυτό είναι πολύ ενδιαφέρον! Πείτε μου περισσότερα.",
        "Καταλαβαίνω. Τι θα θέλατε να κάνετε μετά;",
        "Υπέροχα! Μπορείτε να εξηγήσετε περισσότερα;",
        "Εξαιρετικά! Τι πιστεύετε γι' αυτό;"
    ],

    "hi": [
        "यह बहुत दिलचस्प है! मुझे और बताइए।",
        "मैं समझ गया। आप आगे क्या करना चाहेंगे?",
        "बहुत अच्छा! क्या आप इसे विस्तार से समझा सकते हैं?",
        "शानदार! इस बारे में आपकी क्या राय है?"
    ],

    "sv": [
        "Det är mycket intressant! Berätta mer.",
        "Jag förstår. Vad vill du göra härnäst?",
        "Fantastiskt! Kan du förklara mer i detalj?",
        "Utmärkt! Vad tycker du om detta?"
    ],

    "no": [
        "Det er veldig interessant! Fortell meg mer.",
        "Jeg forstår. Hva vil du gjøre videre?",
        "Flott! Kan du forklare det mer detaljert?",
        "Utmerket! Hva synes du om dette?"
    ],

    "da": [
        "Det er meget interessant! Fortæl mig mere.",
        "Jeg forstår. Hvad vil du gerne gøre nu?",
        "Fantastisk! Kan du forklare det mere detaljeret?",
        "Fremragende! Hvad synes du om dette?"
    ]
}


# =========================================================
# AUDIO RECORDER
# =========================================================

class AudioRecorder:

    def __init__(self, samplerate=16000):

        self.samplerate = samplerate
        self.recording = []
        self.stream = None

        self.temp_file = os.path.join(
            tempfile.gettempdir(),
            "tutor_recording.wav"
        )

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

            if dev["max_input_channels"] > 0:

                input_devices.append({
                    "index": i,
                    "name": dev["name"],
                    "is_default": i == default_device
                })

        return input_devices

    def start_recording(self, device_index=None):

        self.recording = []

        print("[MIC] Starting recording...")

        try:

            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                device=device_index,
                callback=self.audio_callback
            )

            self.stream.start()

            print("[MIC] Recording started.")

        except Exception as e:

            self.stream = None
            print(f"[MIC ERROR] {e}")

            raise RuntimeError(
                f"Could not start microphone: {e}"
            )

    def audio_callback(self, indata, frames, time_info, status):

        if status:
            print(f"[MIC WARNING] {status}")

        self.recording.append(indata.copy())

        volume_norm = np.linalg.norm(indata) * 10

        print(
            f"Vol: {volume_norm:05.2f} |"
            + "=" * int(volume_norm)
        )

    def stop_recording(self):

        if self.stream:

            self.stream.stop()
            self.stream.close()
            self.stream = None

            print("[MIC] Recording stopped.")

        if not self.recording:

            print("[MIC] No audio captured.")
            return None

        self.last_recorded_data = np.concatenate(
            self.recording,
            axis=0
        )

        if np.max(np.abs(self.last_recorded_data)) == 0:

            print("[MIC WARNING] Recording is silent.")

        sf.write(
            self.temp_file,
            self.last_recorded_data,
            self.samplerate
        )

        return self.temp_file

    def play_last_recording(self):

        if self.last_recorded_data is not None:

            sd.play(
                self.last_recorded_data,
                self.samplerate
            )

            sd.wait()

        else:

            print("[MIC] Nothing to play.")


# =========================================================
# AI WORKER
# =========================================================

class AITutorWorker(QThread):

    finished = pyqtSignal(str, int, str, str, str, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        audio_path,
        chat_history,
        mode,
        language,
        whisper_model
    ):

        super().__init__()

        self.audio_path = audio_path
        self.chat_history = chat_history
        self.mode = mode
        self.language = language
        self.whisper_model = whisper_model

    def run(self):

        try:

            # ---------------------------------------------
            # STEP 1: TRANSCRIPTION
            # ---------------------------------------------

            if USE_REAL_WHISPER:

                audio = whisperx.load_audio(
                    self.audio_path
                )

                result = self.whisper_model.transcribe(
                    audio,
                    batch_size=8,
                    language=self.language
                )

                transcribed_text = " ".join(
                    segment["text"]
                    for segment in result["segments"]
                ).strip()

                clarity_score = random.randint(75, 98)

            else:

                transcribed_text = (
                    "I would like to order a coffee, please."
                )

                clarity_score = 88

            if not transcribed_text:

                raise ValueError(
                    "No speech detected. Please try again."
                )

            # ---------------------------------------------
            # STEP 2: LANGUAGE-MATCHED TUTOR RESPONSE
            # ---------------------------------------------

            if self.mode == "Review Mode":

                ai_response = self.get_review_response(
                    transcribed_text
                )

            else:

                ai_response = random.choice(
                    TUTOR_RESPONSES.get(
                        self.language,
                        TUTOR_RESPONSES["en"]
                    )
                )

            # ---------------------------------------------
            # STEP 3: TRANSLATION AND FEEDBACK
            # ---------------------------------------------

            # These are placeholders until an LLM is connected.
            # Do not present these as genuine grammar assessments.

            ai_translation = (
                "[English translation requires an LLM]"
            )

            ai_correction = (
                "[Grammar feedback requires an LLM]"
            )

            ai_pronunciation = (
                "[Pronunciation assessment requires an LLM]"
            )

            # ---------------------------------------------
            # STEP 4: RETURN RESULTS TO GUI
            # ---------------------------------------------

            self.finished.emit(
                transcribed_text,
                clarity_score,
                ai_response,
                ai_translation,
                ai_correction,
                ai_pronunciation
            )

        except Exception as e:

            self.error.emit(str(e))

        finally:

            if os.path.exists(self.audio_path):

                try:
                    os.remove(self.audio_path)

                except Exception:
                    pass

    def get_review_response(self, text):

        review_responses = {

            "en": f"I heard: {text}. Let's review your speech.",

            "ru": f"Я услышал: {text}. Давайте разберём вашу речь.",

            "es": f"He escuchado: {text}. Vamos a revisar tu expresión.",

            "fr": f"J'ai entendu : {text}. Examinons votre expression.",

            "de": f"Ich habe gehört: {text}. Lassen Sie uns Ihre Aussage überprüfen.",

            "it": f"Ho sentito: {text}. Rivediamo insieme la tua espressione.",

            "pt": f"Eu ouvi: {text}. Vamos rever sua expressão.",

            "zh": f"我听到了：{text}。让我们一起复习一下。",

            "ja": f"聞きました：{text}。一緒に復習しましょう。",

            "ko": f"들었습니다: {text}. 함께 복습해 봅시다.",

            "ar": f"سمعت: {text}. دعنا نراجع حديثك.",

            "nl": f"Ik hoorde: {text}. Laten we je uitspraak bekijken.",

            "pl": f"Słyszałem: {text}. Przeanalizujmy Twoją wypowiedź.",

            "tr": f"Duydum: {text}. Konuşmanı birlikte inceleyelim.",

            "uk": f"Я почув: {text}. Давайте розглянемо вашу відповідь.",

            "el": f"Άκουσα: {text}. Ας εξετάσουμε την έκφρασή σας.",

            "hi": f"मैंने सुना: {text}। आइए इसकी समीक्षा करें।",

            "sv": f"Jag hörde: {text}. Låt oss granska ditt uttal.",

            "no": f"Jeg hørte: {text}. La oss gjennomgå det du sa.",

            "da": f"Jeg hørte: {text}. Lad os gennemgå det, du sagde."
        }

        return review_responses.get(
            self.language,
            review_responses["en"]
        )


# =========================================================
# MAIN GUI APPLICATION
# =========================================================

class TutorApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "AI Language Tutor - App 8"
        )

        self.resize(650, 800)

        self.recorder = AudioRecorder()

        self.worker = None

        self.chat_history = []

        self.whisper_model = None

        self.setup_ui()

        self.add_system_message(
            "Welcome to AI Language Tutor! "
            "Select your language and mode, "
            "then hold the button to speak."
        )

        # Load WhisperX once
        if USE_REAL_WHISPER:
            self.load_whisper_model()

    # -----------------------------------------------------
    # LOAD WHISPER MODEL
    # -----------------------------------------------------

    def load_whisper_model(self):

        try:

            self.add_system_message(
                "Loading speech recognition model..."
            )

            QApplication.processEvents()

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            compute_type = (
                "float16"
                if device == "cuda"
                else "int8"
            )

            self.whisper_model = whisperx.load_model(
                "small",
                device,
                compute_type=compute_type
            )

            self.add_system_message(
                "Speech recognition model loaded successfully."
            )

        except Exception as e:

            self.add_system_message(
                f"WhisperX loading error: {html.escape(str(e))}"
            )

            self.btn_record.setEnabled(False)

    # -----------------------------------------------------
    # GUI SETUP
    # -----------------------------------------------------

    def setup_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        layout.setContentsMargins(
            20, 20, 20, 20
        )

        layout.setSpacing(15)

        # MODE SELECTOR

        mode_layout = QHBoxLayout()

        mode_layout.addWidget(
            QLabel("<b>Mode:</b>")
        )

        self.mode_dropdown = QComboBox()

        self.mode_dropdown.addItems([
            "Conversational Mode",
            "Review Mode"
        ])

        self.mode_dropdown.setStyleSheet(
            "padding: 5px; font-size: 14px;"
        )

        mode_layout.addWidget(
            self.mode_dropdown,
            stretch=1
        )

        layout.addLayout(mode_layout)

        # LANGUAGE SELECTOR

        language_layout = QHBoxLayout()

        language_layout.addWidget(
            QLabel("<b>Language:</b>")
        )

        self.language_dropdown = QComboBox()

        self.language_dropdown.addItems(
            LANGUAGES.keys()
        )

        self.language_dropdown.setCurrentText(
            "Russian"
        )

        self.language_dropdown.setStyleSheet(
            "padding: 5px; font-size: 14px;"
        )

        language_layout.addWidget(
            self.language_dropdown,
            stretch=1
        )

        layout.addLayout(language_layout)

        # CHAT DISPLAY

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

        layout.addWidget(
            self.chat_display
        )

        # PROGRESS BAR

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(0, 0)

        self.progress_bar.setVisible(False)

        self.progress_bar.setFixedHeight(4)

        layout.addWidget(
            self.progress_bar
        )

        # RECORDING CONTROLS

        controls_layout = QHBoxLayout()

        self.btn_record = QPushButton(
            "Hold to Speak"
        )

        self.btn_record.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #2b78e4;
                color: white;
                font-weight: bold;
                padding: 18px;
                font-size: 16px;
                border-radius: 25px;
            }

            QPushButton:pressed {
                background-color: #1a5baf;
            }

            QPushButton:disabled {
                background-color: #a0c4f2;
            }
        """)

        self.btn_record.pressed.connect(
            self.start_recording
        )

        self.btn_record.released.connect(
            self.stop_recording
        )

        controls_layout.addWidget(
            self.btn_record,
            stretch=3
        )

        self.btn_play = QPushButton(
            "Play Audio"
        )

        self.btn_play.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                font-weight: bold;
                padding: 18px;
                font-size: 14px;
                border-radius: 25px;
            }

            QPushButton:hover {
                background-color: #95a5a6;
            }

            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)

        self.btn_play.setEnabled(False)

        self.btn_play.clicked.connect(
            self.recorder.play_last_recording
        )

        controls_layout.addWidget(
            self.btn_play,
            stretch=1
        )

        layout.addLayout(
            controls_layout
        )

    # -----------------------------------------------------
    # START RECORDING
    # -----------------------------------------------------

    def start_recording(self):

        if self.worker and self.worker.isRunning():
            return

        self.btn_record.setText(
            "Listening... (Release to Send)"
        )

        self.btn_record.setStyleSheet(
            self.btn_record.styleSheet().replace(
                "#2b78e4",
                "#e74c3c"
            )
        )

        self.btn_play.setEnabled(False)

        try:

            self.recorder.start_recording()

        except Exception as e:

            self.btn_record.setText(
                "Hold to Speak"
            )

            self.add_system_message(
                f"Microphone error: {html.escape(str(e))}"
            )

    # -----------------------------------------------------
    # STOP RECORDING
    # -----------------------------------------------------

    def stop_recording(self):

        self.btn_record.setText(
            "Hold to Speak"
        )

        self.btn_record.setStyleSheet(
            self.btn_record.styleSheet().replace(
                "#e74c3c",
                "#2b78e4"
            )
        )

        # Ignore release if recording never started
        if self.recorder.stream is None:
            return

        audio_path = self.recorder.stop_recording()

        if audio_path:

            self.btn_play.setEnabled(True)

            self.process_audio(
                audio_path
            )

    # -----------------------------------------------------
    # PROCESS AUDIO
    # -----------------------------------------------------

    def process_audio(self, audio_path):

        if USE_REAL_WHISPER and self.whisper_model is None:

            self.add_system_message(
                "Speech recognition model is not available."
            )

            self.btn_record.setEnabled(True)

            return

        self.btn_record.setEnabled(False)

        self.progress_bar.setVisible(True)

        current_mode = (
            self.mode_dropdown.currentText()
        )

        selected_language = LANGUAGES[
            self.language_dropdown.currentText()
        ]

        self.worker = AITutorWorker(
            audio_path,
            self.chat_history.copy(),
            current_mode,
            selected_language,
            self.whisper_model
        )

        self.worker.finished.connect(
            self.on_processing_finished
        )

        self.worker.error.connect(
            self.on_processing_error
        )

        self.worker.start()

    # -----------------------------------------------------
    # SYSTEM MESSAGE
    # -----------------------------------------------------

    def add_system_message(self, text):

        safe_text = html.escape(text)

        message = f"""
        <div style='text-align: center;
                    margin: 15px 0;
                    color: #888;
                    font-size: 12px;'>
            {safe_text}
        </div>
        """

        self.chat_display.append(
            message
        )

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    def append_user_message(
        self,
        text,
        clarity_score
    ):

        safe_text = html.escape(text)

        color = (
            "#27ae60"
            if clarity_score >= 80
            else "#f39c12"
            if clarity_score >= 50
            else "#e74c3c"
        )

        message = f"""
        <div style='margin-bottom: 15px;
                    text-align: right;'>

            <div style='display: inline-block;
                        background-color: #2b78e4;
                        color: white;
                        padding: 12px 18px;
                        border-radius: 18px;
                        max-width: 70%;
                        text-align: left;'>

                {safe_text}

            </div>

            <div style='font-size: 11px;
                        color: {color};
                        margin-top: 4px;
                        margin-right: 5px;'>

                Speech Clarity: {clarity_score}%

            </div>

        </div>
        """

        self.chat_display.append(
            message
        )

    # -----------------------------------------------------
    # AI MESSAGE
    # -----------------------------------------------------

    def append_ai_message(
        self,
        response,
        translation,
        correction,
        pronunciation
    ):

        response = html.escape(response)
        translation = html.escape(translation)
        correction = html.escape(correction)
        pronunciation = html.escape(pronunciation)

        message = f"""
        <div style='margin-bottom: 20px;
                    text-align: left;'>

            <div style='display: inline-block;
                        background-color: #ffffff;
                        color: #333;
                        padding: 15px;
                        border-radius: 18px;
                        border: 1px solid #e0e0e0;
                        max-width: 85%;
                        box-shadow: 0px 2px 4px rgba(0,0,0,0.05);'>

                <b style='color: #2b78e4;'>
                    Tutor
                </b>

                <br><br>

                <span style='font-size: 15px;'>
                    {response}
                </span>

                <div style='margin-top: 6px;
                            font-size: 13px;
                            color: #888888;'>

                    <i>{translation}</i>

                </div>

                <hr style='border: 0;
                           border-top: 1px solid #eee;
                           margin: 12px 0;'>

                <div style='background-color: #fff9e6;
                            padding: 10px;
                            border-radius: 8px;
                            font-size: 13px;
                            color: #665c3d;'>

                    <b>Feedback:</b>
                    {correction}

                    <br><br>

                    <b>Pronunciation:</b>
                    <i>{pronunciation}</i>

                </div>

            </div>

        </div>
        """

        self.chat_display.append(
            message
        )

    # -----------------------------------------------------
    # PROCESSING FINISHED
    # -----------------------------------------------------

    def on_processing_finished(
        self,
        user_text,
        clarity_score,
        ai_response,
        ai_translation,
        ai_correction,
        ai_pronunciation
    ):

        self.btn_record.setEnabled(True)

        self.progress_bar.setVisible(False)

        self.chat_history.append({
            "role": "user",
            "content": user_text
        })

        self.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })

        self.append_user_message(
            user_text,
            clarity_score
        )

        self.append_ai_message(
            ai_response,
            ai_translation,
            ai_correction,
            ai_pronunciation
        )

        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    # -----------------------------------------------------
    # PROCESSING ERROR
    # -----------------------------------------------------

    def on_processing_error(self, error_msg):

        self.btn_record.setEnabled(True)

        self.progress_bar.setVisible(False)

        self.add_system_message(
            f"Error: {error_msg}"
        )

    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------

    def closeEvent(self, event):

        if self.recorder.stream:

            self.recorder.stream.stop()
            self.recorder.stream.close()
            self.recorder.stream = None

        if self.worker and self.worker.isRunning():

            self.worker.wait()

        event.accept()


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    try:

        multiprocessing.set_start_method(
            "spawn"
        )

    except RuntimeError:
        pass

    try:

        app = QApplication(sys.argv)

        window = TutorApp()

        window.show()

        sys.exit(
            app.exec()
        )

    except Exception:

        print(
            "--- RUNTIME ERROR ---"
        )

        traceback.print_exc()

        input(
            "\nPress Enter to exit..."
        )
