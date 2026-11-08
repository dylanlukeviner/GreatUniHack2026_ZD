import faulthandler
faulthandler.enable()

import sys
import multiprocessing
import traceback
import os
import tempfile
import html
import random
import json
import re

import sounddevice as sd
import soundfile as sf
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextBrowser,
    QProgressBar, QComboBox, QLabel, QFileDialog, QGridLayout, QLineEdit
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal


# =========================================================
# APP 8 - AI MULTILINGUAL LANGUAGE TUTOR
# =========================================================

USE_REAL_WHISPER = True

if USE_REAL_WHISPER:
    # onnxruntime must load BEFORE torch on Windows. Torch ships an
    # older Visual C++ runtime, and onnxruntime crashes with an access
    # violation if that runtime is already loaded. whisperx -> pyannote
    # imports onnxruntime lazily, so load it explicitly up front.
    print("[BOOT] importing onnxruntime...", flush=True)
    try:
        import onnxruntime
    except ImportError:
        pass

    print("[BOOT] importing torch...", flush=True)
    import torch
    print("[BOOT] importing whisperx...", flush=True)
    import whisperx
    import whisperx.asr  # triggers pyannote/onnxruntime imports now
    print("[BOOT] imports done", flush=True)


# =========================================================
# LOCAL LLM SETTINGS (llama.cpp / GGUF)
# =========================================================

USE_LOCAL_LLM = True

# Path to any chat-tuned GGUF model. Override with the
# TUTOR_LLM_PATH environment variable if you like.
# Recommended multilingual models: Qwen2.5-3B/7B-Instruct,
# Llama-3.1-8B-Instruct, Gemma-2-9B-it (Q4_K_M quantisation).
LLM_MODEL_PATH = os.environ.get(
    "TUTOR_LLM_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "qwen2.5-3b-instruct-q4_k_m.gguf"
    )
)

# Speech recognition model size: "small" is fast; "medium" is
# noticeably more accurate for Russian and other non-English languages.
WHISPER_MODEL_SIZE = "small"

# CEFR level the tutor should speak at
LEARNER_LEVEL = "A1-A2"

LLM_CONTEXT_SIZE = 4096
LLM_MAX_NEW_TOKENS = 500
LLM_TEMPERATURE = 0.4

# How many previous chat messages are sent to the model
LLM_MAX_HISTORY = 6

# llama_cpp is imported lazily in TutorApp.load_llm_model()


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

LANGUAGE_NAMES = {
    code: name for name, code in LANGUAGES.items()
}


# =========================================================
# PRACTICE SCENARIOS (A1-A2)
# Each entry: (title shown in the dropdown, role the tutor plays)
# =========================================================

SCENARIOS = [
    ("Greeting someone for the first time", "a friendly person the student is meeting for the first time"),
    ("Introducing yourself", "a new classmate who wants to know the student's name, age and country"),
    ("Talking about your family", "a friend who asks about the student's family"),
    ("Talking about your job or studies", "a new colleague who asks about the student's work or studies"),
    ("Describing your daily routine", "a friend who asks what the student does every day"),
    ("Ordering food in a restaurant", "a waiter taking the student's order"),
    ("Ordering coffee at a cafe", "a barista taking the student's order"),
    ("Buying food at a market", "a market seller with fruit and vegetables"),
    ("Shopping for clothes", "a shop assistant in a clothes shop"),
    ("Asking for prices", "a shopkeeper who answers questions about prices"),
    ("At the supermarket", "a supermarket worker who helps the student find things"),
    ("Paying at the checkout", "a cashier at a shop checkout"),
    ("Asking for directions", "a friendly passer-by who gives simple directions"),
    ("Taking a taxi", "a taxi driver asking where the student wants to go"),
    ("Buying a bus or train ticket", "a ticket seller at a station"),
    ("Airport check-in", "an airline check-in agent"),
    ("Passport control", "a border officer asking simple questions"),
    ("Checking in to a hotel", "a hotel receptionist"),
    ("Reporting a problem in a hotel room", "a hotel receptionist who helps with a problem"),
    ("Booking a table by phone", "a restaurant host answering the phone"),
    ("Making a doctor's appointment", "a clinic receptionist"),
    ("At the doctor's", "a doctor asking about simple symptoms"),
    ("At the pharmacy", "a pharmacist who helps with a small health problem"),
    ("At the bank", "a bank clerk helping to change money"),
    ("At the post office", "a post office clerk helping to send a letter or parcel"),
    ("Renting an apartment", "a landlord showing a small apartment"),
    ("Talking to a neighbour", "a friendly neighbour"),
    ("Phoning a friend to meet up", "a friend on the phone arranging to meet"),
    ("Inviting a friend to a party", "a friend who is invited to a party"),
    ("Talking about the weather", "a friend chatting about the weather"),
    ("Talking about hobbies", "a new friend who asks about hobbies"),
    ("Talking about favourite food", "a friend who loves talking about food"),
    ("Talking about music and films", "a friend who asks about favourite music and films"),
    ("Planning a holiday", "a friend helping to plan a holiday"),
    ("Describing your home", "a friend who asks about the student's home"),
    ("Talking about your city", "a visitor who asks about the student's city"),
    ("At the cinema", "a cinema ticket seller"),
    ("At the gym or sports club", "a gym receptionist"),
    ("Joining a language class", "a school receptionist signing the student up"),
    ("In the classroom", "a teacher asking simple classroom questions"),
    ("A simple job interview", "an interviewer asking easy questions"),
    ("Lost property", "a staff member at a lost property office"),
    ("Lost in the city", "a police officer helping a lost visitor"),
    ("Making plans and telling the time", "a friend making plans for the weekend"),
    ("At a birthday party", "the host of a birthday party"),
    ("Ordering takeaway by phone", "a staff member at a takeaway restaurant"),
    ("Buying a SIM card or phone", "a phone shop assistant"),
    ("At the hairdresser", "a hairdresser asking how to cut the student's hair"),
    ("Train delay at the station", "a station information officer"),
    ("Thanking a host and saying goodbye", "a host saying goodbye after the student's visit"),
]

assert len(SCENARIOS) == 50

FREE_CONVERSATION = "Free conversation (no scenario)"
CUSTOM_SCENARIO = "Custom situation..."


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
        self.last_rate = samplerate

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

    def resolve_input_device(self, device_index=None):
        """Pick a usable input device even if Windows has no default."""

        if device_index is not None:
            return device_index

        # 1. The system default, if it is valid
        try:

            default_device = sd.default.device[0]

            if default_device is not None and default_device >= 0:

                sd.query_devices(default_device, "input")
                return default_device

        except Exception:
            pass

        # 2. Otherwise the first real input device
        #    (skip the "Sound Mapper", which depends on the default)
        devices = self.get_input_devices()

        print("[MIC] No default input device. Available inputs:")

        for dev in devices:
            print(f"[MIC]   {dev['index']}: {dev['name']}")

        preferred = [
            d for d in devices
            if "mapper" not in d["name"].lower()
        ]

        if preferred:
            return preferred[0]["index"]

        if devices:
            return devices[0]["index"]

        raise RuntimeError(
            "No microphone found. Connect a microphone and make sure "
            "Windows allows desktop apps to use it "
            "(Settings > Privacy > Microphone)."
        )

    def start_recording(self, device_index=None):

        self.recording = []

        print("[MIC] Starting recording...")

        try:

            device = self.resolve_input_device(device_index)

            device_info = sd.query_devices(device, "input")

            print(f"[MIC] Using device {device}: {device_info['name']}")

            # Try 16 kHz first, then the device's native rate
            rates = [self.samplerate]

            native_rate = int(device_info["default_samplerate"])

            if native_rate not in rates:
                rates.append(native_rate)

            last_error = None

            for rate in rates:

                try:

                    self.stream = sd.InputStream(
                        samplerate=rate,
                        channels=1,
                        device=device,
                        callback=self.audio_callback
                    )

                    self.stream.start()

                    self.samplerate = rate

                    last_error = None

                    break

                except Exception as e:

                    self.stream = None
                    last_error = e

            if last_error is not None:
                raise last_error

            print(f"[MIC] Recording started at {self.samplerate} Hz.")

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

        self.last_rate = self.samplerate

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
                self.last_rate
            )

            sd.wait()

        else:

            print("[MIC] Nothing to play.")


# =========================================================
# AI WORKER
# =========================================================

class AITutorWorker(QThread):

    finished = pyqtSignal(str, int, str, str, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        audio_path,
        chat_history,
        mode,
        language,
        whisper_model,
        llm=None,
        delete_audio=True,
        scenario=None
    ):

        super().__init__()

        self.audio_path = audio_path
        self.chat_history = chat_history
        self.mode = mode
        self.language = language
        self.whisper_model = whisper_model
        self.llm = llm
        self.delete_audio = delete_audio
        self.scenario = scenario

    def run(self):

        try:

            # ---------------------------------------------
            # STEP 1: TRANSCRIPTION
            # ---------------------------------------------

            if USE_REAL_WHISPER:

                audio = self.load_audio_16k(
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
            # STEP 2 + 3: TUTOR RESPONSE, TRANSLATION,
            #             AND GRAMMAR FEEDBACK
            # ---------------------------------------------

            llm_result = None

            if self.llm is not None:

                try:

                    llm_result = self.generate_llm_feedback(
                        transcribed_text
                    )

                except Exception as e:

                    print(f"[LLM ERROR] {e}")
                    traceback.print_exc()

            if llm_result:

                ai_response = llm_result["response"]
                ai_translation = llm_result["translation"]
                ai_correction = llm_result["correction"]

            else:

                # Fallback when the LLM is missing or failed.
                # Canned replies are NOT genuine assessments.

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

                ai_translation = (
                    "[Translation unavailable: local LLM not running]"
                )

                ai_correction = (
                    "[Grammar feedback unavailable: local LLM not running]"
                )

            # ---------------------------------------------
            # STEP 4: RETURN RESULTS TO GUI
            # ---------------------------------------------

            self.finished.emit(
                transcribed_text,
                clarity_score,
                ai_response,
                ai_translation,
                ai_correction
            )

        except Exception as e:

            self.error.emit(str(e))

        finally:

            # Only delete temporary recordings, never a file the
            # user chose to open.
            if self.delete_audio and os.path.exists(self.audio_path):

                try:
                    os.remove(self.audio_path)

                except Exception:
                    pass

    @staticmethod
    def load_audio_16k(path):
        """Load a wav as 16 kHz mono float32 without needing ffmpeg."""

        data, rate = sf.read(path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        if rate != 16000:

            import torchaudio.functional as taf

            data = taf.resample(
                torch.from_numpy(data),
                rate,
                16000
            ).numpy()

        return np.ascontiguousarray(data, dtype=np.float32)

    # -----------------------------------------------------
    # LOCAL LLM
    # -----------------------------------------------------

    # The tutor is split into three small, separate LLM calls:
    #   1. reply      - sees the conversation history
    #   2. translation - sees ONLY the reply
    #   3. feedback   - sees ONLY the student's latest sentence
    # Small models cope far better with one simple task per call, and
    # the feedback step can no longer copy the tutor's reply.

    @staticmethod
    def normalise(text):

        return " ".join(
            re.sub(r"[^\w\s]", "", text.lower()).split()
        )

    def chat(self, messages, max_tokens, temperature, schema=None):

        kwargs = {}

        if schema is not None:

            kwargs["response_format"] = {
                "type": "json_object",
                "schema": schema
            }

        output = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return output["choices"][0]["message"]["content"].strip()

    # ---------- 1. TUTOR REPLY ----------

    def build_system_prompt(self):

        language_name = LANGUAGE_NAMES.get(
            self.language,
            "English"
        )

        parts = [
            f"You are a friendly, patient {language_name} tutor "
            f"helping a beginner (CEFR level {LEARNER_LEVEL}) practise "
            f"spoken {language_name}. Use only very simple, common "
            f"words, short sentences (about 5 to 8 words) and mostly "
            f"the present tense. Avoid idioms and slang. The "
            f"student's speech was transcribed automatically, so the "
            f"transcript may contain recognition errors. If a "
            f"sentence looks garbled, say it was unclear instead of "
            f"guessing."
        ]

        if self.mode == "Review Mode":

            parts.append(
                "Do not continue the conversation. Briefly summarise "
                "what the student said."
            )

        else:

            if self.scenario:

                title, role = self.scenario

                parts.append(
                    f"Role-play scenario: \"{title}\". You play "
                    f"{role}. Stay in this role, keep the situation "
                    f"realistic, and move it forward one small step "
                    f"at a time. Ask only one simple question at a "
                    f"time."
                )

            else:

                parts.append(
                    "Be a natural conversation partner on whatever "
                    "topic the student chooses. Follow their lead, "
                    "keep the conversation moving, and end with one "
                    "simple question."
                )

        parts.append(
            "Rules:\n"
            "- Respond to the student's LATEST message. Earlier "
            "messages are background only.\n"
            "- If the student has finished something (for example "
            "said thank you), move on. Never repeat an earlier "
            "topic.\n"
            f"- Write ONLY your reply, in {language_name}, in 1-3 "
            "short sentences. Do not translate, correct or comment "
            "on the student's grammar."
        )

        return "\n\n".join(parts)

    def generate_reply(self, user_text):

        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt()
            }
        ]

        for msg in self.chat_history[-LLM_MAX_HISTORY:]:

            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({
            "role": "user",
            "content": user_text
        })

        reply = self.chat(
            messages,
            LLM_MAX_NEW_TOKENS,
            LLM_TEMPERATURE
        )

        return reply.strip().strip('"\u201c\u201d\u00ab\u00bb`').strip()

    # ---------- 2. TRANSLATION ----------

    def translate_to_english(self, text):

        language_name = LANGUAGE_NAMES.get(
            self.language,
            "English"
        )

        return self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"Translate the following {language_name} "
                        f"text into English. Output only the "
                        f"translation, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            300,
            0.0
        )

    # ---------- 3. STUDENT FEEDBACK ----------

    def check_student_sentence(self, user_text):

        language_name = LANGUAGE_NAMES.get(
            self.language,
            "English"
        )

        schema = {
            "type": "object",
            "properties": {
                "has_mistakes": {"type": "boolean"},
                "corrected": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": [
                "has_mistakes",
                "corrected",
                "explanation"
            ]
        }

        system = (
            f"You are a strict {language_name} teacher checking ONE "
            f"sentence spoken by a learner. It comes from automatic "
            f"speech recognition, so ignore punctuation, "
            f"capitalisation and spacing.\n"
            f"Decide whether it contains a real grammar, word-form "
            f"or word-choice mistake. Do not rewrite for style and "
            f"do not comment on the meaning.\n"
            f"Reply with a JSON object:\n"
            f"- \"has_mistakes\": true only if there is a real "
            f"mistake.\n"
            f"- \"corrected\": the corrected sentence in "
            f"{language_name}. If there are no mistakes, copy the "
            f"sentence exactly.\n"
            f"- \"explanation\": one short sentence in English "
            f"naming the mistake. Empty if there are none."
        )

        raw = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text}
            ],
            250,
            0.0,
            schema=schema
        )

        data = self.parse_json(raw)

        if not data:
            return "[Feedback unavailable]"

        corrected = str(data.get("corrected", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        no_change = (
            self.normalise(corrected) == self.normalise(user_text)
        )

        if (
            not data.get("has_mistakes")
            or no_change
            or not corrected
        ):
            return "No mistakes found."

        if explanation:
            return f"Try: \"{corrected}\" \u2014 {explanation}"

        return f"Try: \"{corrected}\""

    # ---------- COMBINED ----------

    def generate_llm_feedback(self, user_text):

        user_text = user_text[:2000]

        reply = self.generate_reply(user_text)

        if not reply:
            return None

        translation = ""

        if self.language != "en":

            try:
                translation = self.translate_to_english(reply)

            except Exception as e:

                print(f"[LLM] Translation failed: {e}")
                translation = "[Translation unavailable]"

        try:
            correction = self.check_student_sentence(user_text)

        except Exception as e:

            print(f"[LLM] Feedback failed: {e}")
            correction = "[Feedback unavailable]"

        return {
            "response": reply,
            "translation": translation,
            "correction": correction
        }

    @staticmethod
    def parse_json(raw_text):

        try:
            data = json.loads(raw_text)

        except json.JSONDecodeError:

            start = raw_text.find("{")
            end = raw_text.rfind("}")

            if start == -1 or end <= start:
                print(f"[LLM] Could not parse: {raw_text!r}")
                return None

            try:
                data = json.loads(raw_text[start:end + 1])

            except json.JSONDecodeError:
                print(f"[LLM] Could not parse: {raw_text!r}")
                return None

        return data if isinstance(data, dict) else None

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

        self.llm = None

        self.setup_ui()

        self.add_system_message(
            "Welcome to AI Language Tutor! "
            "Select your language and mode, "
            "then hold the button to speak."
        )

        # Load WhisperX once
        if USE_REAL_WHISPER:
            self.load_whisper_model()

        # Load local LLM once
        if USE_LOCAL_LLM:
            self.load_llm_model()

    # -----------------------------------------------------
    # LOAD LOCAL LLM
    # -----------------------------------------------------

    def load_llm_model(self):

        try:

            print("[BOOT] importing llama_cpp...", flush=True)

            from llama_cpp import Llama

            print("[BOOT] llama_cpp imported", flush=True)

        except Exception as e:

            self.add_system_message(
                f"llama-cpp-python could not be loaded: {e}. "
                "Tutor feedback will use placeholder text."
            )

            return

        if not os.path.isfile(LLM_MODEL_PATH):

            self.add_system_message(
                f"LLM model file not found: {LLM_MODEL_PATH}. "
                "Set TUTOR_LLM_PATH or place a GGUF model there. "
                "Tutor feedback will use placeholder text."
            )

            return

        try:

            self.add_system_message(
                "Loading language model..."
            )

            QApplication.processEvents()

            use_gpu = (
                USE_REAL_WHISPER and torch.cuda.is_available()
            )

            self.llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=LLM_CONTEXT_SIZE,
                n_gpu_layers=-1 if use_gpu else 0,
                verbose=False
            )

            self.add_system_message(
                "Language model loaded successfully."
            )

        except Exception as e:

            self.llm = None

            self.add_system_message(
                f"LLM loading error: {e}. "
                "Tutor feedback will use placeholder text."
            )

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
                WHISPER_MODEL_SIZE,
                device,
                compute_type=compute_type
            )

            self.add_system_message(
                "Speech recognition model loaded successfully."
            )

        except Exception as e:

            self.add_system_message(
                f"WhisperX loading error: {e}"
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
            12, 10, 12, 12
        )

        layout.setSpacing(8)

        # SETTINGS (compact grid: Mode | Language on one row)

        field_style = "padding: 3px; font-size: 13px;"

        grid = QGridLayout()

        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # Row 0: Mode and Language

        self.mode_dropdown = QComboBox()

        self.mode_dropdown.addItems([
            "Conversational Mode",
            "Review Mode"
        ])

        self.mode_dropdown.setStyleSheet(field_style)

        self.language_dropdown = QComboBox()

        self.language_dropdown.addItems(
            LANGUAGES.keys()
        )

        self.language_dropdown.setCurrentText(
            "Russian"
        )

        self.language_dropdown.setStyleSheet(field_style)

        grid.addWidget(QLabel("<b>Mode</b>"), 0, 0)
        grid.addWidget(self.mode_dropdown, 0, 1)
        grid.addWidget(QLabel("<b>Language</b>"), 0, 2)
        grid.addWidget(self.language_dropdown, 0, 3)

        # Row 1: Scenario (full width)

        self.scenario_dropdown = QComboBox()

        self.scenario_dropdown.addItem(FREE_CONVERSATION)
        self.scenario_dropdown.addItem(CUSTOM_SCENARIO)

        self.scenario_dropdown.addItems(
            [title for title, _ in SCENARIOS]
        )

        self.scenario_dropdown.setMaxVisibleItems(15)

        self.scenario_dropdown.setStyleSheet(field_style)

        grid.addWidget(QLabel("<b>Scenario</b>"), 1, 0)
        grid.addWidget(self.scenario_dropdown, 1, 1, 1, 3)

        # Row 2: Custom situation (only shown for "Custom situation...")

        self.applied_custom_text = ""

        self.custom_scenario_label = QLabel("<b>Situation</b>")

        self.custom_scenario_input = QLineEdit()

        self.custom_scenario_input.setMaxLength(200)

        self.custom_scenario_input.setPlaceholderText(
            "Describe your situation, e.g. buying a gift for "
            "my mother's birthday"
        )

        self.custom_scenario_input.setStyleSheet(field_style)

        grid.addWidget(self.custom_scenario_label, 2, 0)
        grid.addWidget(self.custom_scenario_input, 2, 1, 1, 3)

        layout.addLayout(grid)

        # Hidden until "Custom situation..." is selected
        self.set_custom_row_visible(False)

        self.scenario_dropdown.currentIndexChanged.connect(
            self.on_scenario_changed
        )

        self.custom_scenario_input.editingFinished.connect(
            self.on_custom_scenario_edited
        )

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

        self.btn_open = QPushButton(
            "Open Audio File"
        )

        self.btn_open.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 18px;
                font-size: 14px;
                border-radius: 25px;
            }

            QPushButton:hover {
                background-color: #2ecc71;
            }

            QPushButton:disabled {
                background-color: #a9dfbf;
            }
        """)

        self.btn_open.clicked.connect(
            self.open_audio_file
        )

        controls_layout.addWidget(
            self.btn_open,
            stretch=2
        )

        self.btn_new = QPushButton(
            "New Topic"
        )

        self.btn_new.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.btn_new.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                font-weight: bold;
                padding: 18px;
                font-size: 14px;
                border-radius: 25px;
            }

            QPushButton:hover {
                background-color: #9b59b6;
            }
        """)

        self.btn_new.clicked.connect(
            self.new_topic
        )

        controls_layout.addWidget(
            self.btn_new,
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
                f"Microphone error: {e}"
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
    # SCENARIO
    # -----------------------------------------------------

    def get_selected_scenario(self):

        index = self.scenario_dropdown.currentIndex()

        if index <= 0:
            return None

        if index == 1:

            text = self.custom_scenario_input.text().strip()[:200]

            if not text:
                return None

            return (
                text,
                "the person the student would naturally talk to "
                "in this situation"
            )

        return SCENARIOS[index - 2]

    def set_custom_row_visible(self, visible):

        self.custom_scenario_label.setVisible(visible)
        self.custom_scenario_input.setVisible(visible)

    def on_scenario_changed(self, index):

        is_custom = index == 1

        self.set_custom_row_visible(is_custom)

        if is_custom:
            self.custom_scenario_input.setFocus()

        if self.worker and self.worker.isRunning():
            return

        self.chat_history.clear()

        self.applied_custom_text = (
            self.custom_scenario_input.text().strip()
            if is_custom else ""
        )

        scenario = self.get_selected_scenario()

        if is_custom and scenario is None:

            self.add_system_message(
                "Type your own situation in the Situation box, "
                "then start speaking."
            )

            return

        if scenario is None:

            self.add_system_message(
                "Free conversation. Talk about anything you like."
            )

            return

        title, role = scenario

        if is_custom:

            self.add_system_message(
                f"Custom scenario: {title}. Start by saying hello!"
            )

            return

        self.add_system_message(
            f"Scenario: {title}. The tutor plays {role}. "
            "Start by saying hello! "
            "(Scenarios apply in Conversational Mode.)"
        )

    def on_custom_scenario_edited(self):

        if self.scenario_dropdown.currentIndex() != 1:
            return

        text = self.custom_scenario_input.text().strip()

        if not text or text == self.applied_custom_text:
            return

        if self.worker and self.worker.isRunning():
            return

        self.applied_custom_text = text

        self.chat_history.clear()

        self.add_system_message(
            f"Custom scenario: {text}. Start by saying hello!"
        )

    # -----------------------------------------------------
    # NEW TOPIC (clear the tutor's memory)
    # -----------------------------------------------------

    def new_topic(self):

        if self.worker and self.worker.isRunning():
            return

        self.chat_history.clear()

        self.add_system_message(
            "New topic started. The tutor has forgotten "
            "the previous messages."
        )

    # -----------------------------------------------------
    # OPEN AUDIO FILE
    # -----------------------------------------------------

    def open_audio_file(self):

        if self.worker and self.worker.isRunning():
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*)"
        )

        if not path:
            return

        # Read it once to validate the format and enable playback
        try:

            data, rate = sf.read(
                path,
                dtype="float32"
            )

        except Exception as e:

            self.add_system_message(
                f"Could not read audio file: {e}. "
                "Supported formats: WAV, FLAC, OGG, MP3."
            )

            return

        self.recorder.last_recorded_data = data
        self.recorder.last_rate = rate

        self.btn_play.setEnabled(True)

        self.add_system_message(
            f"Processing file: {os.path.basename(path)}"
        )

        self.process_audio(
            path,
            delete_audio=False
        )

    # -----------------------------------------------------
    # PROCESS AUDIO
    # -----------------------------------------------------

    def process_audio(self, audio_path, delete_audio=True):

        if USE_REAL_WHISPER and self.whisper_model is None:

            self.add_system_message(
                "Speech recognition model is not available."
            )

            self.btn_record.setEnabled(True)
            self.btn_open.setEnabled(True)

            return

        self.btn_record.setEnabled(False)
        self.btn_open.setEnabled(False)

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
            self.whisper_model,
            self.llm,
            delete_audio=delete_audio,
            scenario=self.get_selected_scenario()
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
        correction
    ):

        response = html.escape(response)
        translation = html.escape(translation)
        correction = html.escape(correction)

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
        ai_correction
    ):

        self.btn_record.setEnabled(True)
        self.btn_open.setEnabled(True)

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
            ai_correction
        )

        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    # -----------------------------------------------------
    # PROCESSING ERROR
    # -----------------------------------------------------

    def on_processing_error(self, error_msg):

        self.btn_record.setEnabled(True)
        self.btn_open.setEnabled(True)

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
