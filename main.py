"""
================================================================================
BENGALI PERSONAL AI VOICE ASSISTANT — Single-File Standalone Script
================================================================================
Features:
  1. Continuous wake-word detection ("hey jarvis") via Porcupine OR fallback
     to push-to-talk / VAD-based continuous listening.
  2. Bengali Speech-to-Text (Whisper API, with Google Speech Recognition fallback).
  3. LLM brain using DeepSeek API (OpenAI-compatible) with Gemini fallback.
  4. Bengali Text-to-Speech via edge-tts (with gTTS + pyttsx3 fallbacks).
  5. System automation: open apps, Google search, YouTube, WhatsApp, Facebook,
     calculator, notepad, time/date, volume, shutdown, etc.

Setup:
  pip install pvporcupine pvrecorder openai edge-tts gtts pyttsx3 sounddevice
              numpy SpeechRecognition pydub requests python-dotenv

Environment variables (set before running):
  PICOVOICE_ACCESS_KEY  — from https://console.picovoice.ai/
  DEEPSEEK_API_KEY      — from https://platform.deepseek.com/
  OPENAI_API_KEY        — (optional) for Whisper STT
  GEMINI_API_KEY        — (optional) fallback LLM

Run:
  python main.py
================================================================================
"""

import os
import sys
import io
import re
import time
import wave
import json
import queue
import shutil
import platform
import tempfile
import threading
import subprocess
import webbrowser
import urllib.parse
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------ #
# Optional dependency imports with graceful fallbacks
# ------------------------------------------------------------------ #
try:
    import numpy as np
except ImportError:
    np = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import pvporcupine
    from pvrecorder import PvRecorder
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import asyncio
except ImportError:
    asyncio = None


# ================================================================== #
#                       CONFIGURATION SECTION                        #
# ================================================================== #

class Config:
    """Central configuration loaded from environment."""

    # API Keys
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
    DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "").strip()
    OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "").strip()
    GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "").strip()

    # Wake word
    WAKE_KEYWORD         = "jarvis"          # built-in keyword
    WAKE_SENSITIVITY     = 0.55

    # Audio
    SAMPLE_RATE          = 16000
    FRAME_LENGTH         = 512
    RECORD_SECONDS       = 6                 # max record duration per turn
    SILENCE_THRESHOLD    = 500               # RMS threshold for VAD fallback

    # TTS
    BENGALI_VOICE        = "bn-IN-TanishaaNeural"
    TTS_RATE             = "+0%"
    TTS_VOLUME           = "+0%"

    # LLM
    DEEPSEEK_BASE_URL    = "https://api.deepseek.com"
    DEEPSEEK_MODEL       = "deepseek-chat"
    GEMINI_MODEL         = "gemini-1.5-flash"
    GEMINI_URL           = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={key}"
    )

    # Language for STT
    STT_LANGUAGE         = "bn-BD"

    # System prompt
    SYSTEM_PROMPT = (
        "You are a helpful, friendly personal AI assistant named Jarvis. "
        "Always respond ONLY in natural, fluent Bengali (বাংলা). "
        "Keep answers concise, warm, and conversational — no more than 2-3 sentences "
        "unless the user explicitly asks for detail. "
        "If the user asks you to perform a computer action (open app, search, etc.), "
        "confirm briefly in Bengali what you are doing."
    )


# ================================================================== #
#                          LOGGER UTILITY                            #
# ================================================================== #

class Log:
    """Simple colored console logger."""

    COLORS = {
        "info":    "\033[96m",   # cyan
        "ok":      "\033[92m",   # green
        "warn":    "\033[93m",   # yellow
        "error":   "\033[91m",   # red
        "assist":  "\033[95m",   # magenta
        "user":    "\033[94m",   # blue
        "reset":   "\033[0m",
    }

    @classmethod
    def _p(cls, tag, color, msg):
        print(f"{cls.COLORS[color]}[{tag}]{cls.COLORS['reset']} {msg}", flush=True)

    @classmethod
    def info(cls, m):   cls._p("INFO",   "info",   m)
    @classmethod
    def ok(cls, m):     cls._p("OK",     "ok",     m)
    @classmethod
    def warn(cls, m):   cls._p("WARN",   "warn",   m)
    @classmethod
    def error(cls, m):  cls._p("ERROR",  "error",  m)
    @classmethod
    def assist(cls, m): cls._p("JARVIS", "assist", m)
    @classmethod
    def user(cls, m):   cls._p("USER",   "user",   m)


# ================================================================== #
#                     WAKE WORD DETECTOR                             #
# ================================================================== #

class WakeWordDetector:
    """
    Two modes:
      1. Porcupine (preferred) — ultra-low-latency on-device wake word.
      2. Fallback: VAD-based push-to-talk using sounddevice + energy threshold.
    """

    def __init__(self):
        self.mode = None
        self.porcupine = None
        self.recorder = None

        if PORCUPINE_AVAILABLE and Config.PICOVOICE_ACCESS_KEY:
            try:
                self.porcupine = pvporcupine.create(
                    access_key=Config.PICOVOICE_ACCESS_KEY,
                    keywords=[Config.WAKE_KEYWORD],
                    sensitivities=[Config.WAKE_SENSITIVITY],
                )
                self.recorder = PvRecorder(frame_length=self.porcupine.frame_length)
                self.recorder.start()
                self.mode = "porcupine"
                Log.ok(f"Wake word engine: Porcupine (keyword='{Config.WAKE_KEYWORD}')")
            except Exception as e:
                Log.warn(f"Porcupine init failed: {e}. Falling back to VAD mode.")
                self.mode = "vad"
        else:
            if not PORCUPINE_AVAILABLE:
                Log.warn("pvporcupine not installed. Using VAD-based fallback.")
            else:
                Log.warn("PICOVOICE_ACCESS_KEY missing. Using VAD-based fallback.")
            self.mode = "vad"

    def wait(self):
        """Block until wake word (or any speech in VAD mode) is detected."""
        if self.mode == "porcupine":
            return self._wait_porcupine()
        return self._wait_vad()

    def _wait_porcupine(self):
        while True:
            try:
                pcm = self.recorder.read()
                idx = self.porcupine.process(pcm)
                if idx >= 0:
                    return True
            except Exception as e:
                Log.error(f"Porcupine read error: {e}")
                time.sleep(0.5)

    def _wait_vad(self):
        """Fallback: wait for any loud-enough audio to trigger."""
        if sd is None or np is None:
            Log.error("sounddevice/numpy missing; cannot run fallback VAD.")
            time.sleep(2)
            return False
        try:
            with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=Config.FRAME_LENGTH,
            ) as stream:
                while True:
                    data, _ = stream.read(Config.FRAME_LENGTH)
                    rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                    if rms > Config.SILENCE_THRESHOLD:
                        return True
        except Exception as e:
            Log.error(f"VAD error: {e}")
            time.sleep(1)
            return False

    def cleanup(self):
        try:
            if self.recorder:
                self.recorder.stop()
                self.recorder.delete()
            if self.porcupine:
                self.porcupine.delete()
        except Exception:
            pass


# ================================================================== #
#                     SPEECH-TO-TEXT (STT)                           #
# ================================================================== #

class SpeechToText:
    """
    Bengali STT with priority chain:
      1. OpenAI Whisper API (highest accuracy)
      2. Google Speech Recognition (free, offline-ish network call)
      3. SpeechRecognition default
    """

    def __init__(self):
        self.whisper_client = None
        if OpenAI and Config.OPENAI_API_KEY:
            try:
                self.whisper_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                Log.ok("STT backend: Whisper API")
            except Exception as e:
                Log.warn(f"Whisper init failed: {e}")

        self.recognizer = sr.Recognizer() if sr else None
        if not self.whisper_client and self.recognizer:
            Log.ok("STT backend: Google Speech Recognition (fallback)")

        if not self.whisper_client and not self.recognizer:
            Log.error("No STT backend available! Install 'openai' or 'SpeechRecognition'.")

    # ------------- recording ------------- #

    def record_audio(self, duration=Config.RECORD_SECONDS):
        """Record mono int16 audio. Returns numpy array."""
        if sd is None or np is None:
            Log.error("sounddevice/numpy not installed; cannot record.")
            return None
        try:
            Log.info(f"Recording for up to {duration}s...")
            frames = sd.rec(
                int(duration * Config.SAMPLE_RATE),
                samplerate=Config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            return frames
        except Exception as e:
            Log.error(f"Recording failed: {e}")
            return None

    def record_until_silence(self, max_seconds=Config.RECORD_SECONDS,
                             silence_after=1.2, threshold=Config.SILENCE_THRESHOLD):
        """Record until silence detected or timeout. Adaptive end-pointing."""
        if sd is None or np is None:
            return None

        Log.info("Listening... (speak now)")
        buffer = []
        silent_frames = 0
        silence_frames_needed = int(
            silence_after * Config.SAMPLE_RATE / Config.FRAME_LENGTH
        )
        started = time.time()
        speech_started = False

        try:
            with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=Config.FRAME_LENGTH,
            ) as stream:
                while time.time() - started < max_seconds:
                    data, _ = stream.read(Config.FRAME_LENGTH)
                    buffer.append(data.copy())
                    rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

                    if rms > threshold:
                        speech_started = True
                        silent_frames = 0
                    elif speech_started:
                        silent_frames += 1
                        if silent_frames >= silence_frames_needed:
                            break
        except Exception as e:
            Log.error(f"Stream error: {e}")
            return None

        if not buffer:
            return None

        # Small tail of silence for context
        for _ in range(int(0.3 * Config.SAMPLE_RATE / Config.FRAME_LENGTH)):
            buffer.append(np.zeros((Config.FRAME_LENGTH, 1), dtype=np.int16))

        return np.concatenate(buffer, axis=0)

    # ------------- transcription ------------- #

    def _write_wav(self, frames, path):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(frames.tobytes())

    def transcribe(self, frames):
        """Convert audio frames to Bengali text."""
        if frames is None or len(frames) == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        self._write_wav(frames, wav_path)

        try:
            # Priority 1: Whisper
            if self.whisper_client:
                try:
                    with open(wav_path, "rb") as af:
                        result = self.whisper_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=af,
                            language="bn",
                        )
                    text = (result.text or "").strip()
                    if text:
                        return text
                except Exception as e:
                    Log.warn(f"Whisper API failed: {e}; trying Google...")

            # Priority 2: Google SR
            if self.recognizer:
                try:
                    with sr.AudioFile(wav_path) as source:
                        audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(
                        audio_data, language=Config.STT_LANGUAGE
                    )
                    return (text or "").strip()
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError as e:
                    Log.warn(f"Google SR unavailable: {e}")
                    return ""
                except Exception as e:
                    Log.warn(f"Google SR error: {e}")
                    return ""

            return ""
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass


# ================================================================== #
#                     LLM BRAIN (DeepSeek / Gemini)                  #
# ================================================================== #

class Brain:
    """
    LLM brain with priority chain:
      1. DeepSeek (OpenAI-compatible)
      2. Google Gemini REST API
      3. Rule-based fallback (returns a canned Bengali reply)
    """

    def __init__(self):
        self.client = None
        self.provider = None
        self.history = [{"role": "system", "content": Config.SYSTEM_PROMPT}]

        if OpenAI and Config.DEEPSEEK_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=Config.DEEPSEEK_API_KEY,
                    base_url=Config.DEEPSEEK_BASE_URL,
                )
                self.provider = "deepseek"
                Log.ok(f"LLM backend: DeepSeek ({Config.DEEPSEEK_MODEL})")
            except Exception as e:
                Log.warn(f"DeepSeek init failed: {e}")

        if self.provider is None and Config.GEMINI_API_KEY:
            self.provider = "gemini"
            Log.ok(f"LLM backend: Gemini ({Config.GEMINI_MODEL})")

        if self.provider is None:
            Log.warn("No LLM API key set. Using rule-based fallback.")

    def think(self, user_text):
        if not user_text or not user_text.strip():
            return "দুঃখিত, আমি কিছু বুঝতে পারিনি।"

        try:
            if self.provider == "deepseek":
                return self._ask_deepseek(user_text)
            if self.provider == "gemini":
                return self._ask_gemini(user_text)
        except Exception as e:
            Log.error(f"LLM error: {e}")

        return self._fallback_reply(user_text)

    def _ask_deepseek(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        # Trim history
        if len(self.history) > 21:
            self.history = [self.history[0]] + self.history[-20:]

        resp = self.client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=self.history,
            stream=False,
            temperature=0.7,
            max_tokens=512,
        )
        reply = resp.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _ask_gemini(self, user_text):
        import urllib.request
        url = Config.GEMINI_URL.format(
            model=Config.GEMINI_MODEL, key=Config.GEMINI_API_KEY
        )
        # Build simple contents
        contents = []
        for m in self.history:
            role = "user" if m["role"] == "user" else "model"
            if m["role"] == "system":
                continue
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        body = {
            "systemInstruction": {"parts": [{"text": Config.SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))

        try:
            reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            reply = "দুঃখিত, আমি উত্তর তৈরি করতে পারিনি।"

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _fallback_reply(self, user_text):
        t = user_text.lower()
        if any(w in t for w in ["সময়", "time", "কতটা বাজে"]):
            return f"এখন সময় {datetime.now().strftime('%I:%M %p')}।"
        if any(w in t for w in ["তারিখ", "date", "আজ কি বার"]):
            return f"আজ {datetime.now().strftime('%A, %d %B %Y')}।"
        if any(w in t for w in ["হ্যালো", "hello", "hi", "নমস্কার"]):
            return "নমস্কার! আমি জারভিস, আপনার সহকারী।"
        return "আমি এই মুহূর্তে একটি এপিআই কী ছাড়া চলছি, তাই সীমিত উত্তর দিতে পারি।"

    def reset(self):
        self.history = [{"role": "system", "content": Config.SYSTEM_PROMPT}]


# ================================================================== #
#                     TEXT-TO-SPEECH (TTS)                           #
# ================================================================== #

class TextToSpeech:
    """
    Bengali TTS with priority chain:
      1. edge-tts (best quality)
      2. gTTS (online, decent)
      3. pyttsx3 (offline, robotic — but always works)
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        if pyttsx3 is not None:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None

        if edge_tts is not None:
            Log.ok("TTS backend: edge-tts")
        elif gTTS is not None:
            Log.ok("TTS backend: gTTS")
        elif self.engine is not None:
            Log.ok("TTS backend: pyttsx3 (offline)")

    def speak(self, text):
        if not text:
            return
        Log.assist(text)
        with self.lock:
            if self._try_edge(text):
                return
            if self._try_gtts(text):
                return
            self._pyttsx3(text)

    # ------------- edge-tts ------------- #

    def _try_edge(self, text):
        if edge_tts is None:
            return False
        try:
            out = os.path.join(
                tempfile.gettempdir(),
                f"jarvis_tts_{abs(hash(text)) % 10**10}.mp3",
            )
            async def _save():
                comm = edge_tts.Communicate(
                    text,
                    Config.BENGALI_VOICE,
                    rate=Config.TTS_RATE,
                    volume=Config.TTS_VOLUME,
                )
                await comm.save(out)
            asyncio.run(_save())
            if os.path.exists(out) and os.path.getsize(out) > 0:
                self._play(out)
                return True
        except Exception as e:
            Log.warn(f"edge-tts failed: {e}")
        return False

    # ------------- gTTS ------------- #

    def _try_gtts(self, text):
        if gTTS is None:
            return False
        try:
            out = os.path.join(
                tempfile.gettempdir(),
                f"jarvis_gtts_{abs(hash(text)) % 10**10}.mp3",
            )
            gTTS(text=text, lang="bn").save(out)
            if os.path.exists(out) and os.path.getsize(out) > 0:
                self._play(out)
                return True
        except Exception as e:
            Log.warn(f"gTTS failed: {e}")
        return False

    # ------------- pyttsx3 (offline) ------------- #

    def _pyttsx3(self, text):
        if self.engine is None:
            Log.error("All TTS backends failed.")
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            Log.error(f"pyttsx3 failed: {e}")

    # ------------- audio playback ------------- #

    def _play(self, path):
        system = platform.system()
        try:
            if system == "Windows":
                # Try PowerShell SoundPlayer first (fast, no external dep)
                try:
                    subprocess.run(
                        ["powershell", "-c",
                         f'(New-Object Media.SoundPlayer "{path}").PlaySync()'],
                        capture_output=True, timeout=60,
                    )
                    return
                except Exception:
                    pass
                # Fallback to ffplay
                if shutil.which("ffplay"):
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                        capture_output=True,
                    )
            elif system == "Darwin":
                subprocess.run(["afplay", path], capture_output=True)
            else:  # Linux / Android
                for player in (["mpv", "--no-video", "--really-quiet", path],
                               ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                               ["aplay", path]):
                    if shutil.which(player[0]):
                        subprocess.run(player, capture_output=True)
                        return
                Log.warn("No audio player found (install mpv/ffplay/aplay).")
        except Exception as e:
            Log.error(f"Playback failed: {e}")


# ================================================================== #
#                     SYSTEM AUTOMATION                              #
# ================================================================== #

class Automator:
    """
    Bengali → system action parser.
    Handles: app opening, web search, YouTube, time/date, volume, etc.
    """

    def __init__(self):
        self.system = platform.system()

    def execute(self, text):
        """Return a Bengali status string if handled, else None."""
        if not text:
            return None

        t = text.lower().strip()

        # --- Time / Date --- #
        if any(k in t for k in ["সময় কত", "কতটা বাজে", "what time", "time koto"]):
            now = datetime.now().strftime("%I:%M %p")
            return f"এখন সময় {now}।"

        if any(k in t for k in ["আজকের তারিখ", "তারিখ কি", "what date", "আজ কি বার"]):
            d = datetime.now().strftime("%A, %d %B %Y")
            return f"আজ {d}।"

        # --- Open by URL keyword --- #
        if any(k in t for k in ["ইউটিউব", "youtube"]):
            if any(k in t for k in ["খুল", "ওপেন", "open", "চালু", "নিয়ে চলো"]):
                webbrowser.open("https://www.youtube.com")
                return "ইউটিউব খুলে দিচ্ছি।"

        if any(k in t for k in ["ফেসবুক", "facebook"]):
            if any(k in t for k in ["খুল", "ওপেন", "open", "চালু"]):
                webbrowser.open("https://www.facebook.com")
                return "ফেসবুক খুলে দিচ্ছি।"

        if any(k in t for k in ["হোয়াটসঅ্যাপ", "হোয়াটসঅ্যাপ", "whatsapp"]):
            if any(k in t for k in ["খুল", "ওপেন", "open", "চালু"]):
                webbrowser.open("https://web.whatsapp.com")
                return "হোয়াটসঅ্যাপ ওয়েব খুলে দিচ্ছি।"

        if any(k in t for k in ["জিমেইল", "gmail", "ইমেইল"]):
            if any(k in t for k in ["খুল", "ওপেন", "open", "চালু"]):
                webbrowser.open("https://mail.google.com")
                return "জিমেইল খুলে দিচ্ছি।"

        # --- Google search --- #
        m = re.search(
            r"(?:গুগলে|google(?:-এ)?|সার্চ\s+করো|খোঁজো|খুঁজে\s+দাও|search)"
            r"\s*(?:করো|করে|for)?\s*(.+)",
            t,
        )
        if m:
            query = m.group(1).strip(" ?।.,")
            if query:
                webbrowser.open(
                    "https://www.google.com/search?q=" + urllib.parse.quote(query)
                )
                return f"গুগলে '{query}' সার্চ করছি।"

        # --- YouTube search --- #
        m = re.search(
            r"(?:ইউটিউবে|youtube(?:-এ)?)\s*(.+?)\s*(?:সার্চ|খোঁজো|দেখাও|প্লে)",
            t,
        )
        if m:
            query = m.group(1).strip(" ?।.,")
            webbrowser.open(
                "https://www.youtube.com/results?search_query="
                + urllib.parse.quote(query)
            )
            return f"ইউটিউবে '{query}' খুঁজছি।"

        # --- Open generic app --- #
        m = re.search(
            r"(?:খুলে\s*দাও|ওপেন\s*করো|চালু\s*করো|open|launch|start)\s+(.+)",
            t,
        )
        if m:
            app = m.group(1).strip(" ?।.,")
            return self._open_app(app)

        # --- Volume / shutdown / screenshot --- #
        if "স্ক্রিনশট" in t or "screenshot" in t:
            return self._screenshot()

        if "শাটডাউন" in t or "shutdown" in t or "বন্ধ করো কম্পিউটার" in t:
            return self._shutdown()

        return None

    # ------------- app launching ------------- #

    def _open_app(self, app):
        a = app.lower().strip()

        # Known web apps
        web_map = {
            "youtube": "https://www.youtube.com",
            "facebook": "https://www.facebook.com",
            "whatsapp": "https://web.whatsapp.com",
            "gmail": "https://mail.google.com",
            "google": "https://www.google.com",
            "chatgpt": "https://chat.openai.com",
            "github": "https://github.com",
            "stackoverflow": "https://stackoverflow.com",
        }
        for key, url in web_map.items():
            if key in a:
                webbrowser.open(url)
                return f"{key} খুলে দিচ্ছি।"

        # Native app map
        native_map = {
            "notepad": ["notepad"],
            "নোটপ্যাড": ["notepad"],
            "calculator": ["calc"],
            "ক্যালকুলেটর": ["calc"],
            "paint": ["mspaint"],
            "cmd": ["cmd"],
            "terminal": ["cmd"] if self.system == "Windows" else ["x-terminal-emulator"],
            "explorer": ["explorer"],
            "ফাইল": ["explorer"],
            "chrome": ["chrome"],
            "edge": ["msedge"],
            "firefox": ["firefox"],
            "vscode": ["code"],
            "spotify": ["spotify"],
        }

        for key, cmd in native_map.items():
            if key in a:
                try:
                    if self.system == "Darwin":
                        subprocess.Popen(["open", "-a", key])
                    else:
                        subprocess.Popen(cmd, shell=(self.system == "Windows"))
                    return f"{key} খুলে দিচ্ছি।"
                except Exception as e:
                    Log.warn(f"open {key} failed: {e}")
                    return f"দুঃখিত, {key} খুলতে পারছি না।"

        # Unknown — try direct launch
        try:
            if self.system == "Windows":
                subprocess.Popen(f'start "" "{a}"', shell=True)
            elif self.system == "Darwin":
                subprocess.Popen(["open", "-a", a])
            else:
                subprocess.Popen([a])
            return f"{a} খোলার চেষ্টা করছি।"
        except Exception as e:
            Log.warn(f"open '{a}' failed: {e}")
            return f"দুঃখিত, '{a}' নামে কোনো অ্যাপ খুঁজে পাইনি।"

    # ------------- system actions ------------- #

    def _screenshot(self):
        try:
            import pyautogui  # optional
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                os.path.expanduser("~"), f"screenshot_{ts}.png"
            )
            pyautogui.screenshot(path)
            return f"স্ক্রিনশট নিয়ে সংরক্ষণ করেছি: {path}"
        except ImportError:
            return "স্ক্রিনশট নিতে 'pyautogui' ইনস্টল করুন।"
        except Exception as e:
            return f"স্ক্রিনশট ব্যর্থ: {e}"

    def _shutdown(self):
        # Safety: require explicit confirm via keyword presence; cancel scheduling
        try:
            if self.system == "Windows":
                subprocess.Popen(["shutdown", "/s", "/t", "20"])
            elif self.system == "Darwin":
                subprocess.Popen(["sudo", "shutdown", "-h", "+1"])
            else:
                subprocess.Popen(["shutdown", "-h", "+1"])
            return "২০ সেকেন্ডের মধ্যে কম্পিউটার বন্ধ হবে। বাতিল করতে 'shutdown /a' চালান।"
        except Exception as e:
            return f"শাটডাউন ব্যর্থ: {e}"


# ================================================================== #
#                     MAIN ORCHESTRATOR                              #
# ================================================================== #

class Assistant:
    """Wires every component together into the live loop."""

    def __init__(self):
        Log.info("=" * 60)
        Log.info("  বাংলা পার্সোনাল এআই ভয়েস অ্যাসিস্ট্যান্ট — Jarvis")
        Log.info("=" * 60)

        self.wake = WakeWordDetector()
        self.stt = SpeechToText()
        self.brain = Brain()
        self.tts = TextToSpeech()
        self.automator = Automator()

        self.running = True

    def greet(self):
        hour = datetime.now().hour
        if hour < 12:
            greet = "শুভ সকাল।"
        elif hour < 17:
            greet = "শুভ অপরাহ্ন।"
        elif hour < 20:
            greet = "শুভ সন্ধ্যা।"
        else:
            greet = "শুভ রাত্রি।"
        self.tts.speak(f"{greet} আমি জারভিস, আপনার সহকারী। বলুন, কীভাবে সাহায্য করতে পারি?")

    def run(self):
        self.greet()
        Log.info("Say the wake word to activate. Press Ctrl+C to stop.\n")

        while self.running:
            try:
                # 1. Wait for wake word
                triggered = self.wake.wait()
                if not triggered:
                    continue

                Log.ok("Wake word detected!")
                self.tts.speak("হ্যাঁ, আমি শুনছি।")

                # 2. Record + transcribe
                frames = self.stt.record_until_silence()
                if frames is None or len(frames) == 0:
                    self.tts.speak("দুঃখিত, কিছু শুনতে পাইনি।")
                    continue

                text = self.stt.transcribe(frames)
                if not text:
                    self.tts.speak("দুঃখিত, আমি বুঝতে পারিনি। আবার বলুন।")
                    continue

                Log.user(text)

                # 3. Try automation first
                auto_reply = self.automator.execute(text)

                # 4. Otherwise ask LLM
                if auto_reply:
                    reply = auto_reply
                else:
                    reply = self.brain.think(text)

                # 5. Speak response
                self.tts.speak(reply)

            except KeyboardInterrupt:
                Log.warn("Interrupted by user.")
                self.running = False
            except Exception as e:
                Log.error(f"Runtime error: {e}")
                try:
                    self.tts.speak("দুঃখিত, একটি সমস্যা হয়েছে।")
                except Exception:
                    pass

    def cleanup(self):
        Log.info("Shutting down...")
        self.wake.cleanup()
        Log.ok("Bye!")


# ================================================================== #
#                             ENTRY POINT                            #
# ================================================================== #

def main():
    # Load .env if available (optional)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        # Refresh Config in case .env defines keys
        Config.PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "").strip()
        Config.DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "").strip()
        Config.OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "").strip()
        Config.GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "").strip()
    except ImportError:
        pass

    # Sanity check: at least one LLM path
    if not (Config.DEEPSEEK_API_KEY or Config.GEMINI_API_KEY):
        Log.warn("No LLM API key found. Assistant will use rule-based replies only.")

    assistant = Assistant()
    try:
        assistant.run()
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()