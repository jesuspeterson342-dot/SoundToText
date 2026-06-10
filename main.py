import logging
import os
import select
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Optional

import httpx
import numpy as np
import sounddevice as sd
from evdev import InputDevice, ecodes, list_devices
from faster_whisper import WhisperModel
from scipy.io import wavfile
from sympy.codegen.ast import continue_

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

FS = 16000
CHANNELS = 1
LANGUAGE = "ru"
HOTKEY = ecodes.KEY_F8
HOTKEY_NAME = "F8"
HOTKEY_DEBOUNCE_SECONDS = 0.30

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TIMEOUT = 20.0

TECH_TERMS = [
    "OpenClaw",
    "Ollama",
    "Whisper",
    "faster-whisper",
    "Python",
    "Linux",
    "Arch Linux",
    "Hyprland",
    "Wayland",
    "DevAgentXD",
    "accessAI",
    "Kwork",
    "DeepSeek",
    "OpenAI",
    "Claude",
    "Gemini",
    "GitHub",
    "PyCharm",
    "VS Code",
    "CLI",
    "API",
    "JSON",
]


class VoiceToTextApp:
    def __init__(self) -> None:
        self.recording_chunks: list[np.ndarray] = []
        self.recording_lock = threading.Lock()
        self.toggle_lock = threading.Lock()
        self.transcribe_lock = threading.Lock()

        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self.running = True
        self.last_hotkey_ts = 0.0
        self.was_playing = False

        model_name = os.environ.get("WHISPER_MODEL", "large-v3").strip()

        logging.info("Loading model...")

        try:
            logging.info("Trying CUDA GPU...")
            self.model = WhisperModel(
                model_name,
                device="cuda",
                compute_type="float16",
            )
            logging.info("Model loaded on CUDA GPU")

        except Exception:
            logging.exception("CUDA failed, falling back to CPU")

            self.model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
            )
            logging.info("Model loaded on CPU")

    def postprocess_text_with_ollama(self, text: str) -> str:
        """
        Постобработка текста через Ollama LLM.
        Исправляет опечатки, термины, добавляет пунктуацию.
        При ошибке возвращает исходный текст.
        """
        if not text.strip():
            return text

        terms_list = ", ".join(TECH_TERMS)

        prompt = f"""Ты модуль постобработки текста после speech-to-text.

Твоя задача:
исправить ошибки распознавания речи, опечатки, пунктуацию, регистр букв и технические термины.

Жёсткие правила вывода:
1. Верни только исправленный текст.
2. Не пиши вступления.
3. Не пиши "Вот исправленный текст:".
4. Не объясняй изменения.
5. Не отвечай на смысл текста.
6. Не добавляй новые факты.
7. Не удаляй смысловые части.
8. Не меняй стиль автора без причины.
9. Не используй Markdown.
10. Не оборачивай ответ в кавычки.
11. Если текст уже нормальный, верни его почти без изменений.

Словарь терминов, которые нужно сохранять:
{terms_list}

Исправляемый текст:
<<<
{text}
>>>

Верни только исправленный текст."""

        try:
            logging.info("Starting Ollama postprocessing with model: %s", OLLAMA_MODEL)

            with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
                response = client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                    },
                )

                if response.status_code != 200:
                    logging.error("Ollama returned status %d: %s", response.status_code, response.text)
                    return text

                data = response.json()
                processed = data.get("response", "").strip()

                if not processed:
                    logging.warning("Ollama returned empty response, using original text")
                    return text

                logging.info("Ollama postprocessing completed")
                return processed

        except httpx.TimeoutException:
            logging.error("Ollama request timeout after %.1f seconds, using original text", OLLAMA_TIMEOUT)
            return text
        except httpx.ConnectError:
            logging.error("Cannot connect to Ollama at %s, using original text", OLLAMA_URL)
            return text
        except Exception:
            logging.exception("Ollama postprocessing failed, using original text")
            return text


    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logging.warning("Audio status: %s", status)

        if not self.is_recording:
            return

        with self.recording_lock:
            self.recording_chunks.append(indata.copy())

    def pause_audio (self) -> None: # To pause the player, during recording
        self.was_playing = False

        try:
            result = subprocess.run(
                ["playerctl", "status"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                status = result.stdout.strip()

                if status == "Playing":
                    self.was_playing = True
                    logging.warning("Player paused, starting recording...")

                    subprocess.run(["playerctl", "pause"], check=True)

                else:
                    logging.warning("Player paused, starting recording...")

        except Exception:
            logging.exception("Failed to pause audio")

    def restore_audio(self) -> None: # To restore the player, after recording
        if not self.was_playing:
            return

        try:
            logging.info("Resuming audio playback...")
            subprocess.run(["playerctl", "play"], check=True)
            self.was_playing = False

        except Exception:
            logging.exception("Failed to resume audio")

    def start_recording(self) -> None:
        if self.is_recording:
            return

        subprocess.Popen(["notify-send", "Recording started", "Press F8 to stop"])
        self.pause_audio()

        try:
            with self.recording_lock:
                self.recording_chunks = []

            self.stream = sd.InputStream(
                samplerate=FS,
                channels=CHANNELS,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
            self.is_recording = True
            logging.info("Recording started")

        except Exception:
            self.is_recording = False
            self.stream = None
            logging.exception("Failed to start recording")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        subprocess.Popen(["notify-send", "Recording stopped", "Press F8 to start"])
        self.restore_audio()
        logging.info("Stopping recording...")
        self.is_recording = False

        local_stream = self.stream
        self.stream = None

        try:
            if local_stream is not None:
                local_stream.stop()
                local_stream.close()
        except Exception:
            logging.exception("Failed to stop audio stream")

        with self.recording_lock:
            chunks = self.recording_chunks
            self.recording_chunks = []

        if not chunks:
            logging.warning("No audio captured")
            return

        try:
            audio = np.concatenate(chunks, axis=0).squeeze().astype(np.float32)
        except Exception:
            logging.exception("Failed to combine audio chunks")
            return

        threading.Thread(
            target=self.transcribe_and_insert,
            args=(audio,),
            daemon=True,
        ).start()

    def transcribe_and_insert(self, audio: np.ndarray) -> None:
        with self.transcribe_lock:
            try:
                duration = len(audio) / FS
                logging.info("Transcribing...")
                logging.info("Processing audio with duration %.3f sec", duration)

                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    wavfile.write(tmp.name, FS, audio)

                    segments, info = self.model.transcribe(
                        tmp.name,
                        language=LANGUAGE,
                        task="transcribe",
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        initial_prompt="Это русская диктовка. Распознавай только русский текст, без перевода на английский."
                    )

                    parts = []

                    for segment in segments:
                        piece = segment.text.strip()
                        if piece:
                            parts.append(piece)

                raw_text = " ".join(parts).strip()
                logging.info("Raw recognized text: %s", raw_text)

                if not raw_text:
                    logging.info("Nothing recognized")
                    return

                final_text = self.postprocess_text_with_ollama(raw_text)
                logging.info("Final text after postprocessing: %s", final_text)

                if final_text:
                    self.insert_text(final_text)
                else:
                    logging.warning("Postprocessing returned empty text")

            except Exception:
                logging.exception("Transcription failed")

    def insert_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if not shutil.which("wtype"):
            logging.error("wtype is not installed")
            return

        try:
            result = subprocess.run(
                ["wtype", text],
                text=True,
                capture_output=True,
                check=False,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip() or "unknown error"
                logging.error("wtype failed: %s", stderr)
                return

            logging.info("Text inserted")

        except Exception:
            logging.exception("Insert error")

    def toggle_recording(self) -> None:
        with self.toggle_lock:
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()

    def should_accept_hotkey(self) -> bool:
        now = time.monotonic()
        if now - self.last_hotkey_ts < HOTKEY_DEBOUNCE_SECONDS:
            return False
        self.last_hotkey_ts = now
        return True

    def find_keyboard_devices(self) -> list[InputDevice]:
        devices: list[InputDevice] = []

        for path in list_devices():
            try:
                dev = InputDevice(path)
                caps = dev.capabilities().get(ecodes.EV_KEY, [])
                if HOTKEY in caps:
                    devices.append(dev)
            except Exception:
                logging.exception("Failed to inspect device %s", path)

        return devices

    def keyboard_listener(self) -> None:
        devices = self.find_keyboard_devices()

        if not devices:
            logging.error("No input devices with %s found", HOTKEY_NAME)
            return

        logging.info("Listening on devices:")
        for dev in devices:
            logging.info("  %s  %s", dev.path, dev.name)

        while self.running:
            try:
                ready, _, _ = select.select(devices, [], [], 1.0)

                for dev in ready:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue

                        if event.code != HOTKEY:
                            continue

                        if event.value != 1:
                            continue

                        if not self.should_accept_hotkey():
                            continue

                        logging.info("%s pressed on %s (%s)", HOTKEY_NAME, dev.path, dev.name)
                        self.toggle_recording()

            except Exception:
                logging.exception("Keyboard listener error")
                time.sleep(1)

    def shutdown(self) -> None:
        self.running = False
        if self.is_recording:
            self.stop_recording()

    def run(self) -> None:
        listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        listener_thread.start()

        logging.info("Ready. Press %s to start/stop recording.", HOTKEY_NAME)

        while self.running:
            time.sleep(1)


def main() -> None:
    app = VoiceToTextApp()

    def handle_signal(signum, frame) -> None:
        logging.info("Exiting...")
        app.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    app.run()


if __name__ == "__main__":
    main()