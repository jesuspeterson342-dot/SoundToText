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

import numpy as np
import sounddevice as sd
from evdev import InputDevice, ecodes, list_devices
from faster_whisper import WhisperModel
from scipy.io import wavfile


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

        whisper_device = os.environ.get("WHISPER_DEVICE", "cpu").strip().lower()
        compute_type = "int8" if whisper_device == "cpu" else "float16"

        logging.info("Loading model...")
        self.model = WhisperModel(
            "large-v3",
            device=whisper_device,
            compute_type=compute_type,
        )
        logging.info("Model loaded (device=%s, compute_type=%s)", whisper_device, compute_type)

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logging.warning("Audio status: %s", status)

        if not self.is_recording:
            return

        with self.recording_lock:
            self.recording_chunks.append(indata.copy())

    def start_recording(self) -> None:
        if self.is_recording:
            return

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

                text = " ".join(parts).strip()
                logging.info("Recognized: %s", text)

                if text:
                    self.insert_text(text)
                else:
                    logging.info("Nothing recognized")

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