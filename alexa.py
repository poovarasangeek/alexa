"""
Alexa main loop — openWakeWord edition, no LLM.

Wake detection runs a lightweight always-on keyword spotter
(openWakeWord, "alexa" model). Whisper only runs AFTER the wake word fires.
Commands are routed through intent classification + dispatcher.
Unknown commands get a fixed spoken fallback (no LLM / Ollama).

Install:
    pip install openwakeword faster-whisper sounddevice scipy piper-tts

Wake word model file needed:
    ~/alexa/wakeword/alexa_v0.1.onnx
"""

import os
import re
import time
import queue
import tempfile
import subprocess
import wave
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from openwakeword.model import Model as OWWModel
from piper.voice import PiperVoice

from intent import classify
from dispatcher import execute


# =========================
# ALEXA CONFIGURATION
# =========================

PIPER_MODEL = os.path.expanduser(
    "~/alexa/voices/en_US-hfc_female-medium.onnx"
)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # openWakeWord expects 80ms chunks @ 16kHz

WAKEWORD_MODEL = os.path.expanduser("~/alexa/wakeword/alexa_v0.1.onnx")
WAKEWORD_THRESHOLD = 0.4
WAKEWORD_CONFIRM_FRAMES = 3
WAKEWORD_MIN_RMS = 250

# --- Silence-based recording (VAD) settings ---
VAD_SILENCE_RMS = 300          # tuned for normal-gain mics; see debug prints
VAD_SILENCE_DURATION = 0.3     # wait this long after you stop speaking
VAD_MAX_SECONDS = 8
VAD_MIN_SPEECH_SECONDS = 0.6
VAD_NO_SPEECH_TIMEOUT = 5.0

# --- Debug flags ---
DEBUG_AUDIO = True             # set False once everything works


# =========================
# MODELS
# =========================

whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
oww = OWWModel(wakeword_model_paths=[WAKEWORD_MODEL])
_voice=PiperVoice.load(PIPER_MODEL)        # <-- ADD THIS LINE

# =========================
# TEXT TO SPEECH
# =========================

def _split_into_chunks(text: str) -> list:
    """
    Break a response into speakable chunks: split on line breaks and
    sentence boundaries so Piper can start speaking immediately.
    """
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        chunks.extend(s.strip() for s in sentences if s.strip())
    return chunks or [text.strip()]


def _speak_chunk(text: str):
    """Synthesize and play a single chunk using the in-memory Piper model."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_file = f.name
    try:
        with wave.open(wav_file, "wb") as wav:
            _voice.synthesize_wav(text, wav)
        subprocess.run(
            ["aplay", wav_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)


def speak(text):
    print(f"Alexa: {text}")
    for chunk in _split_into_chunks(text):
        _speak_chunk(chunk)


# =========================
# AUDIO STATE + QUEUES
# =========================

command_q: "queue.Queue[np.ndarray]" = queue.Queue()
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
_recording_command = False
_listening_for_wake = True


def _frame_rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def _audio_callback(indata, frames, time_info, status):
    if status and DEBUG_AUDIO:
        print("⚠️ Audio status:", status)
    if _recording_command:
        command_q.put(indata[:, 0].copy())
    elif _listening_for_wake:
        audio_q.put(indata[:, 0].copy())


# =========================
# RECORD MICROPHONE
# =========================

def record_audio() -> str:
    """
    Records until end-of-speech is detected via RMS-based VAD.
    Assumes _recording_command was set True by main() BEFORE calling.
    """
    global _recording_command
    print("🎤 Listening...")

    silence_frames_needed = int(VAD_SILENCE_DURATION * SAMPLE_RATE / FRAME_SAMPLES)
    min_speech_frames = int(VAD_MIN_SPEECH_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
    max_frames = int(VAD_MAX_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
    no_speech_timeout_frames = int(VAD_NO_SPEECH_TIMEOUT * SAMPLE_RATE / FRAME_SAMPLES)

    frames = []
    consecutive_silence = 0
    total_frames = 0
    speech_started = False

    t_start = time.time()

    while total_frames < max_frames:
        try:
            frame = command_q.get(timeout=1.0)
        except queue.Empty:
            print("  [rec] ⚠️ queue empty for 1s — callback may be dead")
            continue

        frames.append(frame)
        total_frames += 1

        rms = _frame_rms(frame)

        if DEBUG_AUDIO and (total_frames % 10 == 0):
            print(f"  [rec] f={total_frames} rms={rms:.0f} "
                  f"sp={speech_started} sil={consecutive_silence}/{silence_frames_needed}")

        if rms >= VAD_SILENCE_RMS:
            speech_started = True
            consecutive_silence = 0
        elif speech_started:
            consecutive_silence += 1

        # End-of-speech
        if (speech_started
                and total_frames >= min_speech_frames
                and consecutive_silence >= silence_frames_needed):
            print(f"  [rec] end-of-speech after {total_frames} frames")
            break

        # No speech at all
        if not speech_started and total_frames >= no_speech_timeout_frames:
            print("  [rec] no speech detected, bailing")
            break

        # Safety cap on total duration
        if time.time() - t_start > VAD_MAX_SECONDS:
            print("  [rec] hit wall-clock cap, bailing")
            break

    _recording_command = False

    audio = (np.concatenate(frames)
             if frames else np.zeros(FRAME_SAMPLES, dtype=np.int16))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
    write(filename, SAMPLE_RATE, audio)

    dur = len(audio) / SAMPLE_RATE
    peak = int(np.max(np.abs(audio))) if audio.size else 0
    avg_rms = _frame_rms(audio.astype(np.int16)) if audio.size else 0.0
    print(f"  [rec] done: {dur:.2f}s peak={peak} avg_rms={avg_rms:.0f}")

    return filename


# =========================
# TRANSCRIBE
# =========================

def transcribe(filename):
    segments, info = whisper.transcribe(
        filename,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
        best_of=1,
        temperature=0.0,
        without_timestamps=True,
    )
    text = " ".join(seg.text for seg in segments).strip().lower()
    print(f"  [asr] duration={info.duration:.2f}s text='{text}'")
    return text


def remove_wake_word(text):
    text = re.sub(r"\b(hey|hello)?\s*alexa\b", "", text, flags=re.IGNORECASE)
    return text.strip()


# =========================
# COMMAND HANDLING (no LLM)
# =========================

def handle_command(command: str):
    intent = classify(command)
    print(f"  [intent] {intent}")
    if intent.get("intent") not in (None, "unknown"):
        response = execute(intent)
    else:
        response = "Sorry, I don't know that one yet."
    speak(response)


# =========================
# WAKE WORD
# =========================

def flush_wakeword_buffer():
    silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    for _ in range(25):
        oww.predict(silent_frame)
    for q in (audio_q, command_q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


def wait_for_wake_word():
    print("👂 Waiting for wake word (say 'Alexa')...")
    consecutive_hits = 0
    while True:
        try:
            frame = audio_q.get(timeout=1.0)
        except queue.Empty:
            continue

        predictions = oww.predict(frame)
        rms = _frame_rms(frame)

        for model_name, score in predictions.items():
            if score > WAKEWORD_THRESHOLD and rms > WAKEWORD_MIN_RMS:
                consecutive_hits += 1
                if consecutive_hits >= WAKEWORD_CONFIRM_FRAMES:
                    print(f"✨ Wake word detected "
                          f"({model_name}, score={score:.2f}, rms={rms:.0f})")
                    return
            else:
                consecutive_hits = 0


# =========================
# MAIN LOOP
# =========================

def main():
    global _listening_for_wake, _recording_command

    print()
    print("================================")
    print("        ALEXA ASSISTANT")
    print("================================")
    print()

    with sd.InputStream(
        device=15,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=_audio_callback,
    ):
        while True:
            _listening_for_wake = True
            time.sleep(0.4)
            _recording_command = False
            wait_for_wake_word()

            # Begin capturing command audio immediately.
            _listening_for_wake = False
            _recording_command = True

            t0 = time.time()
            filename = record_audio()
            print(f"  [main] record_audio returned in {time.time()-t0:.2f}s")

            t0 = time.time()
            command = transcribe(filename)
            print(f"  [main] transcribe took {time.time()-t0:.2f}s")

            try:
                os.remove(filename)
            except OSError:
                pass

            if not command:
                speak("I didn't hear anything.")
                flush_wakeword_buffer()
                continue

            print(f"You: {command}")
            command = remove_wake_word(command)

            if not command:
                speak("Yes?")
                flush_wakeword_buffer()
                continue

            if any(x in command
                   for x in ["goodbye alexa", "bye alexa",
                             "exit alexa", "quit alexa"]):
                speak("Goodbye.")
                break

            handle_command(command)
            flush_wakeword_buffer()


if __name__ == "__main__":
    main()
