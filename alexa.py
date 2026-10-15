"""
Alexa main loop — openWakeWord + Moonshine + Qwen 2.5 1.5B.

- Wake detection: openWakeWord ("alexa" model)
- STT: Moonshine Base (batch transcription via transcribe_without_streaming)
- Intent: regex-first via intent.py, LLM fallback (Qwen 1.7B) for fuzzy phrases
- Conversational fallback: Qwen 2.5 1.5B via Ollama
- TTS: Piper

Install:
    pip install openwakeword moonshine-voice sounddevice scipy piper-tts requests

Download Moonshine model (one-time):
    python -m moonshine_voice.download --language en

Pull Qwen models:
    ollama pull qwen3:1.7b
    ollama pull qwen2.5:1.5b
"""

import os
import re
import time
import queue
import json
import tempfile
import subprocess
import wave


import numpy as np
import requests
import sounddevice as sd
from scipy.io.wavfile import write
from openwakeword.model import Model as OWWModel
from piper.voice import PiperVoice

from moonshine_voice import Transcriber, ModelArch, get_model_for_language

from intent import classify
from dispatcher import execute


# =========================
# ALEXA CONFIGURATION
# =========================

_conversation_history = []

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "tomng/lfm2.5-instruct:1.2b-q8_0"   # conversational fallback

PIPER_MODEL = os.path.expanduser(
    "~/alexa/voices/en_GB-jenny_dioco-medium.onnx"
)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # openWakeWord expects 80ms chunks @ 16kHz

WAKEWORD_MODEL = os.path.expanduser("~/alexa/wakeword/alexa_v0.1.onnx")
WAKEWORD_THRESHOLD = 0.4
WAKEWORD_CONFIRM_FRAMES = 3
WAKEWORD_MIN_RMS = 250

# --- Silence-based recording (VAD) settings ---
VAD_SILENCE_RMS = 900
VAD_SILENCE_DURATION = 0.3 #was 3
VAD_MAX_SECONDS = 8
VAD_MIN_SPEECH_SECONDS = 0.6
VAD_NO_SPEECH_TIMEOUT = 5.0

DEBUG_AUDIO = True


# =========================
# MODELS
# =========================

oww = OWWModel(wakeword_model_paths=[WAKEWORD_MODEL])


print("Loading Moonshine model...")
model_path, model_arch = get_model_for_language("en")
moonshine = Transcriber(
    model_path=model_path,
    model_arch=model_arch,
)
print("Moonshine loaded.")

_voice = PiperVoice.load(PIPER_MODEL)


# =========================
# TEXT TO SPEECH
# =========================

def _split_into_chunks(text: str) -> list:
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        chunks.extend(s.strip() for s in sentences if s.strip())
    return chunks or [text.strip()]


def _speak_chunk(text: str):
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

def record_audio() -> np.ndarray:
    """
    Records until end-of-speech is detected via RMS-based VAD.
    Returns a float32 numpy array (mono, 16kHz, -1.0 to 1.0)
    ready for Moonshine.
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
            if DEBUG_AUDIO:
                print("  [rec] ⚠️ queue empty for 1s")
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

        if (speech_started
                and total_frames >= min_speech_frames
                and consecutive_silence >= silence_frames_needed):
            if DEBUG_AUDIO:
                print(f"  [rec] end-of-speech after {total_frames} frames")
            break

        if not speech_started and total_frames >= no_speech_timeout_frames:
            if DEBUG_AUDIO:
                print("  [rec] no speech detected, bailing")
            break

        if time.time() - t_start > VAD_MAX_SECONDS:
            if DEBUG_AUDIO:
                print("  [rec] wall-clock cap, bailing")
            break

    _recording_command = False

    if frames:
        audio_int16 = np.concatenate(frames)
    else:
        audio_int16 = np.zeros(FRAME_SAMPLES, dtype=np.int16)

    audio_float = audio_int16.astype(np.float32) / 32768.0

    dur = len(audio_float) / SAMPLE_RATE
    peak = int(np.max(np.abs(audio_int16))) if audio_int16.size else 0
    avg_rms = _frame_rms(audio_int16) if audio_int16.size else 0.0
    print(f"  [rec] done: {dur:.2f}s peak={peak} avg_rms={avg_rms:.0f}")

    return audio_float


# =========================
# TRANSCRIBE (Moonshine)
# =========================

def transcribe(audio_float: np.ndarray) -> str:
    if audio_float.size == 0:
        return ""
    try:
        transcript = moonshine.transcribe_without_streaming(
            audio_data=audio_float.tolist(),
            sample_rate=SAMPLE_RATE,
        )
        text = " ".join(line.text for line in transcript.lines).strip().lower()
        print(f"  [asr] text='{text}'")
        return text
    except Exception as e:
        print(f"  [asr] Moonshine error: {e}")
        return ""


def remove_wake_word(text):
    text = re.sub(r"\b(hey|hello)?\s*alexa\b", "", text, flags=re.IGNORECASE)
    return text.strip()


# =========================
# ASK QWEN 2.5 1.5B (conversational fallback)
# =========================

def ask_alexa_stream(question: str):
    """
    Sends the question to Ollama, streams the response token by token,
    and speaks each completed sentence immediately. Now with memory!
    """
    global _conversation_history
    
    # Add the new user question to memory
    _conversation_history.append({"role": "user", "content": question})
    
    # Keep only the last 6 messages (3 exchanges) to prevent context overload
    history = _conversation_history[-6:]
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are Alexa, a friendly voice assistant. "
                "Give concise answers suitable for speech. "
                "Do not use markdown. Keep responses under 2 sentences."
            ),
        },
        *history
    ]

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,       # <--- CRITICAL: Enable streaming!
                "think": False,
                "options": {
                    "temperature": 0.5, #was 0.7 #0.3 used
                    "num_predict": 60,  #was 120
                    "top_k": 50,       #new #was 20
                    "repeat_penalty": 1.05, #new
                },
            },
            stream=True,              # <--- CRITICAL: Tell requests to stream
            timeout=30,
        )

        buffer = ""
        full_reply = "" # <-- Tracks the entire response for memory
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                token = chunk.get("message", {}).get("content", "")
                if not token:
                    continue

                buffer += token
                full_reply += token # <-- Add token to full reply

                # Check if we hit a sentence boundary (. ! ?)
                match = re.search(r'([.!?])(\s+|$)', buffer)
                if match:
                    end_idx = match.end(1)  # End of the punctuation
                    to_speak = buffer[:end_idx].strip()
                    buffer = buffer[end_idx:]

                    if to_speak:
                        print(f"Alexa (streaming): {to_speak}")
                        _speak_chunk(to_speak)

            except json.JSONDecodeError:
                continue

        # Speak any remaining text that didn't end with punctuation
        if buffer.strip():
            print(f"Alexa (streaming): {buffer.strip()}")
            _speak_chunk(buffer.strip())
            
        # Add the assistant's full reply to memory
        if full_reply.strip():
            _conversation_history.append({"role": "assistant", "content": full_reply.strip()})

    except Exception as e:
        print("Ollama error:", e)
        speak("Sorry, I couldn't connect to my brain.")

# =========================
# COMMAND HANDLING
# =========================

def handle_command(command: str):
    global _last_response_time
    _last_response_time = time.time()

    # Split the command on "and", "then", or commas
    parts = re.split(r"\band\b|\bthen\b|,", command)

    # Clean up any empty spaces and remove empty strings
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        print(f"  [seq] Detected sequence: {parts}")
        for part in parts:
            _handle_single_command(part)
    else:
        _handle_single_command(command)


def _handle_single_command(command: str):
    intent = classify(command)
    print(f"  [intent] {intent}")

    if intent.get("intent") not in (None, "unknown"):
        response = execute(intent)
        speak(response)
    else:
        ask_alexa_stream(command)


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

            _listening_for_wake = False
            _recording_command = True

            t0 = time.time()
            audio = record_audio()
            print(f"  [main] record_audio returned in {time.time()-t0:.2f}s")

            t0 = time.time()
            command = transcribe(audio)
            print(f"  [main] transcribe took {time.time()-t0:.2f}s")

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
