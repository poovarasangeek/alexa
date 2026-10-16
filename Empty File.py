#!/usr/bin/env python3
"""
alexa.py
Efficient local voice assistant.

Features:
- openWakeWord wake word + voice/beep acknowledgment
- Follow-up window: chained commands without repeating the wake word
- Silero VAD with RMS fallback
- Faster-Whisper STT with Moonshine fallback
- Kokoro TTS with Piper fallback
- Ollama LLM fallback + tool calling
- Regex intent fast path
- Long-term memory (SQLite FTS5) + persistent chat history
- RAG knowledge base over your personal text notes
- Optional terminal visual UI
- NO barge-in (by design)

Usage:
    python alexa.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import queue
import wave
import glob
import random
import shutil
import sqlite3
import threading
import tempfile
import subprocess
import datetime
import urllib.parse
import urllib.request

import numpy as np
import requests
import sounddevice as sd


# ============================================================
# CONFIG
# ============================================================

ASSISTANT_NAME = "alexa"

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms chunks @ 16kHz

BASE_DIR = os.path.expanduser("~/alexa")
MEMORY_DB = os.path.expanduser("~/.alexa_memory.db")

# --- VAD ---
ENABLE_SILERO_VAD = True
SILERO_THRESHOLD = 0.55
SILERO_MIN_RMS = 120
RMS_SPEECH_THRESHOLD = 900

VAD_MAX_SECONDS = 8
VAD_MIN_SPEECH_SECONDS = 0.5
VAD_SILENCE_DURATION = 0.35
VAD_NO_SPEECH_TIMEOUT = 5.0

# --- Debug / UI ---
DEBUG_AUDIO = False
VISUAL_UI = False

# --- STT ---
# Options: auto, faster-whisper, moonshine
STT_BACKEND = "auto"
FAST_WHISPER_MODEL = "base.en"
FAST_WHISPER_COMPUTE = "int8"

# --- TTS ---
# Options: auto, kokoro, piper
TTS_BACKEND = "auto"

PIPER_MODEL = os.path.join(
    BASE_DIR,
    "voices",
    "en_GB-jenny_dioco-medium.onnx"
)

KOKORO_MODEL = os.path.join(
    BASE_DIR,
    "kokoro",
    "kokoro-v1.0.onnx"
)

KOKORO_VOICES = os.path.join(
    BASE_DIR,
    "kokoro",
    "voices-v1.0.bin"
)

KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0

# --- Wake word ---
WAKEWORD_MODEL = os.path.join(
    BASE_DIR,
    "wakeword",
    "alexa_v0.1.onnx"
)

WAKEWORD_THRESHOLD = 0.20  # was 0.40 (lower = easier wake, more false wakes)
WAKEWORD_CONFIRM_FRAMES = 3
WAKEWORD_MIN_RMS = 250

WAKE_WORDS = [
    "alexa",
]

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434/api/chat"

# Fast:     qwen2.5:1.5b
# Balanced: qwen2.5:3b
# Env override:
#   export ALEXA_LLM_FAST="qwen2.5:1.5b"
#   export ALEXA_LLM_SMART="qwen2.5:3b"
LLM_FAST_MODEL = os.getenv("ALEXA_LLM_FAST", "qwen2.5:3b")
LLM_SMART_MODEL = os.getenv("ALEXA_LLM_SMART", LLM_FAST_MODEL)

USE_LLM_TOOLS = True
MEMORY_ENABLED = True
MEMORY_LIMIT = 3

# --- RAG knowledge base (your personal text notes) ---
RAG_ENABLED = True
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
RAG_TOP_K = 3

# --- Wake acknowledgment + follow-up session ---
WAKE_ACK_MODE = "voice"            # "voice" | "beep" | "none"
WAKE_ACK_PHRASES = ["Yes?", "Yes, how can I help?", "I'm listening."]
FOLLOWUP_WINDOW_SECONDS = 6.0      # how long to wait for a follow-up command
FOLLOWUP_GUARD_SECONDS = 0.8       # ignore mic right after TTS (echo guard)

# --- Memory ---
CHAT_HISTORY_SESSION = 10          # messages kept in RAM
CHAT_HISTORY_LOAD = 6              # messages reloaded after restart


# Heuristic used before trying LLM tool calling.
COMMAND_HINTS = {
    "open", "launch", "start", "run", "close",
    "turn", "set", "change", "increase", "decrease", "mute",
    "volume", "brightness",
    "wifi", "wireless", "bluetooth",
    "battery", "weather", "time", "date", "ip",
    "screenshot", "lock", "shutdown", "reboot", "restart",
    "suspend", "sleep",
    "search", "find", "play", "song", "music",
    "keyboard", "light", "backlight",
    "power", "profile", "performance", "balanced", "saver",
    "terminal", "browser", "brave", "firefox", "chrome",
    "calculator", "files", "settings", "calendar",
    "vlc", "spotify", "localsend",
}


# ============================================================
# GLOBAL STATE
# ============================================================

_conversation_history = []

command_q: "queue.Queue[np.ndarray]" = queue.Queue()
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

state = {
    "listening_wake": True,
    "recording": False,
}


# ============================================================
# OPTIONAL IMPORT FLAGS
# ============================================================

HAVE_TORCH = False
if ENABLE_SILERO_VAD:
    try:
        import torch
        HAVE_TORCH = True
    except Exception:
        HAVE_TORCH = False

HAVE_FASTER_WHISPER = False
if STT_BACKEND in ("auto", "faster-whisper"):
    try:
        from faster_whisper import WhisperModel
        HAVE_FASTER_WHISPER = True
    except Exception:
        HAVE_FASTER_WHISPER = False

HAVE_MOONSHINE = False
if STT_BACKEND in ("auto", "moonshine"):
    try:
        from moonshine_voice import Transcriber, get_model_for_language
        HAVE_MOONSHINE = True
    except Exception:
        HAVE_MOONSHINE = False

HAVE_PIPER = False
if TTS_BACKEND in ("auto", "piper"):
    try:
        import piper.voice  # noqa: F401
        HAVE_PIPER = True
    except Exception:
        HAVE_PIPER = False

HAVE_KOKORO = False
if TTS_BACKEND in ("auto", "kokoro"):
    try:
        import kokoro_onnx  # noqa: F401
        HAVE_KOKORO = True
    except Exception:
        HAVE_KOKORO = False

HAVE_OPENWAKEWORD = False
if os.path.exists(WAKEWORD_MODEL):
    try:
        from openwakeword.model import Model as OWWModel
        HAVE_OPENWAKEWORD = True
    except Exception:
        HAVE_OPENWAKEWORD = False


# ============================================================
# VISUAL UI
# ============================================================

class VisualUI:
    """
    Very lightweight terminal UI.
    Set VISUAL_UI = True for a simple holographic-style status display.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.state = "idle"
        self.text = ""

    def update(self, new_state: str, text: str = ""):
        self.state = new_state
        self.text = text

        if not self.enabled:
            return

        self.render()

    def render(self):
        orbs = {
            "idle": "◇",
            "waiting": "◌",
            "listening": "●",
            "thinking": "◐",
            "speaking": "◉",
        }

        orb = orbs.get(self.state, "◇")

        try:
            sys.stdout.write("\033[2J\033[H")
            print()
            print(f"        {orb}  {ASSISTANT_NAME.upper()}")
            print(f"        state: {self.state.upper()}")
            print()

            if self.text:
                print(f"        > {self.text}")
                print()

            sys.stdout.flush()
        except Exception:
            pass


ui = VisualUI(VISUAL_UI)


# ============================================================
# LONG TERM MEMORY
# ============================================================

class Memory:
    """
    Local long-term memory + persistent chat log.
    Uses SQLite FTS5 when available.
    """

    def __init__(self, db_path: str):
        self.enabled = MEMORY_ENABLED
        self.fts = False
        self.conn = None
        self.lock = threading.Lock()

        if not self.enabled:
            return

        try:
            self.conn = sqlite3.connect(
                db_path,
                check_same_thread=False,
            )

            with self.lock:
                try:
                    self.conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS memory USING fts5(text)"
                    )
                    self.fts = True
                except Exception:
                    self.conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS memory(
                            id INTEGER PRIMARY KEY,
                            text TEXT,
                            created TEXT
                        )
                        """
                    )

                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat(
                        id INTEGER PRIMARY KEY,
                        role TEXT,
                        content TEXT,
                        created TEXT
                    )
                    """
                )

                self.conn.commit()

        except Exception as e:
            print("Memory initialization error:", e)
            self.enabled = False

    def log_chat(self, role: str, content: str):
        if not self.enabled or not self.conn:
            return

        content = str(content).strip()
        if not content:
            return

        try:
            with self.lock:
                self.conn.execute(
                    "INSERT INTO chat(role, content, created) VALUES (?, ?, ?)",
                    (role, content, datetime.datetime.now().isoformat()),
                )
                # keep only the last 100 messages on disk
                self.conn.execute(
                    "DELETE FROM chat WHERE id NOT IN "
                    "(SELECT id FROM chat ORDER BY id DESC LIMIT 100)"
                )
                self.conn.commit()
        except Exception as e:
            print("Chat log error:", e)

    def load_chat(self, limit: int = 6) -> list:
        if not self.enabled or not self.conn:
            return []

        try:
            with self.lock:
                rows = self.conn.execute(
                    "SELECT role, content FROM chat ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [{"role": r, "content": c} for r, c in reversed(rows)]
        except Exception:
            return []

    def add(self, text: str):
        if not self.enabled or not self.conn:
            return

        text = text.strip()
        if not text:
            return

        try:
            with self.lock:
                if self.fts:
                    self.conn.execute(
                        "INSERT INTO memory(text) VALUES (?)",
                        (text,),
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO memory(text, created) VALUES (?, ?)",
                        (text, datetime.datetime.now().isoformat()),
                    )
                self.conn.commit()
        except Exception as e:
            print("Memory add error:", e)

    def search(self, query: str, limit: int = 3) -> list:
        if not self.enabled or not self.conn:
            return []

        terms = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())[:6]
        if not terms:
            return []

        try:
            with self.lock:
                if self.fts:
                    match = " OR ".join(f'"{term}"' for term in terms)
                    try:
                        rows = self.conn.execute(
                            """
                            SELECT text
                            FROM memory
                            WHERE memory MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (match, limit),
                        ).fetchall()
                        return [row[0] for row in rows]
                    except Exception:
                        pass

                term = terms[0]
                rows = self.conn.execute(
                    """
                    SELECT text
                    FROM memory
                    WHERE text LIKE ?
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    (f"%{term}%", limit),
                ).fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            print("Memory search error:", e)
            return []

    def recent(self, limit: int = 3) -> list:
        if not self.enabled or not self.conn:
            return []

        try:
            with self.lock:
                if self.fts:
                    rows = self.conn.execute(
                        """
                        SELECT text
                        FROM memory
                        ORDER BY rowid DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        """
                        SELECT text
                        FROM memory
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            print("Memory recent error:", e)
            return []

    def clear(self):
        if not self.enabled or not self.conn:
            return

        try:
            with self.lock:
                self.conn.execute("DELETE FROM memory")
                self.conn.commit()
        except Exception as e:
            print("Memory clear error:", e)


memory = Memory(MEMORY_DB)


# ============================================================
# RAG KNOWLEDGE BASE (personal text notes)
# ============================================================

class KnowledgeBase:
    """
    Simple RAG over your personal notes.
    Put .txt / .md files in KNOWLEDGE_DIR (one fact per line or paragraph).
    Re-indexed automatically at every startup.
    """

    def __init__(self, directory: str):
        self.enabled = RAG_ENABLED and MEMORY_ENABLED
        self.fts = False
        self.conn = None
        self.lock = threading.Lock()

        if not self.enabled:
            return

        try:
            os.makedirs(directory, exist_ok=True)
            self.conn = sqlite3.connect(MEMORY_DB, check_same_thread=False)

            with self.lock:
                self.conn.execute("DROP TABLE IF EXISTS kb")

                try:
                    self.conn.execute(
                        "CREATE VIRTUAL TABLE kb USING fts5(chunk)"
                    )
                    self.fts = True
                except Exception:
                    self.conn.execute(
                        "CREATE TABLE IF NOT EXISTS kb(chunk TEXT)"
                    )

                count = 0
                for fname in sorted(os.listdir(directory)):
                    if not fname.lower().endswith((".txt", ".md")):
                        continue

                    path = os.path.join(directory, fname)
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    for chunk in self._chunk(content):
                        self.conn.execute(
                            "INSERT INTO kb(chunk) VALUES (?)", (chunk,)
                        )
                        count += 1

                self.conn.commit()

            print(f"Knowledge base: {count} chunks indexed from {directory}")

        except Exception as e:
            print("Knowledge base error:", e)
            self.enabled = False

    @staticmethod
    def _chunk(text: str):
        chunks = []
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if not para:
                continue
            for i in range(0, len(para), 400):
                chunks.append(para[i:i + 400])
        return chunks

    def search(self, query: str, limit: int = 3):
        if not self.enabled or not self.conn:
            return []

        terms = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())[:6]
        if not terms:
            return []

        try:
            with self.lock:
                if self.fts:
                    match = " OR ".join(f'"{t}"' for t in terms)
                    try:
                        rows = self.conn.execute(
                            "SELECT chunk FROM kb WHERE kb MATCH ? "
                            "ORDER BY rank LIMIT ?",
                            (match, limit),
                        ).fetchall()
                        if rows:
                            return [r[0] for r in rows]
                    except Exception:
                        pass

                rows = self.conn.execute(
                    "SELECT chunk FROM kb WHERE chunk LIKE ? LIMIT ?",
                    (f"%{terms[0]}%", limit),
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []


kb = KnowledgeBase(KNOWLEDGE_DIR)


def rag_context(question: str) -> str:
    hits = kb.search(question, RAG_TOP_K)
    if not hits:
        return ""
    return " Facts about the user (use when relevant): " + " | ".join(hits)


def remember_exchange(question: str, reply: str):
    """
    Store one user/assistant exchange in RAM history + on-disk chat log.
    """
    global _conversation_history

    question = str(question or "").strip()
    reply = str(reply or "").strip()

    if not question or not reply:
        return

    _conversation_history.append({"role": "user", "content": question})
    _conversation_history.append({"role": "assistant", "content": reply})

    max_msgs = CHAT_HISTORY_SESSION * 2
    if len(_conversation_history) > max_msgs:
        _conversation_history = _conversation_history[-max_msgs:]

    memory.log_chat("user", question)
    memory.log_chat("assistant", reply)


# ============================================================
# BUILT-IN FALLBACK INTENT + DISPATCHER
#
# If your original intent.py and dispatcher.py exist, this file
# will use them automatically. Otherwise it uses these built-ins.
# ============================================================

def _run(cmd, timeout=5):
    """Safely run a shell command. Returns (success, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return result.returncode == 0, result.stderr.decode().strip()
    except Exception as e:
        return False, str(e)


def built_in_classify(text: str) -> dict:
    text = text.lower().strip()

    # Power
    if re.search(
        r"\b(shut ?down|power ?off|turn off (the )?(system|computer|laptop|pc)|power down)\b",
        text,
    ):
        return {"intent": "shutdown"}

    if re.search(
        r"\b(reboot|restart (the )?(system|computer|laptop|pc)|restart now)\b",
        text,
    ):
        return {"intent": "reboot"}

    if re.search(
        r"\b(lock (the )?(system|screen|computer|laptop|pc)|lock now)\b",
        text,
    ):
        return {"intent": "lock"}

    if re.search(r"\b(suspend|sleep (the )?(system|computer))\b", text):
        return {"intent": "suspend"}

    # Power profiles
    if re.search(
        r"\b(power\s*save|powersave|power-saver|battery\s*save|battery\s*saver)\b",
        text,
    ):
        return {"intent": "set_power_profile", "arg": "power-saver"}

    if re.search(r"\b(performance|high\s*performance|max\s*performance)\b", text):
        return {"intent": "set_power_profile", "arg": "performance"}

    if re.search(r"\b(balanced|normal|default)\b", text):
        return {"intent": "set_power_profile", "arg": "balanced"}

    # Wi-Fi
    if re.search(
        r"\b(turn on|enable|start|switch on)\s+(the\s+)?(wifi|wi-?fi|wireless)\b",
        text,
    ):
        return {"intent": "wifi_on"}

    if re.search(
        r"\b(turn off|disable|stop|switch off)\s+(the\s+)?(wifi|wi-?fi|wireless)\b",
        text,
    ):
        return {"intent": "wifi_off"}

    # Bluetooth
    if re.search(
        r"\b(turn on|enable|start|switch on)\s+(the\s+)?(bluetooth|blue tooth)\b",
        text,
    ):
        return {"intent": "bluetooth_on"}

    if re.search(
        r"\b(turn off|disable|stop|switch off)\s+(the\s+)?(bluetooth|blue tooth)\b",
        text,
    ):
        return {"intent": "bluetooth_off"}

    # Info
    if re.search(r"\b(my ip|ip address|what'?s my ip)\b", text):
        return {"intent": "get_ip"}

    if re.search(r"\b(time|what time|current time)\b", text):
        return {"intent": "get_time"}

    if re.search(r"\b(date|what date|what day|today'?s date)\b", text):
        return {"intent": "get_date"}

    if re.search(r"\b(weather|forecast|temperature)\b", text):
        return {"intent": "get_weather"}

    if re.search(r"\b(battery|power level|charge)\b", text):
        return {"intent": "get_battery"}

    # Volume
    m = re.search(r"\b(set|change|turn|make)\b.*\bvolume\b.*?(\d+)", text)
    if m:
        return {"intent": "set_volume", "arg": m.group(2)}

    m = re.search(r"\bvolume\b.*?(\d+)", text)
    if m:
        return {"intent": "set_volume", "arg": m.group(1)}

    if re.search(r"\b(volume up|increase volume|turn it up|louder)\b", text):
        return {"intent": "volume_up"}

    if re.search(r"\b(volume down|decrease volume|turn it down|quieter)\b", text):
        return {"intent": "volume_down"}

    if re.search(r"\b(mute|silence|stop sound)\b", text):
        return {"intent": "mute"}

    # Media / search
    if re.search(r"\b(play|start|put on)\b.*\b(song|music)\b", text):
        return {"intent": "play_song"}

    search_match = re.search(
        r"\b(search|look up|google|find)\b\s*(for|about)?\s*(.+)",
        text,
    )
    if search_match:
        return {
            "intent": "brave_search",
            "arg": search_match.group(3).strip(),
        }

    # Screenshot
    if re.search(
        r"\b(screenshot|screen shot|take a screenshot|capture screen)\b",
        text,
    ):
        return {"intent": "screenshot"}

    # Open apps
    apps = [
        "terminal", "vlc", "firefox", "chrome", "brave", "browser",
        "code", "vscode", "calculator", "files", "file manager",
        "nautilus", "spotify", "localsend", "local send", "text editor",
        "settings", "software center", "calendar",
    ]

    for app in apps:
        if re.search(
            rf"\b(open|launch|start|run)\s+(the\s+)?{re.escape(app)}\b",
            text,
        ):
            normalized = {
                "browser": "brave",
                "file manager": "files",
                "local send": "localsend",
                "vscode": "code",
                "text editor": "gedit",
                "software center": "gnome-software",
            }.get(app, app)

            return {"intent": "open_app", "arg": normalized}

    # Small talk
    if re.search(r"\b(hello|hi|hey|good morning|good evening)\b", text):
        return {"intent": "greeting"}

    if re.search(r"\b(how are you|how'?s it going)\b", text):
        return {"intent": "status"}

    if re.search(r"\b(who are you|what is your name)\b", text):
        return {"intent": "identity"}

    if re.search(r"\b(thank you|thanks)\b", text):
        return {"intent": "thanks"}

    # Fun
    if re.search(r"\b(joke|tell me a joke|make me laugh)\b", text):
        return {"intent": "joke"}

    if re.search(r"\b(flip a coin|coin flip|heads or tails)\b", text):
        return {"intent": "coin_flip"}

    if re.search(r"\b(roll a die|roll a dice|random number)\b", text):
        return {"intent": "roll_dice"}

    # Keyboard backlight
    if re.search(
        r"\b(turn on|enable)\b.*\b(keyboard|kbd)\b.*\b(light|backlight|backlit)\b",
        text,
    ):
        return {"intent": "keyboard_light_on"}

    if re.search(
        r"\b(turn off|disable)\b.*\b(keyboard|kbd)\b.*\b(light|backlight|backlit)\b",
        text,
    ):
        return {"intent": "keyboard_light_off"}

    return {"intent": "unknown"}


def built_in_execute(intent: dict) -> str:
    name = intent.get("intent")
    arg = intent.get("arg")

    # Time / date
    if name == "get_time":
        now = datetime.datetime.now()
        return f"It's {now.strftime('%I:%M %p').lstrip('0')}."

    if name == "get_date":
        now = datetime.datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    # Weather
    if name == "get_weather":
        try:
            url = "https://wttr.in/?format=%C+%t"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "curl/7.68.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                weather = response.read().decode("utf-8").strip()
            return f"The weather is currently {weather}."
        except Exception:
            return "Sorry, I couldn't fetch the weather right now."

    # Battery
    if name == "get_battery":
        try:
            for bat in ["BAT0", "BAT1"]:
                cap_path = f"/sys/class/power_supply/{bat}/capacity"
                status_path = f"/sys/class/power_supply/{bat}/status"

                if os.path.exists(cap_path):
                    with open(cap_path) as f:
                        percent = f.read().strip()

                    status = "unknown"
                    if os.path.exists(status_path):
                        with open(status_path) as f:
                            status = f.read().strip().lower()

                    return f"The battery is at {percent}% and is {status}."

            return "I couldn't find a battery on this system."
        except Exception:
            return "I couldn't read the battery status."

    # IP
    if name == "get_ip":
        try:
            url = "https://ifconfig.me/ip"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "curl/7.68.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                ip = response.read().decode("utf-8").strip()
            return f"Your public IP address is {ip}."
        except Exception:
            return "Sorry, I couldn't fetch your IP address."

    # Volume
    if name == "set_volume":
        try:
            vol = int(arg)
            if 0 <= vol <= 100:
                _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"])
                return f"Volume set to {vol} percent."
            return "Volume must be between 0 and 100."
        except Exception:
            return "I didn't catch the volume level."

    if name == "volume_up":
        _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"])
        return "Volume increased."

    if name == "volume_down":
        _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"])
        return "Volume decreased."

    if name == "mute":
        _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"])
        return "Audio muted or unmuted."

    # Power profile
    if name == "set_power_profile":
        profile = str(arg).lower().strip() if arg else ""

        normalize = {
            "power save": "power-saver",
            "powersave": "power-saver",
            "power-saver": "power-saver",
            "performance": "performance",
            "balanced": "balanced",
        }

        target = normalize.get(profile)

        if not target:
            return f"I don't recognize the power profile '{profile}'."

        ok, err = _run(["powerprofilesctl", "set", target])

        if ok:
            return f"Power profile set to {target}."

        return (
            f"Couldn't set power profile to {target}. "
            "Make sure power-profiles-daemon is installed."
        )

    # Wi-Fi
    if name == "wifi_on":
        ok, err = _run(["nmcli", "radio", "wifi", "on"])
        return "Wi-Fi turned on." if ok else f"Couldn't turn on Wi-Fi: {err}"

    if name == "wifi_off":
        ok, err = _run(["nmcli", "radio", "wifi", "off"])
        return "Wi-Fi turned off." if ok else f"Couldn't turn off Wi-Fi: {err}"

    # Bluetooth
    if name == "bluetooth_on":
        ok, err = _run(["rfkill", "unblock", "bluetooth"])
        _run(["bluetoothctl", "power", "on"])
        return "Bluetooth turned on." if ok else f"Couldn't turn on Bluetooth: {err}"

    if name == "bluetooth_off":
        ok, err = _run(["rfkill", "block", "bluetooth"])
        _run(["bluetoothctl", "power", "off"])
        return "Bluetooth turned off." if ok else f"Couldn't turn off Bluetooth: {err}"

    # Power
    if name == "shutdown":
        subprocess.Popen(["systemctl", "poweroff"])
        return "Shutting down. Goodbye."

    if name == "reboot":
        subprocess.Popen(["systemctl", "reboot"])
        return "Rebooting now."

    if name == "lock":
        for locker in ["loginctl", "gnome-screensaver-command", "xdg-screensaver"]:
            if locker == "loginctl":
                ok, _ = _run(["loginctl", "lock-session"])
            elif locker == "gnome-screensaver-command":
                ok, _ = _run(["gnome-screensaver-command", "-l"])
            else:
                ok, _ = _run(["xdg-screensaver", "lock"])

            if ok:
                return "Screen locked."

        return "Couldn't lock the screen."

    if name == "suspend":
        subprocess.Popen(["systemctl", "suspend"])
        return "Suspending."

    # Screenshot
    if name == "screenshot":
        path = os.path.expanduser(
            f"~/Pictures/screenshot_{int(datetime.datetime.now().timestamp())}.png"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tools = [
            (["spectacle", "-b", "-n", "-f", "-o", path], "spectacle"),
            (["gnome-screenshot", "-f", path], "gnome-screenshot"),
            (["scrot", path], "scrot"),
            (["grim", path], "grim"),
        ]

        for cmd, tool in tools:
            if shutil.which(tool):
                ok, err = _run(cmd, timeout=10)
                if ok:
                    return "Screenshot taken."

        return "Couldn't take a screenshot."

    # Keyboard backlight
    if name == "keyboard_light_on":
        ok = False
        ok, _ = _run(["brightnessctl", "-d", "*kbd*", "set", "100%"])

        if not ok:
            try:
                for led in os.listdir("/sys/class/leds/"):
                    if "kbd" in led.lower() or "keyboard" in led.lower():
                        with open(f"/sys/class/leds/{led}/brightness", "w") as f:
                            f.write("1")
                        ok = True
            except Exception:
                pass

        return "Keyboard light turned on." if ok else "Couldn't turn on keyboard light."

    if name == "keyboard_light_off":
        ok = False
        ok, _ = _run(["brightnessctl", "-d", "*kbd*", "set", "0%"])

        if not ok:
            try:
                for led in os.listdir("/sys/class/leds/"):
                    if "kbd" in led.lower() or "keyboard" in led.lower():
                        with open(f"/sys/class/leds/{led}/brightness", "w") as f:
                            f.write("0")
                        ok = True
            except Exception:
                pass

        return "Keyboard light turned off." if ok else "Couldn't turn off keyboard light."

    # Search
    if name == "brave_search":
        if not arg:
            return "What would you like me to search for?"

        query = urllib.parse.quote_plus(str(arg))
        url = f"https://search.brave.com/search?q={query}"

        for cmd in [["brave-browser", url], ["brave", url]]:
            if shutil.which(cmd[0]):
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return f"Searching Brave for {arg}."
                except Exception:
                    pass

        try:
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Searching for {arg}."
        except Exception:
            return "I couldn't open a browser search."

    # Music
    if name == "play_song":
        music_dirs = [
            os.path.expanduser("~/Music"),
            os.path.expanduser("~/Downloads"),
        ]

        music_files = []

        for d in music_dirs:
            if os.path.isdir(d):
                music_files.extend(
                    glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True)
                )
                music_files.extend(
                    glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
                )
                music_files.extend(
                    glob.glob(os.path.join(d, "**", "*.flac"), recursive=True)
                )

        if music_files:
            chosen = random.choice(music_files)
            try:
                subprocess.Popen(
                    ["vlc", chosen],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return f"Playing {os.path.basename(chosen)}."
            except FileNotFoundError:
                return "VLC is not installed."
        else:
            try:
                subprocess.Popen(
                    ["vlc"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return "I couldn't find music files, but I opened VLC for you."
            except FileNotFoundError:
                return "VLC is not installed."

    # Open apps
    if name == "open_app":
        app = str(arg).lower().strip() if arg else ""

        if not app:
            return "What application should I open?"

        app_map = {
            "terminal": [["x-terminal-emulator"], ["gnome-terminal"], ["konsole"]],
            "vlc": [["vlc"]],
            "firefox": [["firefox"]],
            "chrome": [["google-chrome"], ["chromium"]],
            "brave": [["brave"], ["brave-browser"]],
            "code": [["code"]],
            "calculator": [["gnome-calculator"], ["kcalc"]],
            "files": [["nautilus"], ["dolphin"], ["thunar"]],
            "file manager": [["nautilus"], ["dolphin"], ["thunar"]],
            "spotify": [["spotify"]],
            "localsend": [["localsend_app"], ["localsend"]],
            "gedit": [["gedit"], ["gnome-text-editor"]],
            "gnome-software": [["gnome-software"]],
            "settings": [
                ["systemsettings5"],
                ["systemsettings"],
                ["gnome-control-center"],
                ["xdg-open", "settings://"],
            ],
            "calendar": [["gnome-calendar"], ["thunderbird"]],
        }

        candidates = app_map.get(app, [[app]])

        for cmd in candidates:
            exe = cmd[0]
            exe_path = shutil.which(exe)

            if exe_path:
                try:
                    full_cmd = [exe_path] + cmd[1:]

                    proc = subprocess.Popen(
                        full_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=os.environ.copy(),
                    )

                    time.sleep(0.4)

                    if proc.poll() is not None:
                        _, err = proc.communicate()
                        continue

                    return f"Opening {app}."
                except Exception:
                    continue

        return f"Sorry, I couldn't find the {app} application."

    # Small talk
    if name == "greeting":
        return random.choice([
            "Hello! How can I help?",
            "Hi there!",
            "Hey! What can I do for you?",
        ])

    if name == "status":
        return "I'm doing great, thanks for asking!"

    if name == "identity":
        return f"I'm {ASSISTANT_NAME}, your personal voice assistant."

    if name == "thanks":
        return random.choice([
            "You're welcome!",
            "No problem.",
            "Anytime.",
        ])

    # Fun
    if name == "joke":
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything.",
            "What do you call a fake noodle? An impasta.",
            "Why did the scarecrow win an award? Because he was outstanding in his field.",
            "What do you call a bear with no teeth? A gummy bear.",
        ]
        return random.choice(jokes)

    if name == "coin_flip":
        return random.choice(["Heads!", "Tails!"])

    if name == "roll_dice":
        return f"You rolled a {random.randint(1, 6)}."

    return "Sorry, I don't know that one yet."


# Prefer external intent.py / dispatcher.py if available.
try:
    from intent import classify as external_classify
    from dispatcher import execute as external_execute

    regex_classify = external_classify
    dispatcher_execute = external_execute

except Exception:
    regex_classify = built_in_classify
    dispatcher_execute = built_in_execute


# ============================================================
# OLLAMA TOOL CALLING SCHEMA
# ============================================================

TOOL_INTENTS = [
    "get_time", "get_date", "get_weather", "get_battery", "get_ip",
    "set_volume", "volume_up", "volume_down", "mute",
    "set_power_profile",
    "wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off",
    "shutdown", "reboot", "lock", "suspend",
    "screenshot", "keyboard_light_on", "keyboard_light_off",
    "brave_search", "play_song", "open_app",
    "greeting", "status", "identity", "thanks",
    "joke", "coin_flip", "roll_dice",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_intent",
            "description": (
                "Execute a local computer control action. "
                "Use this only when the user wants to control the device, "
                "open apps, change settings, search, or ask for system information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": TOOL_INTENTS,
                    },
                    "arg": {
                        "type": "string",
                        "description": (
                            "Optional argument such as volume number, app name, "
                            "power profile, or search query."
                        ),
                    },
                },
                "required": ["intent"],
            },
        },
    }
]


# ============================================================
# WAKE WORD
# ============================================================

class WakeWord:
    def __init__(self):
        self.model = None

        if HAVE_OPENWAKEWORD:
            try:
                self.model = OWWModel(wakeword_model_paths=[WAKEWORD_MODEL])
            except Exception as e:
                print("Wake word initialization error:", e)
                self.model = None

    def predict(self, frame: np.ndarray) -> dict:
        if self.model is None:
            return {}

        try:
            return self.model.predict(frame)
        except Exception:
            return {}

    def flush(self):
        if self.model is None:
            return

        silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)

        for _ in range(25):
            self.predict(silent_frame)


wake = WakeWord()


# ============================================================
# SILERO VAD
# ============================================================

class SileroVAD:
    def __init__(self):
        self.model = None
        self.buffer = np.empty(0, dtype=np.int16)

        if ENABLE_SILERO_VAD and HAVE_TORCH:
            try:
                self.model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                )
                self.model.eval()
            except Exception as e:
                print("Silero VAD unavailable, falling back to RMS VAD:", e)
                self.model = None

    def reset(self):
        self.buffer = np.empty(0, dtype=np.int16)

        if self.model is not None:
            try:
                self.model.reset_states()
            except Exception:
                pass

    def feed(self, frame: np.ndarray) -> float:
        """
        Feed a 1280-sample frame.
        Internally split into 512-sample chunks for Silero.
        Returns max speech probability for this frame.
        """
        if self.model is None:
            return 0.0

        frame = np.asarray(frame, dtype=np.int16)

        if self.buffer.size:
            self.buffer = np.concatenate([self.buffer, frame])
        else:
            self.buffer = frame.copy()

        max_prob = 0.0

        while self.buffer.size >= 512:
            chunk = self.buffer[:512]
            self.buffer = self.buffer[512:]

            try:
                tensor = torch.from_numpy(
                    chunk.astype(np.float32) / 32768.0
                )

                with torch.no_grad():
                    prob = self.model(tensor, SAMPLE_RATE)

                    if isinstance(prob, torch.Tensor):
                        prob = prob.item()

                    max_prob = max(max_prob, float(prob))

            except Exception:
                pass

        if self.buffer.size > SAMPLE_RATE:
            self.buffer = self.buffer[-512:]

        return max_prob


vad = SileroVAD()


# ============================================================
# SPEECH TO TEXT
# ============================================================

class SpeechToText:
    def __init__(self):
        self.backend = None
        self.model = None

        if HAVE_FASTER_WHISPER:
            try:
                self.model = WhisperModel(
                    FAST_WHISPER_MODEL,
                    device="cpu",
                    compute_type=FAST_WHISPER_COMPUTE,
                )
                self.backend = "faster-whisper"
                print("STT backend: Faster-Whisper")
                return
            except Exception as e:
                print("Faster-Whisper unavailable:", e)

        if HAVE_MOONSHINE:
            try:
                model_path, model_arch = get_model_for_language("en")

                self.model = Transcriber(
                    model_path=model_path,
                    model_arch=model_arch,
                )
                self.backend = "moonshine"
                print("STT backend: Moonshine")
                return
            except Exception as e:
                print("Moonshine unavailable:", e)

        print("WARNING: No STT backend available.")

    def transcribe(self, audio_float: np.ndarray) -> str:
        if audio_float.size == 0:
            return ""

        if self.backend == "faster-whisper":
            try:
                segments, _ = self.model.transcribe(
                    audio_float,
                    language="en",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=False,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                )

                text = " ".join(segment.text.strip() for segment in segments)
                return text.lower().strip()

            except Exception as e:
                print("Faster-Whisper transcription error:", e)
                return ""

        if self.backend == "moonshine":
            try:
                transcript = self.model.transcribe_without_streaming(
                    audio_data=audio_float.tolist(),
                    sample_rate=SAMPLE_RATE,
                )

                text = " ".join(line.text for line in transcript.lines).strip()
                return text.lower()

            except Exception as e:
                print("Moonshine transcription error:", e)
                return ""

        return ""


stt = SpeechToText()


# ============================================================
# TEXT TO SPEECH
# ============================================================

def split_sentences(text: str) -> list:
    chunks = []

    for line in str(text).splitlines():
        line = line.strip()

        if not line:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", line)
        chunks.extend(s.strip() for s in sentences if s.strip())

    return chunks or [str(text).strip()]


class PiperTTS:
    def __init__(self):
        from piper.voice import PiperVoice

        self.voice = PiperVoice.load(PIPER_MODEL)

    def speak(self, text: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_file = f.name

        try:
            with wave.open(wav_file, "wb") as wav:
                self.voice.synthesize_wav(text, wav)

            subprocess.run(
                ["aplay", wav_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if os.path.exists(wav_file):
                os.remove(wav_file)


class KokoroTTS:
    def __init__(self):
        from kokoro_onnx import Kokoro

        self.engine = Kokoro(KOKORO_MODEL, KOKORO_VOICES)

    def speak(self, text: str):
        samples, sample_rate = self.engine.create(
            text,
            voice=KOKORO_VOICE,
            speed=KOKORO_SPEED,
        )

        samples = np.asarray(samples)

        if samples.dtype != np.int16:
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_file = f.name

        try:
            with wave.open(wav_file, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(samples.tobytes())

            subprocess.run(
                ["aplay", wav_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if os.path.exists(wav_file):
                os.remove(wav_file)


class SpeechManager:
    def __init__(self):
        self.engines = []

        if TTS_BACKEND in ("auto", "kokoro") and HAVE_KOKORO:
            try:
                self.engines.append(KokoroTTS())
                print("TTS backend: Kokoro")
            except Exception as e:
                print("Kokoro TTS unavailable:", e)

        if TTS_BACKEND in ("auto", "piper") and HAVE_PIPER:
            try:
                self.engines.append(PiperTTS())
                print("TTS backend: Piper")
            except Exception as e:
                print("Piper TTS unavailable:", e)

    def speak(self, text: str):
        text = str(text).strip()
        if not text:
            return

        print(f"{ASSISTANT_NAME}: {text}")
        ui.update("speaking", text)

        for chunk in split_sentences(text):
            spoken = False

            for engine in self.engines:
                try:
                    engine.speak(chunk)
                    spoken = True
                    break
                except Exception as e:
                    print("TTS engine error:", e)

            if not spoken:
                print(chunk)


tts = SpeechManager()


# ============================================================
# AUDIO HELPERS
# ============================================================

def frame_rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def audio_callback(indata, frames, time_info, status):
    if status and DEBUG_AUDIO:
        print("Audio status:", status)

    if state["recording"]:
        command_q.put(indata[:, 0].copy())
    elif state["listening_wake"]:
        audio_q.put(indata[:, 0].copy())


def flush_wakeword_buffer():
    wake.flush()

    for q in (audio_q, command_q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


def wait_for_wake_word():
    ui.update("waiting", f"Say '{ASSISTANT_NAME}'")

    if wake.model is None:
        print("Wake word disabled. Press Enter to speak...")
        try:
            input()
        except Exception:
            time.sleep(1)
        return

    print("👂 Waiting for wake word...")

    consecutive_hits = 0

    while True:
        try:
            frame = audio_q.get(timeout=1.0)
        except queue.Empty:
            continue

        predictions = wake.predict(frame)
        rms = frame_rms(frame)

        hit = False

        for _, score in predictions.items():
            if score > WAKEWORD_THRESHOLD and rms > WAKEWORD_MIN_RMS:
                hit = True
                break

        if hit:
            consecutive_hits += 1

            if consecutive_hits >= WAKEWORD_CONFIRM_FRAMES:
                print("✨ Wake word detected")
                return
        else:
            consecutive_hits = 0


def listen_for_followup(timeout: float):
    """
    Keep the mic open for `timeout` seconds after a response.
    If the user starts speaking, return the captured frames so the
    command can be processed WITHOUT saying the wake word again.
    Returns None on silence.
    """
    # Echo guard: let the assistant's own voice die out first
    time.sleep(FOLLOWUP_GUARD_SECONDS)

    while not command_q.empty():
        try:
            command_q.get_nowait()
        except queue.Empty:
            break

    state["recording"] = True
    vad.reset()

    frames = []
    speech_started = False
    speech_frames = 0
    deadline = time.time() + timeout

    print("👂 Listening for follow-up command...")
    ui.update("waiting", "follow-up window")

    while time.time() < deadline:
        try:
            frame = command_q.get(timeout=0.25)
        except queue.Empty:
            continue

        frames.append(frame)
        rms = frame_rms(frame)

        if vad.model is not None:
            prob = vad.feed(frame)
            is_speech = prob >= SILERO_THRESHOLD and rms >= SILERO_MIN_RMS
        else:
            is_speech = rms >= RMS_SPEECH_THRESHOLD

        if is_speech:
            speech_started = True
            speech_frames += 1

            if speech_frames >= 2:
                return frames  # hand over to record_audio()
        else:
            speech_frames = 0

        if not speech_started and len(frames) > 25:
            frames = frames[-12:]  # keep only recent silence

    state["recording"] = False
    return None


def process_audio(audio: np.ndarray) -> bool:
    """Transcribe + handle one utterance. Returns False on exit command."""
    command = stt.transcribe(audio)

    if not command:
        tts.speak("I didn't hear anything.")
        return True

    command = remove_wake_word(command)

    if not command:
        tts.speak("Yes?")
        return True

    print(f"You: {command}")
    ui.update("thinking", command)

    if (
        re.search(r"\b(goodbye|bye|exit|quit)\b", command)
        and len(command.split()) <= 3
    ):
        tts.speak("Goodbye.")
        return False

    handle_command(command)
    return True


def record_audio(pre_frames=None) -> np.ndarray:
    """
    Record until endpointing detects end of speech.
    Uses Silero VAD if available, otherwise RMS VAD.
    """
    ui.update("listening", "Speak now")

    vad.reset()

    use_silero = vad.model is not None

    silence_frames_needed = int(
        VAD_SILENCE_DURATION * SAMPLE_RATE / FRAME_SAMPLES
    )
    min_speech_frames = int(
        VAD_MIN_SPEECH_SECONDS * SAMPLE_RATE / FRAME_SAMPLES
    )
    max_frames = int(
        VAD_MAX_SECONDS * SAMPLE_RATE / FRAME_SAMPLES
    )
    no_speech_timeout_frames = int(
        VAD_NO_SPEECH_TIMEOUT * SAMPLE_RATE / FRAME_SAMPLES
    )

    frames = list(pre_frames) if pre_frames else []
    consecutive_silence = 0
    total_frames = len(frames)
    speech_started = bool(pre_frames)

    t_start = time.time()

    while total_frames < max_frames:
        try:
            frame = command_q.get(timeout=1.0)
        except queue.Empty:
            continue

        frames.append(frame)
        total_frames += 1

        rms = frame_rms(frame)

        if use_silero:
            prob = vad.feed(frame)
            speech_now = (
                prob >= SILERO_THRESHOLD
                and rms >= SILERO_MIN_RMS
            )
        else:
            speech_now = rms >= RMS_SPEECH_THRESHOLD

        if DEBUG_AUDIO and total_frames % 10 == 0:
            print(
                f"[rec] f={total_frames} rms={rms:.0f} "
                f"speech={speech_started} sil={consecutive_silence}"
            )

        if speech_now:
            speech_started = True
            consecutive_silence = 0
        elif speech_started:
            consecutive_silence += 1

        if (
            speech_started
            and total_frames >= min_speech_frames
            and consecutive_silence >= silence_frames_needed
        ):
            break

        if not speech_started and total_frames >= no_speech_timeout_frames:
            break

        if time.time() - t_start > VAD_MAX_SECONDS:
            break

    state["recording"] = False

    if frames:
        audio_int16 = np.concatenate(frames)
    else:
        audio_int16 = np.zeros(FRAME_SAMPLES, dtype=np.int16)

    audio_float = audio_int16.astype(np.float32) / 32768.0

    duration = len(audio_float) / SAMPLE_RATE
    print(f"[rec] done: {duration:.2f}s")

    return audio_float


def remove_wake_word(text: str) -> str:
    for word in WAKE_WORDS:
        text = re.sub(
            rf"\b(hey|hello|ok|okay)?\s*{word}\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

    return text.strip(" ,.!")


# ============================================================
# OLLAMA HELPERS
# ============================================================

def ollama_options(num_predict: int = 60) -> dict:
    return {
        "temperature": 0.2,
        "num_predict": num_predict,
        "top_k": 20,
        "top_p": 0.8,
        "repeat_penalty": 1.05,
        "num_ctx": 2048,
    }


def disable_thinking_if_needed(payload: dict, model: str) -> dict:
    if "qwen3" in model.lower():
        payload["think"] = False

    return payload


def memory_context(command: str) -> str:
    if not MEMORY_ENABLED:
        return ""

    memories = memory.search(command, limit=MEMORY_LIMIT)

    if not memories:
        return ""

    return " Relevant memories: " + " | ".join(memories)


def likely_command(command: str) -> bool:
    words = set(re.findall(r"[a-z0-9']+", command.lower()))

    if len(words) <= 5:
        return True

    return any(word in COMMAND_HINTS for word in words)


def ask_llm_tools(command: str):
    """
    Ask Ollama to choose a local tool/intent.
    Returns a spoken response if successful, otherwise None.
    """
    if not USE_LLM_TOOLS:
        return None

    system_prompt = (
        f"You are {ASSISTANT_NAME}, a concise voice assistant. "
        "Use execute_intent when the user wants to control the computer. "
        "If no device control is needed, answer briefly for speech. "
        "Do not use markdown."
    )

    system_prompt += memory_context(command)
    system_prompt += rag_context(command)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": command},
    ]

    payload = {
        "model": LLM_SMART_MODEL,
        "messages": messages,
        "stream": False,
        "tools": TOOLS,
        "keep_alive": "30m",
        "options": ollama_options(100),
    }

    payload = disable_thinking_if_needed(payload, LLM_SMART_MODEL)

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            print("Ollama tool error:", response.text[:300])
            return None

        data = response.json()
        message = data.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        replies = []

        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name")
            args = function.get("arguments") or {}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if not isinstance(args, dict):
                args = {}

            if name != "execute_intent":
                continue

            intent_name = str(args.get("intent") or "").strip()
            arg_value = args.get("arg")

            if arg_value is None:
                for key in ("arg", "value", "level", "app", "query", "search"):
                    if key in args:
                        arg_value = str(args[key])
                        break

            if intent_name in TOOL_INTENTS:
                try:
                    reply = dispatcher_execute({
                        "intent": intent_name,
                        "arg": arg_value,
                    })

                    reply = str(reply or "").strip()

                    if reply:
                        replies.append(reply)

                except Exception as e:
                    print("Tool execution error:", e)

        if replies:
            return " ".join(replies)

        content = str(message.get("content") or "").strip()

        if content:
            return content

    except Exception as e:
        print("Ollama tool exception:", e)

    return None


def ask_llm_stream(question: str):
    """
    Conversational fallback with streaming TTS.
    """
    system_prompt = (
        f"You are {ASSISTANT_NAME}, a friendly voice assistant. "
        "Give concise answers suitable for speech. "
        "Do not use markdown. Keep responses under 2 sentences."
    )

    system_prompt += memory_context(question)
    system_prompt += rag_context(question)

    history = _conversation_history[-CHAT_HISTORY_SESSION:] + [
        {"role": "user", "content": question}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    payload = {
        "model": LLM_FAST_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",
        "options": ollama_options(60),
    }

    payload = disable_thinking_if_needed(payload, LLM_FAST_MODEL)

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=30,
        )

        if response.status_code != 200:
            print("Ollama error:", response.text[:300])
            tts.speak("There is a problem with the language model.")
            return

        buffer = ""
        full_reply = ""

        for line in response.iter_lines():
            if not line:
                continue

            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            token = chunk.get("message", {}).get("content", "")

            if not token:
                continue

            buffer += token
            full_reply += token

            match = re.search(r"([.!?])(\s+|$)", buffer)

            if match:
                end_idx = match.end(1)
                to_speak = buffer[:end_idx].strip()
                buffer = buffer[end_idx:]

                if to_speak:
                    tts.speak(to_speak)

        if buffer.strip():
            tts.speak(buffer.strip())

        remember_exchange(question, full_reply.strip())

    except Exception as e:
        print("Ollama exception:", e)
        tts.speak("Sorry, I couldn't connect to my brain.")


# ============================================================
# COMMAND HANDLING
# ============================================================

def handle_command(command: str):
    ui.update("thinking", command)

    parts = re.split(r"\band\b|\bthen\b|,", command)
    parts = [p.strip() for p in parts if p.strip()]

    remember_re = re.compile(r"^(?:please\s+)?remember\b", re.IGNORECASE)

    if len(parts) > 1:
        intents = [regex_classify(p) for p in parts]

        all_known = all(
            i.get("intent") not in (None, "unknown")
            for i in intents
        )

        has_remember = any(remember_re.match(p) for p in parts)

        if all_known or has_remember:
            for part in parts:
                _handle_single_command(part)
            return

    _handle_single_command(command)


def _handle_single_command(command: str):
    command = command.strip()

    if not command:
        return

    # Explicit memory commands
    remember_match = re.match(
        r"^(?:please\s+)?remember\s+(?:that\s+)?(.+)$",
        command,
        re.IGNORECASE,
    )

    if remember_match:
        memory.add(remember_match.group(1).strip())
        tts.speak("I will remember that.")
        return

    if re.search(r"\bforget everything\b", command, re.IGNORECASE):
        memory.clear()
        tts.speak("Memory cleared.")
        return

    if re.search(r"\bwhat do you remember\b", command, re.IGNORECASE):
        items = memory.recent(3)

        if not items:
            tts.speak("I don't have any saved memories yet.")
        else:
            tts.speak("I remember: " + " | ".join(items))

        return

    # Fast regex intent path
    intent = regex_classify(command)

    if intent.get("intent") not in (None, "unknown"):
        reply = str(dispatcher_execute(intent) or "").strip()

        if reply:
            tts.speak(reply)

        remember_exchange(command, reply)
        return

    # LLM tool-calling path
    if USE_LLM_TOOLS and likely_command(command):
        reply = ask_llm_tools(command)

        if reply:
            tts.speak(reply)
            remember_exchange(command, reply)
            return

    # Conversational fallback (logs itself)
    ask_llm_stream(command)


def _play_beep(freq=880, duration=0.12):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(tone.tobytes())

        subprocess.run(
            ["aplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


def acknowledge_wake():
    """
    Audible confirmation right after the wake word.
    Safe from self-hearing: at this point both mic flags are False,
    so microphone frames are discarded while TTS plays.
    """
    if WAKE_ACK_MODE == "none":
        return

    if WAKE_ACK_MODE == "beep":
        _play_beep()
        return

    tts.speak(random.choice(WAKE_ACK_PHRASES))


def warmup_llm():
    """
    Load the LLM into RAM in the background at startup
    so the first spoken answer is not delayed by model loading.
    """
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_FAST_MODEL,
                "messages": [{"role": "user", "content": "warmup"}],
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        print("[warmup] LLM loaded into RAM")
    except Exception:
        pass


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print()
    print("================================")
    print(f"        {ASSISTANT_NAME.upper()} ASSISTANT")
    print("================================")
    print()

    if stt.backend is None:
        print("WARNING: No speech-to-text backend available.")

    # Reload recent chat from disk so memory survives restarts
    if MEMORY_ENABLED:
        previous = memory.load_chat(CHAT_HISTORY_LOAD)
        if previous:
            _conversation_history.extend(previous)
            print(f"Memory: loaded {len(previous)} previous messages.")

    # Pre-load the LLM in the background (no startup delay)
    threading.Thread(target=warmup_llm, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        latency="low",
        callback=audio_callback,
    ):
        running = True

        while running:
            # ---- Wake word phase ----
            state["listening_wake"] = True
            state["recording"] = False

            time.sleep(0.2)

            wait_for_wake_word()

            state["listening_wake"] = False

            acknowledge_wake()  # Alexa answers the wake word

            while not command_q.empty():
                try:
                    command_q.get_nowait()
                except queue.Empty:
                    break

            state["recording"] = True
            audio = record_audio()

            running = process_audio(audio)

            if not running:
                break

            flush_wakeword_buffer()

            # ---- Follow-up phase: accept chained commands ----
            while running:
                follow = listen_for_followup(FOLLOWUP_WINDOW_SECONDS)

                if follow is None:
                    break  # silence -> back to wake word

                state["recording"] = True
                audio = record_audio(pre_frames=follow)

                running = process_audio(audio)

                if not running:
                    break

            flush_wakeword_buffer()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Stopped.")