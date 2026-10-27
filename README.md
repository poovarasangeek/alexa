# 🎙️ Alexa — 100% Local Voice Assistant

A private, offline‑first voice assistant for Linux that runs entirely on your own hardware.
Say **"Alexa"**, speak naturally, and control your machine, chat, and remember things — with **no cloud, no telemetry, and no API keys**.

Built with a warm, conversational *girlfriend* persona, an adaptive noise‑floor microphone pipeline, and a fast regex intent engine backed by a local LLM for everything the rules don't cover.

> Runs comfortably on a laptop CPU (tested on Intel i5‑12450H / 16 GB RAM / KDE Plasma).

---

## ✨ Highlights

| Area | What it does |
|---|---|
| **Wake word** | `openWakeWord` ("alexa") with confirmation frames + adaptive RMS gate |
| **Voice activity** | Silero VAD **plus** a self‑tuning noise‑floor tracker that adapts to quiet *and* noisy rooms automatically |
| **Speech‑to‑text** | `faster‑whisper` (`distil-small.en`, int8) with built‑in VAD filter |
| **Text‑to‑speech** | `piper` (local, real‑time) — markdown/asterisks stripped so it never reads symbols aloud |
| **Brain** | `llama.cpp` server (OpenAI‑compatible API) running **Qwen2.5‑1.5B‑Instruct Q4_K_M**, streamed sentence‑by‑sentence |
| **Intent engine** | Instant regex fast‑path for ~40 system commands; LLM only for open‑ended conversation |
| **Memory** | SQLite **FTS5** long‑term memory + persistent chat log that survives reboots |
| **RAG** | Indexes your personal `.txt` / `.md` notes and injects relevant facts into every reply |
| **Persona** | Time‑aware affectionate greetings (long on first wake, short after) + strict "never say I'm an AI" prompting |
| **UX** | Follow‑up window (chain commands without re‑waking), echo guard, optional terminal "hologram" UI |
| **Ops** | systemd user services for auto‑start + a one‑word `alexa on` / `alexa off` terminal command |

> 🔒 **Privacy note:** Everything is local *except* three opt‑in commands — `get_weather` (wttr.in), `get_ip` (ifconfig.me), and `web_search` (opens a local browser to a search URL). Delete those intents in the code for zero network use.

---

## 🧠 How it works

```
        ┌──────────────┐
        │  "Alexa ..." │  (microphone, always listening)
        └──────┬───────┘
               ▼
   ┌───────────────────────┐
   │ openWakeWord + RMS    │  adaptive noise-floor gate
   └──────────┬────────────┘
              ▼  (wake detected)
   ┌───────────────────────┐
   │ Girlfriend greeting   │  long first time, short after
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ Silero VAD recording  │  endpointing (start/stop on speech)
   └──────────────────────┘
              ▼
   ┌───────────────────────┐
   │ faster-whisper STT    │  distil-small.en (int8) + vad_filter
   └──────────┬────────────┘
              ▼
   ┌───────────────────────────────────────────┐
   │ Regex intent fast-path?  ── yes ─► execute│  instant, no LLM
   │            │ no                            │
   │            ▼                               │
   │ llama.cpp stream (girlfriend persona)      │  conversational fallback
   └──────────┬────────────────────────────────┘
              ▼
   ┌───────────────────────┐
   │ piper TTS             │  markdown stripped, sentence-streamed
   └──────────┬────────────┘
              ▼
        follow-up window (6s) ──► back to recording, no wake word needed
```

---

## 📦 Requirements

### Hardware
- x86‑64 Linux laptop/desktop (Ubuntu 24.04 / KDE tested; works on GNOME/XFCE too)
- ≥ 8 GB RAM (16 GB recommended), a working microphone & speaker
- No GPU required — everything runs on CPU

### System packages
```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  pulseaudio-utils alsa-utils \
  network-manager rfkill bluez \
  power-profiles-daemon \
  brightnessctl \
  vlc \
  xdg-utils
# screenshot tool (pick one you have)
sudo apt install -y gnome-screenshot   # or: spectacle / scrot / grim
```

### Python packages
```bash
python3 -m venv ~/alexa/venv
source ~/alexa/venv/bin/activate
pip install --upgrade pip
pip install numpy requests sounddevice torch faster-whisper piper-tts openwakeword
```

### llama.cpp
Install the pre‑built `llama-server` binary (no compilation needed):
```bash
cd ~
wget -O llama-bin.zip "https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b6544-bin-ubuntu-x64.zip"
unzip -o llama-bin.zip -d llama-cpp
sudo cp llama-cpp/llama-server /usr/local/bin/
llama-server --version
```

---

## 🗂️ Project layout

```
~/alexa/
├── alexa.py                  # the assistant (this repo's main file)
├── venv/                     # python virtualenv
├── voices/
│   ├── en_GB-jenny_dioco-medium.onnx
│   └── en_GB-jenny_dioco-medium.onnx.json   # ← REQUIRED, else "# channels not specified"
├── wakeword/
│   └── alexa_v0.1.onnx
├── knowledge/                # drop your .txt / .md notes here for RAG
│   └── about-me.txt
└── (runtime) ~/.alexa_memory.db   # SQLite memory + chat log + KB index

~/models/
└── qwen2.5-1.5b-instruct-q4_k_m.gguf   # the LLM brain
```

---

## ⚙️ Setup

### 1. Create folders
```bash
mkdir -p ~/alexa/{voices,wakeword,knowledge} ~/models
```

### 2. Piper voice (model **+ json config**)
Both files must share the same basename in `~/alexa/voices/`:
```bash
cd ~/alexa/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
```
> Prefer a different voice? Download any `*-high.onnx` + matching `.json` (e.g. `en_US-amy-high`, `en_US-lessac-high`) and update `PIPER_MODEL` in `alexa.py`. To slow a fast voice down, pass `length_scale=1.3` to `synthesize_wav(...)`.

### 3. Wake word model
Place your `alexa_v0.1.onnx` in `~/alexa/wakeword/`. (Train your own with [openWakeWord](https://github.com/dscripka/openWakeWord) or use a community model.)

### 4. LLM model
```bash
# fetch any Qwen2.5-1.5B-Instruct Q4_K_M GGUF into ~/models/
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ~/models
```
(`faster-whisper`'s `distil-small.en` and Silero VAD download automatically on first run.)

### 5. (Optional) Personal notes for RAG
Put plain text / markdown facts about yourself in `~/alexa/knowledge/`. They are indexed at every startup and fed to the LLM as context.

### 6. (Required for brightness) Passwordless backlight control
Alexa runs as your user via systemd, so it cannot type a `sudo` password. Grant it once:
```bash
echo "ALL ALL=(ALL) NOPASSWD: /usr/bin/brightnessctl" | sudo tee /etc/sudoers.d/brightnessctl
sudo chmod 440 /etc/sudoers.d/brightnessctl
```
> Verify the binary path first with `which brightnessctl` and adjust the rule if it differs.

---

## ▶️ Running it

### Terminal 1 — start the brain
```bash
llama-server -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 4096 -t 8 --host 127.0.0.1 --port 8080
```
Wait for `listening on http://127.0.0.1:8080`.

### Terminal 2 — start Alexa
```bash
cd ~/alexa
./venv/bin/python alexa.py
```
You should see:
```
VAD backend: Silero ONNX
STT backend: Faster-Whisper (distil-small.en)
TTS backend: Piper
Knowledge base: N chunks indexed from /home/<you>/alexa/knowledge
Memory: loaded X previous messages.
👂 Waiting for wake word...
```
Say **"Alexa"** → she greets you → speak a command.

---

## 🤖 Auto‑start with systemd + `alexa on|off`

Create **user‑level** services so the assistant (and its brain) launch on login and can be toggled with one word.

`~/.config/systemd/user/llama-server.service`
```ini
[Unit]
Description=llama.cpp server for Alexa
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/llama-server -m %h/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 4096 -t 8 --host 127.0.0.1 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/alexa.service`
```ini
[Unit]
Description=Alexa Voice Assistant
After=llama-server.service
Wants=llama-server.service

[Service]
Type=simple
WorkingDirectory=%h/alexa
# GUI apps (settings, browser) need the display + dbus session:
Environment=DISPLAY=:0
Environment=XAUTHORITY=%h/.Xauthority
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus
ExecStartPre=/bin/sleep 15
ExecStart=%h/alexa/venv/bin/python %h/alexa/alexa.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

Enable & start:
```bash
systemctl --user daemon-reload
systemctl --user enable --now llama-server.service
systemctl --user enable --now alexa.service
```

One‑word control script — `~/.local/bin/alexa`:
```bash
#!/bin/bash
case "$1" in
  on)     systemctl --user start llama-server.service; sleep 8; systemctl --user start alexa.service; echo "✅ Alexa online." ;;
  off)    systemctl --user stop alexa.service; systemctl --user stop llama-server.service; echo "🛑 Alexa offline." ;;
  status) systemctl --user status llama-server.service alexa.service --no-pager | grep -E "Active:|●" ;;
  *)      echo "Usage: alexa [on|off|status]" ;;
esac
```
```bash
chmod +x ~/.local/bin/alexa
```
Now from any terminal: `alexa on`, `alexa off`, `alexa status`.
Logs: `journalctl --user -u alexa.service -f`

---

## 🗣️ Things you can say

**System control** (instant regex path — no LLM latency):
`what time is it` · `what's today's date` · `what's the weather` · `battery level` · `my ip` ·
`set volume to 50` / `volume up` / `mute` · `set brightness to 60%` ·
`turn on/off wifi` · `turn on/off bluetooth` · `set performance / balanced / power-saver mode` ·
`take a screenshot` · `lock the screen` · `suspend` · `reboot` · `shut down` ·
`turn on/off keyboard backlight` ·
`open terminal / files / settings / calculator / vlc / spotify / code / browser / calendar / localsend` ·
`search for <topic>` · `play some music`

**Memory:**
`remember that <fact>` · `what do you remember` · `forget everything` · `what did I ask before?`

**Conversation / fun** (LLM path, girlfriend persona):
`how are you` · `who are you` · `who developed you` · `tell me a joke` · `flip a coin` · `roll a dice` · anything else.

**Session:** say `goodbye` / `exit` to quit. Within ~6 s after a reply you can keep talking **without** repeating "Alexa".

---

## 🎭 Customizing the persona

Three places control her personality:

1. **Greetings** — edit `get_girlfriend_greeting()`. A session counter (`_session_wake_count`) gives a long, time‑aware line on the **first** wake and short `short_phrases` afterwards, so it never gets repetitive.
2. **Conversation brain** — edit the `system_prompt` inside `ask_llm_stream()`. It enforces the girlfriend identity, names **Poovarasan** as her creator, and hard‑bans `"As an AI..."` / markdown / asterisks / "Amazon".
3. **Identity short‑circuit** — the regex in `built_in_classify` catches `who are you / who made you / who developed you / who created you` and `built_in_execute` answers instantly, so the LLM is never even asked (zero chance of a wrong "developer").

To swap the voice, change `PIPER_MODEL`; to change the brain, change the `llama-server -m ...` path and `LLM_MODEL`.

---

## 🔬 Under the hood (design notes)

- **Adaptive noise floor** — `NoiseFloorTracker` keeps a sliding window of recent frame RMS values and estimates the room's noise as the ~10th percentile, smoothed over time. Wake‑word and speech thresholds derive from it (`floor × 1.8` and `floor × 3.5`), so a loud café raises the bar (fewer false wakes) while a silent bedroom lowers it (easier wakes) — no manual tuning.
- **Two‑tier intent routing** — deterministic commands never touch the LLM, keeping them sub‑second and 100% reliable; only open‑ended questions stream from llama.cpp, sentence‑by‑sentence, so she starts talking before the full answer is generated.
- **Markdown sanitization** — `SpeechManager.speak()` strips `* _ \` #` before synthesis, so even if the model leaks formatting you'll never hear "asterisk".
- **No barge‑in (by design)** — the mic is muted while she speaks, and an echo guard ignores the tail of her own voice, preventing self‑triggering loops.
- **Persistent memory** — chat is logged to SQLite and reloaded at startup, so context survives `alexa off` / reboots; FTS5 gives fast keyword recall over both saved facts and your RAG notes.
- **Silence guard** — `vad_filter=True` in faster‑whisper plus a minimum‑length check stops background noise from being transcribed into fake commands.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `# channels not specified` from Piper | The voice `.json` config is missing next to the `.onnx`. Download both files (Setup §2). |
| She answers silence / random sentences | Ensure `vad_filter=True` in `SpeechToText.transcribe()` (already set); the noise‑floor tracker also mitigates this. |
| `open settings` works in terminal but not via systemd | The service needs `DISPLAY` / `XAUTHORITY` / `DBUS_SESSION_BUS_ADDRESS` (included in the unit above). |
| Brightness command fails | Apply the passwordless `sudoers` rule (Setup §6); confirm `sudo brightnessctl set 50%` works in a terminal first. |
| `llama-server: command not found` | Install the pre‑built binary (Requirements → llama.cpp) and ensure `/usr/local/bin` is in `PATH`. |
| LLM says "As an AI…" or "made by Amazon" | Keep the strict `system_prompt` and low `temperature` (the code uses `0.1`); the regex identity short‑circuit also prevents this. |
| Wake word too sensitive / not sensitive | Tune `WAKEWORD_THRESHOLD` and `WAKEWORD_CONFIRM_FRAMES` in CONFIG; the adaptive floor handles room noise automatically. |
| systemd says `bad unit file setting` | The unit has a syntax error (tabs, stray spaces, `~` instead of `%h`). Recreate it from the blocks above and run `systemctl --user daemon-reload`. |

---

## License

Licensed under the **Apache License, Version 2.0** — see the [`LICENSE`](LICENSE) file.

In plain terms: you may use, copy, modify, merge, publish, distribute, sublicense,
and sell copies of this software freely. The conditions are simply that you
**keep my name (Poovarasan) and the license/attribution notices with the code**,
mark any files you change, and include a copy of the license with redistributions.
Apache 2.0 also grants an explicit patent license from every contributor.

> In short: use it freely, just keep my name on it. 💜
<p align="center"><sub>Built locally, for local use. No clouds were harmed. 💜 — Poovarasan</sub></p>
