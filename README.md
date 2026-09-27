# 🎙️ Local Voice Assistant (Alexa)

A fully offline, fast, and privacy-focused voice assistant for Linux. It uses **openWakeWord** for wake word detection, **Faster-Whisper** for speech-to-text, and **Piper** for text-to-speech. No cloud APIs, no LLMs—just quick, local responses.

## ✨ Features
- **100% Offline:** No data leaves your machine.
- **Fast Response:** Uses `faster-whisper` (base.en model) and in-process Piper TTS for minimal latency.
- **Custom Intents:** Handles time, date, weather, app launching, volume control, and more.
- **Wake Word:** Triggered by saying "Alexa".
- **Echo Cancellation Support:** Works seamlessly with PipeWire/PulseAudio to prevent the assistant from hearing itself.

## 📋 Prerequisites
- **OS:** Linux (tested on Ubuntu 22.04+)
- **Audio:** PipeWire or PulseAudio
- **Python:** 3.10 or higher

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/alexa.git
   cd alexa
```

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

📦 Model Setup

This repository does not include the large AI models. You must download them manually and place them in the correct folders.

1. Create the folders:
   ```bash
   mkdir -p voices wakeword
   ```
2. Piper TTS Model:
   · Download en_US-hfc_female-medium.onnx (and its .json file) from the Piper Voices page.
   · Place both files in the voices/ folder.
3. OpenWakeWord Model:
   · Download alexa_v0.1.onnx from the openWakeWord GitHub releases.
   · Place it in the wakeword/ folder.

🎧 Recommended Audio Setup (Echo Cancellation)

To prevent the assistant from hearing its own voice through your speakers, it is highly recommended to enable echo cancellation.

```bash
pactl load-module module-echo-cancel aec_method=webrtc
```

Then, open pavucontrol, go to Input Devices, and select "Noise Canceling source" as your default microphone.

🎮 Usage

Run the assistant:

```bash
python3 alexa.py
```

Say "Alexa" followed by your command.

Supported Commands

Category Examples
Time & Date "What time is it?", "What's the date?"
Apps "Open terminal", "Open vlc", "Open firefox"
System "Volume up", "Volume down", "Mute"
Info "What's the weather?", "Battery level"
Fun "Tell me a joke", "Flip a coin", "Roll a dice"
Chat "Hello", "How are you?", "Who are you?"

🛠️ Troubleshooting

· Assistant doesn't hear me: Check your mic gain in alsamixer (press F4 for capture). Aim for silence to read RMS ~50-300, and speech ~1000-5000.
· False wakes: Raise WAKEWORD_THRESHOLD in alexa.py (default is 0.4).
· Cuts off too early: Increase VAD_SILENCE_DURATION (default is 0.3 seconds).

📜 Credits

· openWakeWord for wake word detection.
· Faster-Whisper for speech recognition.
· Piper for text-to-speech.```
