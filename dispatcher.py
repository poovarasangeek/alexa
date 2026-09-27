import os
import random
import subprocess
import datetime
import urllib.request
import json

def execute(intent: dict) -> str:
    name = intent.get("intent")
    arg = intent.get("arg")

    if name == "get_time":
        now = datetime.datetime.now()
        return f"It's {now.strftime('%I:%M %p').lstrip('0')}."

    elif name == "get_date":
        now = datetime.datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    elif name == "get_weather":
        try:
            # wttr.in gives a simple text weather report based on IP location
            url = "https://wttr.in/?format=%C+%t"
            req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.68.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                weather = response.read().decode('utf-8').strip()
            return f"The weather is currently {weather}."
        except Exception:
            return "Sorry, I couldn't fetch the weather right now."

    elif name == "get_battery":
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery is None:
                return "I couldn't detect a battery on this device."
            status = "charging" if battery.power_plugged else "discharging"
            return f"The battery is at {int(battery.percent)}% and is {status}."
        except Exception:
            return "I couldn't read the battery status."

    elif name == "volume_up":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"])
        return "Volume increased."

    elif name == "volume_down":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"])
        return "Volume decreased."

    elif name == "mute":
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"])
        return "Audio muted or unmuted."

    elif name == "open_app":
        app_map = {
            "terminal": "x-terminal-emulator", # or "gnome-terminal", "konsole"
            "vlc": "vlc",
            "firefox": "firefox",
            "chrome": "google-chrome",
            "code": "code",
            "calculator": "gnome-calculator",
            "files": "nautilus",
            "spotify": "spotify"
        }
        cmd = app_map.get(arg, arg)
        try:
            subprocess.Popen([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opening {arg}."
        except FileNotFoundError:
            return f"Sorry, I couldn't find the {arg} application."
        except Exception as e:
            return f"Failed to open {arg}."

    elif name == "greeting":
        return random.choice(["Hello! How can I help?", "Hi there!", "Hey! What can I do for you?"])

    elif name == "status":
        return "I'm doing great, thanks for asking!"

    elif name == "identity":
        return "I'm Alexa, your personal voice assistant."

    elif name == "thanks":
        return random.choice(["You're welcome!", "No problem.", "Anytime."])

    elif name == "joke":
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything.",
            "What do you call a fake noodle? An impasta.",
            "Why did the scarecrow win an award? Because he was outstanding in his field.",
            "What do you call a bear with no teeth? A gummy bear."
        ]
        return random.choice(jokes)

    elif name == "coin_flip":
        return random.choice(["Heads!", "Tails!"])

    elif name == "roll_dice":
        return f"You rolled a {random.randint(1, 6)}."

    return "Sorry, I don't know that one yet."
