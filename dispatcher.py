import os
import random
import glob
import shutil
import subprocess
import datetime
import urllib.request
import urllib.parse
import time


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


def execute(intent: dict) -> str:
    name = intent.get("intent")
    arg = intent.get("arg")

    # ---------- TIME / DATE ----------
    if name == "get_time":
        now = datetime.datetime.now()
        return f"It's {now.strftime('%I:%M %p').lstrip('0')}."

    if name == "get_date":
        now = datetime.datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    # ---------- WEATHER ----------
    if name == "get_weather":
        try:
            url = "https://wttr.in/?format=%C+%t"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                weather = response.read().decode("utf-8").strip()
            return f"The weather is currently {weather}."
        except Exception:
            return "Sorry, I couldn't fetch the weather right now."

    # ---------- BATTERY ----------
    if name == "get_battery":
        try:
            for bat in ["BAT0", "BAT1"]:
                cap_path = f"/sys/class/power_supply/{bat}/capacity"
                status_path = f"/sys/class/power_supply/{bat}/status"
                if os.path.exists(cap_path):
                    with open(cap_path) as f:
                        percent = f.read().strip()
                    with open(status_path) as f:
                        status = f.read().strip().lower()
                    return f"The battery is at {percent}% and is {status}."
            return "I couldn't find a battery on this system."
        except Exception as e:
            print(f"  [battery error] {e}")
            return "I couldn't read the battery status."

    # ---------- IP ADDRESS ----------
    if name == "get_ip":
        try:
            url = "https://ifconfig.me/ip"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                ip = response.read().decode("utf-8").strip()
            return f"Your public IP address is {ip}."
        except Exception:
            return "Sorry, I couldn't fetch your IP address."

    # ---------- VOLUME ----------
    if name == "set_volume":
        try:
            vol = int(arg)
            if 0 <= vol <= 100:
                _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"])
                return f"Volume set to {vol} percent."
            return "Volume must be between 0 and 100."
        except (ValueError, TypeError):
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


    # ---------- POWER PROFILE ----------
    if name == "set_power_profile":
        profile = arg.lower().strip() if arg else ""
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
        else:
            print(f"  [power error] {err}")
            return f"Couldn't set power profile to {target}. Make sure power-profiles-daemon is installed."

    # ---------- WIFI ----------
    if name == "wifi_on":
        ok, err = _run(["nmcli", "radio", "wifi", "on"])
        return "Wi-Fi turned on." if ok else f"Couldn't turn on Wi-Fi: {err}"

    if name == "wifi_off":
        ok, err = _run(["nmcli", "radio", "wifi", "off"])
        return "Wi-Fi turned off." if ok else f"Couldn't turn off Wi-Fi: {err}"

    # ---------- BLUETOOTH ----------
    if name == "bluetooth_on":
        ok, err = _run(["rfkill", "unblock", "bluetooth"])
        _run(["bluetoothctl", "power", "on"])
        return "Bluetooth turned on." if ok else f"Couldn't turn on Bluetooth: {err}"

    if name == "bluetooth_off":
        ok, err = _run(["rfkill", "block", "bluetooth"])
        _run(["bluetoothctl", "power", "off"])
        return "Bluetooth turned off." if ok else f"Couldn't turn off Bluetooth: {err}"

    # ---------- POWER ----------
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

    # ---------- SCREENSHOT ----------
    if name == "screenshot":
        path = os.path.expanduser(f"~/Pictures/screenshot_{int(datetime.datetime.now().timestamp())}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tools = [
            (["spectacle", "-b", "-n", "-f", "-o", path], "spectacle"), # KDE
            (["gnome-screenshot", "-f", path], "gnome-screenshot"),     # GNOME
            (["scrot", path], "scrot"),                                 # Minimalist
            (["grim", path], "grim"),                                   # Wayland minimal
        ]

        for cmd, tool in tools:
            if shutil.which(tool):
                print(f"  [screenshot] Trying {tool}...")
                ok, err = _run(cmd, timeout=10)
                if ok:
                    return "Screenshot taken."
                else:
                    print(f"  [screenshot error] {tool} failed: {err}")
            else:
                print(f"  [screenshot] {tool} is not installed.")

        return "Couldn't take a screenshot. Please install spectacle, gnome-screenshot, scrot, or grim."
    
    # ---------- KEYBOARD BACKLIGHT ----------
    if name == "keyboard_light_on":
        # Try brightnessctl first (works on most modern Linux systems)
        ok, err = _run(["brightnessctl", "-d", "*kbd*", "set", "100%"])
        if not ok:
            # Fallback to sysfs (manual control)
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
        ok, err = _run(["brightnessctl", "-d", "*kbd*", "set", "0%"])
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

    # ---------- MEDIA / SEARCH ----------
    if name == "brave_search":
        if arg:
            query = urllib.parse.quote_plus(arg)
            url = f"https://search.brave.com/search?q={query}"
            for cmd in [["brave-browser", url], ["brave", url]]:
                if shutil.which(cmd[0]):
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return f"Searching Brave for {arg}."
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Searching for {arg}."
        return "What would you like me to search for?"

    if name == "play_song":
        music_dirs = [os.path.expanduser("~/Music"), os.path.expanduser("~/Downloads")]
        music_files = []
        for d in music_dirs:
            if os.path.isdir(d):
                music_files.extend(glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True))
                music_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
                music_files.extend(glob.glob(os.path.join(d, "**", "*.flac"), recursive=True))

        if music_files:
            chosen = random.choice(music_files)
            try:
                subprocess.Popen(["vlc", chosen], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Playing {os.path.basename(chosen)}."
            except FileNotFoundError:
                return "VLC is not installed."
        else:
            try:
                subprocess.Popen(["vlc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return "I couldn't find any music files, but I opened VLC for you."
            except FileNotFoundError:
                return "VLC is not installed."

    # ---------- OPEN APPS ----------
    if name == "open_app":
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
                ["systemsettings5"],                              # KDE Plasma 5
                ["systemsettings"],                               # KDE Plasma 6
                ["gnome-control-center"],                         # GNOME / Ubuntu
                ["xdg-open", "settings://"],                      # Universal fallback
            ],
            "calendar": [["gnome-calendar"], ["thunderbird"]],
        }

        candidates = app_map.get(arg, [[arg]])
        for cmd in candidates:
            exe = cmd[0]
            exe_path = shutil.which(exe)

            if exe_path:
                try:
                    # Create a copy of the command list with the absolute path
                    full_cmd = [exe_path] + cmd[1:]
                    print(f"  [open_app] Running: {' '.join(full_cmd)}")

                    proc = subprocess.Popen(
                        full_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=os.environ.copy()
                    )

                    # Give it a split second to launch
                    time.sleep(0.5)

                    # If it exited immediately, print the error
                    if proc.poll() is not None:
                        _, err = proc.communicate()
                        print(f"  [open_app error] {exe} crashed: {err.decode().strip()}")
                        continue

                    return f"Opening {arg}."

                except Exception as e:
                    print(f"  [open_app error] {exe}: {e}")
            else:
                print(f"  [open_app] {exe} is not installed.")

        return f"Sorry, I couldn't find the {arg} application."

    # ---------- SMALL TALK ----------
    if name == "greeting":
        return random.choice([
            "Hello! How can I help?",
            "Hi there!",
            "Hey! What can I do for you?",
        ])

    if name == "status":
        return "I'm doing great, thanks for asking!"

    if name == "identity":
        return "I'm Alexa, your personal voice assistant."

    if name == "thanks":
        return random.choice(["You're welcome!", "No problem.", "Anytime."])

    # ---------- FUN ----------
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
