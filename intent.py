import re

# Common Whisper mis-transcriptions -> correct word
CORRECTIONS = {
    r"\bblue\s*(tooth|hood|both|toof|tuth|two|to|thoot|truth)\b": "bluetooth",
    r"\bblow\s*(both|tooth|hood|to)\b": "bluetooth",
    r"\bblue\s*t\s*o\s*o\s*t\s*h\b": "bluetooth",
    r"\bblue boot": "bluetooth",
    r"\b(why\s*fi|wi\s*fi|we\s*fi|wife\s*i|whiffy|wifi)\b": "wifi",
    r"\bwire\s*less\b": "wireless",
    r"\blocal\s*send\b": "localsend",
    r"\blocals\s*end\b": "localsend",
    r"\bvallum\b": "volume",
    r"\bvolum\b": "volume",
    r"\b(brow\s*ser|browsers|browse\s*her)\b": "browser",
    r"\b(term\s*inal|terminals|turminel)\b": "terminal",
    r"\b(calc\s*ul\s*ator|calculater|calcu\s*lator)\b": "calculator",
    r"\b(v\s*l\s*c|vee\s*ell\s*see)\b": "vlc",
    r"\b(screen\s*shot|screenshots|screenshot)\b": "screenshot",
    r"\b(re\s*boot|reboo|rebute)\b": "reboot",
    r"\b(shut\s*down|shutdown|shut\s*dawn)\b": "shutdown",
}

def _normalize(text: str) -> str:
    """Fix common Whisper/Moonshine mis-transcriptions."""
    for pattern, replacement in CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def classify(text: str) -> dict:
    text = text.lower().strip()
    text = _normalize(text)

    # =========================================
    # SYSTEM POWER
    # =========================================
    if re.search(r"\b(shut ?down|power ?off|turn off (the )?(system|computer|laptop|pc)|power down)\b", text):
        return {"intent": "shutdown"}
    if re.search(r"\b(reboot|restart (the )?(system|computer|laptop|pc)|restart now)\b", text):
        return {"intent": "reboot"}
    if re.search(r"\b(lock (the )?(system|screen|computer|laptop|pc)|lock now)\b", text):
        return {"intent": "lock"}
    if re.search(r"\b(suspend|sleep (the )?(system|computer))\b", text):
        return {"intent": "suspend"}

    # =========================================
    # POWER PROFILES
    # =========================================
    if re.search(r"\b(power\s*save|powersave|power-saver|battery\s*save|battery\s*saver)\b", text):
        return {"intent": "set_power_profile", "arg": "power-saver"}
    if re.search(r"\b(performance|high\s*performance|max\s*performance)\b", text):
        return {"intent": "set_power_profile", "arg": "performance"}
    if re.search(r"\b(balanced|normal|default)\b", text):
        return {"intent": "set_power_profile", "arg": "balanced"}

    # =========================================
    # WIFI
    # =========================================
    if re.search(r"\b(turn on|enable|start|switch on)\s+(the\s+)?(wifi|wi-?fi|wireless)\b", text):
        return {"intent": "wifi_on"}
    if re.search(r"\b(turn off|disable|stop|switch off)\s+(the\s+)?(wifi|wi-?fi|wireless)\b", text):
        return {"intent": "wifi_off"}

    # =========================================
    # BLUETOOTH
    # =========================================
    if re.search(r"\b(turn on|enable|start|switch on)\s+(the\s+)?(bluetooth|blue tooth)\b", text):
        return {"intent": "bluetooth_on"}
    if re.search(r"\b(turn off|disable|stop|switch off)\s+(the\s+)?(bluetooth|blue tooth)\b", text):
        return {"intent": "bluetooth_off"}

    # =========================================
    # INFO
    # =========================================
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

    # =========================================
    # VOLUME (NEW: Specific percentage)
    # =========================================
    if re.search(r"\b(set|change|turn|make)\b.*\bvolume\b.*?(\d+)", text):
        match = re.search(r"\b(set|change|turn|make)\b.*\bvolume\b.*?(\d+)", text)
        return {"intent": "set_volume", "arg": match.group(2)}

    if re.search(r"\b(volume up|increase volume|turn it up|louder)\b", text):
        return {"intent": "volume_up"}
    if re.search(r"\b(volume down|decrease volume|turn it down|quieter)\b", text):
        return {"intent": "volume_down"}
    if re.search(r"\b(mute|silence|stop sound)\b", text):
        return {"intent": "mute"}

    # =========================================
    # MEDIA / SEARCH (NEW)
    # =========================================
    if re.search(r"\b(play|start|put on)\b.*\b(song|music)\b", text):
        return {"intent": "play_song"}

    # Search rules go BEFORE open_app to prevent "open brave search for x" from being parsed as just opening Brave
    search_match = re.search(r"\b(search|look up|google|find)\b\s*(for|about)?\s*(.+)", text)
    if search_match:
        return {"intent": "brave_search", "arg": search_match.group(3).strip()}

    # =========================================
    # SCREENSHOT
    # =========================================
    if re.search(r"\b(screenshot|screen shot|take a screenshot|capture screen)\b", text):
        return {"intent": "screenshot"}

    # =========================================
    # OPEN APPLICATIONS
    # =========================================
    apps = [
        "terminal", "vlc", "firefox", "chrome", "brave", "browser",
        "code", "vscode", "calculator", "files", "file manager",
        "nautilus", "spotify", "localsend", "local send", "text editor",
        "settings", "software center", "calendar", # <--- Added calendar
    ]
    for app in apps:
        if re.search(rf"\b(open|launch|start|run)\s+(the\s+)?{re.escape(app)}\b", text):
            normalized = {
                "browser": "brave",
                "file manager": "files",
                "local send": "localsend",
                "vscode": "code",
                "text editor": "gedit",
                "software center": "gnome-software",
            }.get(app, app)
            return {"intent": "open_app", "arg": normalized}

    # =========================================
    # SMALL TALK
    # =========================================
    if re.search(r"\b(hello|hi|hey|good morning|good evening)\b", text):
        return {"intent": "greeting"}
    if re.search(r"\b(how are you|how'?s it going)\b", text):
        return {"intent": "status"}
    if re.search(r"\b(who are you|what is your name)\b", text):
        return {"intent": "identity"}
    if re.search(r"\b(thank you|thanks)\b", text):
        return {"intent": "thanks"}

    # =========================================
    # FUN
    # =========================================
    if re.search(r"\b(joke|tell me a joke|make me laugh)\b", text):
        return {"intent": "joke"}
    if re.search(r"\b(flip a coin|coin flip|heads or tails)\b", text):
        return {"intent": "coin_flip"}
    if re.search(r"\b(roll a die|roll a dice|random number)\b", text):
        return {"intent": "roll_dice"}

    # =========================================
    # KEYBOARD BACKLIGHT
    # =========================================
    if re.search(r"\b(turn on|enable)\b.*\b(keyboard|kbd)\b.*\b(light|backlight|backlit)\b", text):
        return {"intent": "keyboard_light_on"}
    if re.search(r"\b(turn off|disable)\b.*\b(keyboard|kbd)\b.*\b(light|backlight|backlit)\b", text):
        return {"intent": "keyboard_light_off"}

    return {"intent": "unknown"}


