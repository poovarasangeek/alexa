import re

def classify(text: str) -> dict:
    text = text.lower().strip()

    # --- System Controls ---
    if re.search(r"\b(time|what time|current time)\b", text):
        return {"intent": "get_time"}

    if re.search(r"\b(date|what date|what day|today's date)\b", text):
        return {"intent": "get_date"}

    if re.search(r"\b(weather|forecast|temperature)\b", text):
        return {"intent": "get_weather"}

    if re.search(r"\b(battery|power level|charge)\b", text):
        return {"intent": "get_battery"}

    if re.search(r"\b(volume up|increase volume|turn it up)\b", text):
        return {"intent": "volume_up"}

    if re.search(r"\b(volume down|decrease volume|turn it down)\b", text):
        return {"intent": "volume_down"}

    if re.search(r"\b(mute|silence|stop sound)\b", text):
        return {"intent": "mute"}

    # --- Open Applications ---
    # Add more apps here as you need them
    apps = ["terminal", "vlc", "firefox", "chrome", "code", "calculator", "files", "spotify"]
    for app in apps:
        if re.search(rf"\b(open|launch|start|run)\s+{app}\b", text):
            return {"intent": "open_app", "arg": app}

    # --- Greetings & Small Talk ---
    if re.search(r"\b(hello|hi|hey|good morning|good evening)\b", text):
        return {"intent": "greeting"}

    if re.search(r"\b(how are you|how's it going)\b", text):
        return {"intent": "status"}

    if re.search(r"\b(who are you|what is your name)\b", text):
        return {"intent": "identity"}

    if re.search(r"\b(thank you|thanks)\b", text):
        return {"intent": "thanks"}

    # --- Fun ---
    if re.search(r"\b(joke|tell me a joke|make me laugh)\b", text):
        return {"intent": "joke"}

    if re.search(r"\b(flip a coin|coin flip|heads or tails)\b", text):
        return {"intent": "coin_flip"}

    if re.search(r"\b(roll a die|roll a dice|random number)\b", text):
        return {"intent": "roll_dice"}

    # --- Fallback ---
    return {"intent": "unknown"}
