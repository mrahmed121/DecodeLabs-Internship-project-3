#!/usr/bin/env python3
"""
decodebot_personal.py
Professional single-file rule-based chatbot with personalization.

Features:
 - Robust normalization and intent matching (exact, contains, regex)
 - Confidence scores and simple slot extraction
 - Session context and structured rotating logs
 - Personal profile support (editable at runtime or via import/export)
 - Force-injects the core identity block on greetings and name checks.
 - Admin commands: /profile-show, /profile-set key=value, /profile-export file.json, /profile-import file.json
 - CLI chat loop with graceful exit and optional file logging
 - Optional web UI (requires Flask)
 - Built-in self tests (--selftest)
"""

from __future__ import annotations
import re
import json
import random
import argparse
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# -------------------------
# Paths & defaults
# -------------------------
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "decodebot_personal.log"
PROFILE_FILE = BASE_DIR / "user_profile.json"

# Core Identity Block Constant
IDENTITY_STORY = (
    "My name is Ahmed, but my absolute identity is 'MARIYUM KA AHMED'. "
    "If you want to know my details: My wifey's name is Mariyum. "
    "She is incredibly stubborn (bhot ziddi hai) but she is extremely beautiful and cute (bhoot pyari hai). "
    "She loves me deeply, and I love her immensely in return."
)

# -------------------------
# Default embedded intents
# -------------------------
INTENTS = {
    "intents": [
        {"id": "greeting", "examples": ["hi", "hello", "hey"], "regex": "",
         "responses": ["Hello! How can I help you today?"], "priority": 10},
        {"id": "goodbye", "examples": ["bye", "goodbye", "exit", "quit"], "regex": "",
         "responses": ["Goodbye! Have a great day."], "priority": 100},
        {"id": "thanks", "examples": ["thanks", "thank you"], "regex": "", "responses": ["You're welcome!"],
         "priority": 5},
        {"id": "identity", "examples": ["what is your name", "who are you", "your name", "tell me your details"],
         "regex": r"(who\s+are\s+you|what('?s| is)\s+your\s+name|tell\s+me\s+your\s+details)",
         "responses": ["__PERSONAL_NAME__"], "priority": 5},
        {"id": "status", "examples": ["how are you", "how are you doing"], "regex": r"(how\s+are\s+you|how\s+are\s+u)",
         "responses": ["I'm a program, running smoothly — thanks for asking!"], "priority": 5},
        {"id": "help", "examples": ["help", "what can you do"], "regex": "", "responses": [
            "I can respond to greetings, answer simple questions, and exit when you say 'bye'. Use /profile-show to view profile."],
         "priority": 5},
        {"id": "tell_me_about_user", "examples": ["tell me about me", "who am i", "my details"], "regex": "",
         "responses": ["__USER_PROFILE__"], "priority": 4}
    ],
    "fallback": {
        "responses": ["Sorry, I didn't understand that. Can you rephrase?", "I don't have an answer for that yet."]}
}

# -------------------------
# Profile management
# -------------------------
DEFAULT_PROFILE = {
    "user_name": "Ahmed",
    "spouse_name": "Mariyum",
    "notes": "Wife is loving and a bit stubborn; user loves her back.",
    "display_name_response": "MARIYUM KA AHMED"
}


def load_profile(path: Path = PROFILE_FILE) -> Dict[str, str]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                for k, v in DEFAULT_PROFILE.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            return DEFAULT_PROFILE.copy()
    return DEFAULT_PROFILE.copy()


def save_profile(profile: Dict[str, str], path: Path = PROFILE_FILE) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)


# -------------------------
# Utilities
# -------------------------
def normalize(text: Optional[str]) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s'?!]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -------------------------
# Session container
# -------------------------
class Session:
    def __init__(self, session_id: str, max_history: int = 20):
        self.session_id = session_id
        self.history: List[Tuple[str, str]] = []
        self.max_history = max_history

    def push(self, user: str, bot: str) -> None:
        self.history.append((user, bot))
        if len(self.history) > self.max_history:
            self.history.pop(0)


# -------------------------
# Rule Engine
# -------------------------
class RuleEngine:
    def __init__(self, intents: Dict[str, Any], profile: Dict[str, str]):
        self.intents = intents.get("intents", [])
        self.fallbacks = intents.get("fallback", {}).get("responses", ["I do not understand."])
        self.profile = profile
        self.phrase_map: Dict[str, Dict[str, Any]] = {}
        for intent in self.intents:
            for ex in intent.get("examples", []):
                self.phrase_map[normalize(ex)] = intent

    def _regex_match(self, norm: str, pattern: str):
        if not pattern:
            return None
        try:
            return re.search(pattern, norm)
        except re.error:
            return None

    def _render_response(self, template: str) -> str:
        if template == "__PERSONAL_NAME__":
            return self.profile.get("display_name_response", DEFAULT_PROFILE["display_name_response"])
        if template == "__USER_PROFILE__":
            return f"My user is {self.profile.get('user_name')}. Their spouse is {self.profile.get('spouse_name')}. Notes: {self.profile.get('notes')}"
        return template

    def match(self, user_input: str) -> Tuple[str, str, float, Dict[str, str]]:
        norm = normalize(user_input)
        matched_intent = None
        slots = {}
        confidence = 0.0

        # 1) Exact phrase matching
        if norm in self.phrase_map:
            matched_intent = self.phrase_map[norm]
            confidence = 1.0

        # 2) Priority-based regex and contains checks
        if not matched_intent:
            for intent in sorted(self.intents, key=lambda i: -i.get("priority", 0)):
                pattern = intent.get("regex", "")
                m = self._regex_match(norm, pattern)
                if m:
                    slots = m.groupdict() if m.groupdict() else {}
                    matched_intent = intent
                    confidence = 0.85
                    break

                # Sample text inclusion check
                found = False
                for ex in intent.get("examples", []):
                    ex_norm = normalize(ex)
                    if " " in ex_norm:
                        if ex_norm in norm:
                            found = True
                            break
                    else:
                        if ex_norm in set(norm.split()):
                            found = True
                            break
                if found:
                    matched_intent = intent
                    confidence = 0.85
                    break

        # Process response if an intent was found
        if matched_intent:
            intent_id = matched_intent.get("id")
            resp_template = random.choice(matched_intent.get("responses", [""]))
            base_response = self._render_response(resp_template)

            # CRITICAL TRIGGER OVERRIDE: Inject identity on greetings or name checks
            if intent_id in ["greeting", "identity", "tell_me_about_user"]:
                final_response = f"{IDENTITY_STORY}\n\nHow can I help you today?"
                return final_response, intent_id, confidence, slots

            return base_response, intent_id, confidence, slots

        # 3) Fallback
        return random.choice(self.fallbacks), "fallback", 0.35, {}


# -------------------------
# Logging & CLI
# -------------------------
def setup_logging(debug: bool = False):
    root = logging.getLogger()
    if root.handlers:
        return
    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def handle_admin_command(cmd: str, profile: Dict[str, str]) -> Optional[str]:
    parts = cmd.strip().split(maxsplit=1)
    action = parts[0].lower()
    try:
        if action == "/profile-show":
            return json.dumps(profile, indent=2, ensure_ascii=False)
        if action == "/profile-set" and len(parts) == 2:
            kv = parts[1]
            if "=" not in kv:
                return "Usage: /profile-set key=value"
            key, value = kv.split("=", 1)
            key, value = key.strip(), value.strip()
            profile[key] = value
            save_profile(profile)
            return f"Profile updated: {key} = {value}"
        if action == "/profile-export" and len(parts) == 2:
            fname = parts[1].strip()
            p = Path(fname)
            save_profile(profile, p)
            return f"Profile exported to {p.resolve()}"
        if action == "/profile-import" and len(parts) == 2:
            fname = parts[1].strip()
            p = Path(fname)
            if not p.exists():
                return f"File not found: {p}"
            new_profile = load_profile(p)
            profile.clear()
            profile.update(new_profile)
            save_profile(profile)
            return f"Profile imported from {p.resolve()}"
    except Exception as e:
        return f"Admin command failed: {e}"
    return None


def run_cli(enable_file_log: bool = False, debug: bool = False):
    setup_logging(debug)
    profile = load_profile()
    engine = RuleEngine(INTENTS, profile)
    session = Session(session_id="local")
    print("DecodeBot Personal — type 'bye' or 'exit' to quit. Admin commands start with '/'.")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! (session ended)")
            break

        if not user:
            print("Bot: Please type something so I can respond.")
            continue

        if user.startswith("/"):
            admin_resp = handle_admin_command(user, profile)
            print(f"Bot: {admin_resp or 'Unknown admin command.'}")
            logging.info("admin_command user=%s result=%s", user, admin_resp)
            continue

        reply, intent_id, confidence, slots = engine.match(user)
        print(f"Bot: {reply}")
        logging.info("session=%s intent=%s conf=%.2f slots=%s user=%s", session.session_id, intent_id, confidence,
                     slots, user)
        session.push(user, reply)

        if intent_id == "goodbye" or re.search(r"\b(exit|quit|bye|goodbye|stop)\b", normalize(user)):
            break


# -------------------------
# Optional web UI
# -------------------------
def run_web(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    try:
        from flask import Flask, request, jsonify, render_template_string
    except Exception:
        print("Flask not installed. Install with: pip install Flask")
        return

    setup_logging(debug)
    profile = load_profile()
    engine = RuleEngine(INTENTS, profile)
    app = Flask(__name__)

    HTML = """
    <!doctype html><title>DecodeBot Personal</title>
    <style>body{font-family:Arial;margin:1.5rem}#chat{border:1px solid #ddd;padding:1rem;height:60vh;overflow:auto;background:#fafafa}.user{color:#0b5}.bot{color:#05b}</style>
    <div id="chat"></div><input id="msg" autofocus style="width:70%"/><button onclick="send()">Send</button>
    <script>
    function append(role,text){const d=document.createElement('div');d.className=role;d.textContent=(role==='user'?'You: ':'Bot: ')+text;document.getElementById('chat').appendChild(d);window.scrollTo(0,document.body.scrollHeight);}
    async function send(){const m=document.getElementById('msg'); if(!m.value) return; append('user',m.value); const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({msg:m.value})}); const j=await res.json(); append('bot',j.reply); m.value='';}
    </script>
    """

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        msg = data.get("msg", "")
        reply, intent, conf, slots = engine.match(msg)
        return jsonify({"reply": reply, "intent": intent, "confidence": conf, "slots": slots})

    print(f"Starting web UI on http://{host}:{port}")
    app.run(host=host, port=port)


# -------------------------
# Self tests
# -------------------------
def self_test():
    profile = load_profile()
    engine = RuleEngine(INTENTS, profile)
    tests = [
        ("Hi", "greeting"),
        ("bye", "goodbye"),
        ("What's your name?", "identity"),
        ("How are you?", "status"),
        ("Tell me about me", "tell_me_about_user"),
        ("qwertyuiop", "fallback")
    ]
    passed = 0
    for inp, expected in tests:
        resp, intent, conf, slots = engine.match(inp)
        ok = (intent == expected)
        print(f"Input: {inp!r} | Expected: {expected} | Got: {intent} | {'OK' if ok else 'FAIL'} | Reply: {resp}")
        if ok:
            passed += 1
    print(f"\nPassed {passed}/{len(tests)} tests.")
    return passed == len(tests)


# -------------------------
# Entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="DecodeBot Personal — single-file chatbot")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument("--web", action="store_true", help="Run web UI (requires Flask)")
    parser.add_argument("--selftest", action="store_true", help="Run built-in self tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.selftest:
        ok = self_test()
        sys.exit(0 if ok else 2)

    if args.web:
        run_web(debug=args.debug)
    else:
        run_cli(enable_file_log=args.log, debug=args.debug)


if __name__ == "__main__":
    main()