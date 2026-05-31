🤖 DecodeBot Personal — Rule-Based AI Chatbot
DecodeLabs AI Engineering Internship — Week 1
Project: AI-Powered Conversational Bot (Rule-Based Edition)
Type: Single-File Python Chatbot with Personalization
Architecture: Intent Matching + Session Memory + Profile Management
📋 Table of Contents
Project Overview
Key Features
Technical Stack
Project Structure
Installation & Setup
How to Run
Commands Guide
Code Breakdown
Screenshots
Learning Outcomes
Acknowledgements
🎯 Project Overview
DecodeBot Personal is a fully functional, rule-based AI chatbot built entirely in pure Python — without any external AI APIs or ML libraries. It uses a powerful intent-matching engine (exact match, contains, regex) to understand user input and respond intelligently.
The bot features:
🧠 Multi-layer intent detection (exact → regex → contains)
👤 Personal profile system (editable at runtime)
📝 Session context memory (20-turn history)
🛡️ Admin commands for profile management
🌐 Optional web UI (Flask-based)
🧪 Built-in self tests
✨ Key Features
Table
Feature	Description
✅ Pure Python	Zero external AI dependencies — fully offline
✅ 3-Tier Intent Matching	Exact phrase → Regex pattern → Keyword contains
✅ Confidence Scoring	Every response has a confidence score (1.0 → 0.35)
✅ Personal Identity	Customizable name, spouse, and personality story
✅ Session Memory	Maintains last 20 conversation turns
✅ Admin Commands	/profile-show, /profile-set, /profile-export, /profile-import
✅ Rotating Logs	Automatic log rotation (2MB max, 3 backups)
✅ Web UI Mode	Optional Flask-based chat interface
✅ Self Tests	Built-in --selftest flag for validation
✅ Graceful Exit	Handles Ctrl+C and EOF cleanly
🛠️ Technical Stack
plain
Python 3.x (Standard Library Only)
├── re          → Regex intent matching
├── json        → Profile import/export
├── logging     → Rotating file handlers
├── argparse    → CLI argument parsing
└── pathlib     → Cross-platform file paths

Optional:
└── Flask       → Web UI mode (pip install Flask)
📁 Project Structure
plain
DecodeLabs-Week1-RuleBot/
│
├── 📄 README.md                    ← You are here
├── 📄 decodebot_personal.py        ← Main Python script (single file!)
├── 📄 user_profile.json            ← Auto-generated profile file
│
└── 📂 logs/                        ← Auto-created on first run
    └── 📄 decodebot_personal.log   ← Conversation & error logs
⚙️ Installation & Setup
Step 1: Clone or Download
bash
git clone https://github.com/yourusername/DecodeLabs-Week1-RuleBot.git
cd DecodeLabs-Week1-RuleBot
Step 2: Run (No dependencies needed!)
bash
python decodebot_personal.py
Optional: Web UI Mode
bash
pip install Flask
python decodebot_personal.py --web
Optional: Run Self Tests
bash
python decodebot_personal.py --selftest
🚀 How to Run
Interactive CLI Mode
bash
python decodebot_personal.py
Sample Conversation
plain
DecodeBot Personal — type 'bye' or 'exit' to quit. Admin commands start with '/'.

You: Hi
Bot: My name is Ahmed, but my absolute identity is 'MARIYUM KA AHMED'. 
     If you want to know my details: My wifey's name is Mariyum. 
     She is incredibly stubborn (bhot ziddi hai) but she is extremely beautiful and cute (bhoot pyari hai). 
     She loves me deeply, and I love her immensely in return.

     How can I help you today?

You: What's your name?
Bot: My name is Ahmed, but my absolute identity is 'MARIYUM KA AHMED'...

You: /profile-show
Bot: {
       "user_name": "Ahmed",
       "spouse_name": "Mariyum",
       "notes": "Wife is loving and a bit stubborn; user loves her back.",
       "display_name_response": "MARIYUM KA AHMED"
     }

You: /profile-set user_name=Ali
Bot: Profile updated: user_name = Ali

You: bye
Bot: Goodbye! Have a great day.
⌨️ Commands Guide
Admin Commands (start with /)
Table
Command	Description	Example
/profile-show	Display current profile JSON	/profile-show
/profile-set key=value	Update a profile field	/profile-set user_name=Ali
/profile-export file.json	Save profile to file	/profile-export my_profile.json
/profile-import file.json	Load profile from file	/profile-import my_profile.json
CLI Arguments
Table
Flag	Description
--log	Enable file logging to /logs/
--web	Run Flask web UI on 127.0.0.1:5000
--selftest	Run built-in intent matching tests
--debug	Enable debug-level logging
🔬 Code Breakdown
1. Intent Matching Engine (3-Tier)
Python
def match(self, user_input: str):
    # Tier 1: Exact phrase match (confidence = 1.0)
    if norm in self.phrase_map:
        return matched_intent, 1.0

    # Tier 2: Regex pattern match (confidence = 0.85)
    for intent in sorted_by_priority:
        if regex_matches(norm, intent.pattern):
            return matched_intent, 0.85

    # Tier 3: Keyword contains match (confidence = 0.85)
    for intent in sorted_by_priority:
        if keywords_in(norm, intent.examples):
            return matched_intent, 0.85

    # Fallback (confidence = 0.35)
    return fallback_response, 0.35
2. Profile System
Python
DEFAULT_PROFILE = {
    "user_name": "Ahmed",
    "spouse_name": "Mariyum",
    "notes": "Wife is loving and a bit stubborn; user loves her back.",
    "display_name_response": "MARIYUM KA AHMED"
}
Auto-saves to user_profile.json
Editable at runtime via /profile-set
Exportable/importable via JSON files
3. Session Memory
Python
class Session:
    def __init__(self, session_id: str, max_history: int = 20):
        self.history: List[Tuple[str, str]] = []
Stores last 20 user-bot exchanges
FIFO eviction when limit reached
4. Web UI (Optional Flask)
Python
@app.route("/api/chat", methods=["POST"])
def chat():
    reply, intent, conf, slots = engine.match(msg)
    return jsonify({"reply": reply, "intent": intent, "confidence": conf})
Simple HTML/JS frontend
REST API endpoint for chat messages
📸 Screenshots
Place your execution screenshots in the /assets/ folder:
Table
Screenshot	Description
Terminal conversation with identity injection
Profile show/set/export commands
Flask-based browser chat interface
Rotating log file output
🎓 Learning Outcomes
Through this project, I have demonstrated:
✅ Rule-Based NLP — Intent matching without neural networks or APIs
✅ Regex Engineering — Pattern-based text understanding
✅ Session State Management — In-memory conversation history
✅ Profile Persistence — JSON-based user data storage
✅ CLI UX Design — Interactive command-line interfaces
✅ Admin System Design — Runtime configuration commands
✅ Logging Best Practices — Rotating file handlers with UTF-8
✅ Web API Basics — Flask REST endpoint for chatbot
✅ Testing Culture — Built-in --selftest validation suite
✅ Graceful Degradation — Fallback responses for unknown inputs
🙏 Acknowledgements
DecodeLabs — For the structured AI Engineering Internship opportunity
Python Software Foundation — For the powerful standard library
📬 Contact
Intern Name: [Your Name]
Email: [your.email@example.com]
LinkedIn: [Your LinkedIn Profile]
GitHub: [Your GitHub Profile]
<div align="center">
⭐ DecodeLabs AI Engineering Internship — Week 1 ⭐
Project 1: AI-Powered Conversational Bot (Rule-Based Edition)
Submitted as part of the official internship program.
</div>
