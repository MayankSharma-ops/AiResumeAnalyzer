"""
Configuration module — loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── ATS Scoring ──────────────────────────────────────────────────────────────
ATS_PASS_THRESHOLD = 6
ATS_MAX_SCORE = 10

# ── Temp files directory ─────────────────────────────────────────────────────
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Gemini Model ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"