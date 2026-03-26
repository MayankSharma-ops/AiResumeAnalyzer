"""
Configuration module — loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8760278781:AAFhvkk_-vjMk9GbE4EBbMAbHp07eOk6vwQ")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCsrg7ingmfRF7cyhqCyvCxWzgREwvKxlo")

# ── ATS Scoring ──────────────────────────────────────────────────────────────
ATS_PASS_THRESHOLD = 6  # Score >= 6 is considered "pass"
ATS_MAX_SCORE = 10

# ── Temp files directory ─────────────────────────────────────────────────────
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Gemini Model ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-flash-latest"
