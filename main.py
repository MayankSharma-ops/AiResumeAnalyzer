"""
Main entry point - starts the ATS Telegram Bot.
"""

import os
import sys
from pathlib import Path

from config import GEMINI_API_KEY, TELEGRAM_BOT_TOKEN
from bot import create_bot

BUILD_ID = "2026-03-26-1306"


def _module_stamp(path_str: str) -> str:
    path = Path(path_str).resolve()
    return f"{path} | modified {path.stat().st_mtime:.0f}"


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        print("ERROR: TELEGRAM_BOT_TOKEN is not set!")
        print("Create a .env file with your token from @BotFather")
        sys.exit(1)

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("ERROR: GEMINI_API_KEY is not set!")
        print("Create a .env file with your key from Google AI Studio")
        sys.exit(1)

    print("ATS Resume Optimizer Bot starting...")
    print(f"Build: {BUILD_ID}")
    print(f"PID: {os.getpid()}")
    print(f"CWD: {Path.cwd().resolve()}")
    print(f"main.py: {_module_stamp(__file__)}")
    print("Press Ctrl+C to stop.\n")

    app = create_bot()
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
