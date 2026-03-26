"""
Main entry point — starts the ATS Telegram Bot.
"""

import sys
from config import TELEGRAM_BOT_TOKEN, GEMINI_API_KEY
from bot import create_bot


def main():
    # Validate configuration
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        print("❌ ERROR: TELEGRAM_BOT_TOKEN is not set!")
        print("   Create a .env file with your token from @BotFather")
        sys.exit(1)

    if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyCsrg7ingmfRF7cyhqCyvCxWzgREwvKxlo":
        print("❌ ERROR: GEMINI_API_KEY is not set!")
        print("   Create a .env file with your key from Google AI Studio")
        sys.exit(1)

    print("🤖 ATS Resume Optimizer Bot starting...")
    print("   Press Ctrl+C to stop.\n")

    app = create_bot()
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
