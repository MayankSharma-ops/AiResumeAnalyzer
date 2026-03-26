# 🤖 ATS Resume Optimizer — Telegram Bot

A Telegram bot that analyzes CVs against Job Descriptions, gives ATS scores, identifies keyword gaps, optimizes resumes with AI, and generates professional DOCX files in multiple templates.

**Powered by Google Gemini API.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 ATS Scoring | Score your CV 0–10 against any JD |
| 🔑 Keyword Analysis | Find matched & missing keywords |
| 💡 Smart Suggestions | Actionable improvement tips |
| 📝 3 Optimized Versions | ATS-Heavy, Balanced, Concise |
| 🎨 4 Templates | 1-Page ATS, 2-Page, Sidebar, Classic |
| 📈 Before/After Score | See your improvement |
| 📥 DOCX Download | One-click download |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- Gemini API Key (from [Google AI Studio](https://aistudio.google.com/apikey))

### 2. Install

```bash
cd "ai agent"
pip install -r requirements.txt
```

### 3. Configure

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_token_here
GEMINI_API_KEY=your_key_here
```

### 4. Run

```bash
python main.py
```

### 5. Use

Open Telegram → Find your bot → Send `/start` → Follow the flow!

---

## 📁 Project Structure

```
ai agent/
├── main.py            # Entry point
├── bot.py             # Telegram conversation handler
├── ai_engine.py       # Gemini API integration
├── cv_parser.py       # PDF/DOCX text extraction
├── cv_generator.py    # DOCX template generator
├── config.py          # Configuration & constants
├── requirements.txt   # Dependencies
├── .env.example       # Environment variable template
└── .gitignore
```

---

## 🧠 How It Works

```
/start
  ↓
Upload CV (PDF/DOCX)
  ↓
Paste Job Description
  ↓
⏳ AI Analysis (Gemini)
  ↓
📊 ATS Score + Keywords + Suggestions
  ↓
Choose Version (ATS / Balanced / Concise)
  ↓
Choose Template (1-Page / 2-Page / Sidebar / Classic)
  ↓
📥 Download Optimized CV + New Score
```

---

## 📄 License

Built for hackathon. Use freely.
