"""
Telegram Bot — conversational ATS analyzer with inline keyboards.

Flow:
  /start → Upload CV → Paste JD (or upload JD as PDF) → Analysis (score + keywords + suggestions)
  → Choose optimized version → Choose template → Download DOCX + new score
"""

import asyncio
import logging
import math
import os
from typing import Any
from telegram import (
    CallbackQuery,
    Message,
    Chat,
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)
from telegram.error import NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from config import TELEGRAM_BOT_TOKEN, ATS_PASS_THRESHOLD, TEMP_DIR
from cv_parser import parse_cv
from ai_engine import analyze_cv, rescore_cv
from cv_generator import generate_cv, TEMPLATES

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Conversation States ──────────────────────────────────────────────────────
UPLOAD_CV, PASTE_JD, CHOOSE_VERSION, CHOOSE_TEMPLATE = range(4)


# ── Helper Functions ─────────────────────────────────────────────────────────

def _score_emoji(score: float) -> str:
    if score >= 8:
        return "🟢"
    elif score >= 6:
        return "🟡"
    else:
        return "🔴"


def _build_analysis_message(result: dict) -> str:
    """Format the Gemini analysis result into a nice Telegram message."""
    score = result["ats_score"]
    emoji = _score_emoji(score)
    status = "✅ PASS" if score >= ATS_PASS_THRESHOLD else "❌ NEEDS IMPROVEMENT"

    msg = f"""
{emoji} <b>ATS Score: {score}/10</b> — {status}

━━━━━━━━━━━━━━━━━━━━

✅ <b>Matched Keywords:</b>
{', '.join(result['matched_keywords']) if result['matched_keywords'] else 'None found'}

❌ <b>Missing Keywords:</b>
{', '.join(result['missing_keywords']) if result['missing_keywords'] else 'All keywords matched!'}

━━━━━━━━━━━━━━━━━━━━

💡 <b>Suggestions:</b>
"""
    for i, suggestion in enumerate(result.get("suggestions", []), 1):
        msg += f"\n{i}. {suggestion}"

    if score < ATS_PASS_THRESHOLD:
        msg += "\n\n⬇️ <b>Your score is below the threshold. Let me optimize your CV!</b>"
        msg += "\n\n<b>Choose an optimized version:</b>"
    else:
        msg += "\n\n🎉 <b>Great score! You can still optimize further if you'd like.</b>"
        msg += "\n\n<b>Choose an optimized version:</b>"

    return msg


def _safe_score(score: Any) -> float | None:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return round(min(10.0, max(0.0, value)), 1)


def _score_label(score: Any) -> str:
    safe_score = _safe_score(score)
    return f"{safe_score:.1f}/10" if safe_score is not None else "N/A"


def _score_emoji(score: Any) -> str:
    safe_score = _safe_score(score)
    if safe_score is None:
        return "⚪"
    if safe_score >= 8:
        return "ðŸŸ¢"
    if safe_score >= 6:
        return "ðŸŸ¡"
    return "ðŸ”´"


def _build_analysis_message(result: dict) -> str:
    """Format the analysis result into a safe Telegram message."""
    score = _safe_score(result.get("ats_score"))
    effective_score = score if score is not None else 0.0
    emoji = _score_emoji(effective_score)
    status = "âœ… PASS" if effective_score >= ATS_PASS_THRESHOLD else "âŒ NEEDS IMPROVEMENT"

    msg = f"""
{emoji} <b>ATS Score: {_score_label(score)}</b> â€” {status}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

âœ… <b>Matched Keywords:</b>
{', '.join(result['matched_keywords']) if result['matched_keywords'] else 'None found'}

âŒ <b>Missing Keywords:</b>
{', '.join(result['missing_keywords']) if result['missing_keywords'] else 'All keywords matched!'}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

ðŸ’¡ <b>Suggestions:</b>
"""
    for i, suggestion in enumerate(result.get("suggestions", []), 1):
        msg += f"\n{i}. {suggestion}"

    if effective_score < ATS_PASS_THRESHOLD:
        msg += "\n\nâ¬‡ï¸ <b>Your score is below the threshold. Let me optimize your CV!</b>"
        msg += "\n\n<b>Choose an optimized version:</b>"
    else:
        msg += "\n\nðŸŽ‰ <b>Great score! You can still optimize further if you'd like.</b>"
        msg += "\n\n<b>Choose an optimized version:</b>"

    return msg


def _score_emoji(score: Any) -> str:
    safe_score = _safe_score(score)
    if safe_score is None:
        return "\u26AA"
    if safe_score >= 8:
        return "\U0001F7E2"
    if safe_score >= 6:
        return "\U0001F7E1"
    return "\U0001F534"


def _build_analysis_message(result: dict) -> str:
    """Format the analysis result into a safe Telegram message."""
    score = _safe_score(result.get("ats_score"))
    effective_score = score if score is not None else 0.0
    emoji = _score_emoji(effective_score)
    status = "PASS" if effective_score >= ATS_PASS_THRESHOLD else "NEEDS IMPROVEMENT"

    msg = f"""
{emoji} <b>ATS Score: {_score_label(score)}</b> - {status}

<b>Matched Keywords:</b>
{', '.join(result['matched_keywords']) if result['matched_keywords'] else 'None found'}

<b>Missing Keywords:</b>
{', '.join(result['missing_keywords']) if result['missing_keywords'] else 'All keywords matched!'}

<b>Suggestions:</b>
"""
    for i, suggestion in enumerate(result.get("suggestions", []), 1):
        msg += f"\n{i}. {suggestion}"

    if effective_score < ATS_PASS_THRESHOLD:
        msg += "\n\n<b>Your score is below the threshold. Let me optimize your CV!</b>"
        msg += "\n\n<b>Choose an optimized version:</b>"
    else:
        msg += "\n\n<b>Great score! You can still optimize further if you'd like.</b>"
        msg += "\n\n<b>Choose an optimized version:</b>"

    return msg


def _score_emoji(score: Any) -> str:
    safe_score = _safe_score(score)
    if safe_score is None:
        return "[N/A]"
    if safe_score >= 8:
        return "[HIGH]"
    if safe_score >= 6:
        return "[MID]"
    return "[LOW]"


def _require_message(update: Update) -> Message:
    """Return the message for handlers that only accept message updates."""
    message = update.effective_message
    if message is None:
        raise ValueError("This handler requires a message update.")
    return message


def _require_callback_query(update: Update) -> CallbackQuery:
    """Return the callback query for handlers that only accept callback updates."""
    query = update.callback_query
    if query is None:
        raise ValueError("This handler requires a callback query update.")
    return query


def _require_user(update: Update) -> User:
    """Return the effective user for handlers that require a user context."""
    user = update.effective_user
    if user is None:
        raise ValueError("This handler requires a user context.")
    return user


def _require_chat(update: Update) -> Chat:
    """Return the effective chat for handlers that require a chat context."""
    chat = update.effective_chat
    if chat is None:
        raise ValueError("This handler requires a chat context.")
    return chat


def _user_data(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    """Return user_data with a concrete mapping type for static analysis."""
    data = context.user_data
    if data is None:
        raise ValueError("user_data is unavailable for this context.")
    return data


def _cleanup_user_files(user_data: dict) -> None:
    """Remove any temp files stored for this user session."""
    for key in ("cv_file_path", "jd_file_path"):
        path = user_data.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


# ── Bot Handlers ─────────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /start — welcome message."""
    user_data = _user_data(context)
    _cleanup_user_files(user_data)
    user_data.clear()
    message = _require_message(update)

    welcome = """
🤖 <b>ATS Resume Optimizer Bot</b>

I'll help you beat ATS systems and land interviews!

<b>Here's how it works:</b>
1️⃣ Upload your CV (PDF or DOCX)
2️⃣ Paste the Job Description <i>or upload it as a PDF</i>
3️⃣ Get your ATS score & analysis
4️⃣ Download optimized CV in multiple templates

<b>Commands:</b>
/start — Begin a new session
/end — End current session
/help — Show help

📎 <b>Start by uploading your CV file:</b>
"""
    await message.reply_text(welcome, parse_mode="HTML")
    return UPLOAD_CV


async def handle_cv_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle CV file upload — download and extract text."""
    user_data = _user_data(context)
    message = _require_message(update)
    document = message.document

    if not document:
        await message.reply_text(
            "⚠️ Please upload a file. I accept <b>PDF</b> and <b>DOCX</b> formats.",
            parse_mode="HTML",
        )
        return UPLOAD_CV

    file_name = document.file_name or "resume"
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in (".pdf", ".docx"):
        await message.reply_text(
            "⚠️ Unsupported format. Please upload a <b>PDF</b> or <b>DOCX</b> file.",
            parse_mode="HTML",
        )
        return UPLOAD_CV

    # Download file
    await message.reply_text("📥 Downloading your CV...")
    file = await document.get_file()
    user = _require_user(update)
    file_path = os.path.join(TEMP_DIR, f"cv_{user.id}{ext}")
    await file.download_to_drive(file_path)

    # Extract text
    try:
        cv_text = parse_cv(file_path)
        if not cv_text.strip():
            await message.reply_text(
                "⚠️ Could not extract text from your CV. "
                "The file might be image-based or corrupt. "
                "Please try a different file.",
            )
            return UPLOAD_CV

        user_data["cv_text"] = cv_text
        user_data["cv_file_path"] = file_path

        await message.reply_text(
            f"✅ CV uploaded successfully! "
            f"({len(cv_text.split())} words extracted)\n\n"
            f"📋 <b>Now provide the Job Description:</b>\n\n"
            f"You can either:\n"
            f"• 📝 <b>Paste the JD text</b> directly\n"
            f"• 📄 <b>Upload the JD as a PDF file</b>",
            parse_mode="HTML",
        )
        return PASTE_JD

    except Exception as e:
        logger.error(f"CV parsing error: {e}")
        await message.reply_text(
            f"❌ Error parsing CV: {str(e)}\nPlease try another file.",
        )
        return UPLOAD_CV


async def handle_jd_pdf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle JD uploaded as a PDF file — extract text and proceed to analysis."""
    user_data = _user_data(context)
    message = _require_message(update)
    document = message.document

    if not document:
        await message.reply_text(
            "⚠️ No file received. Please upload a <b>PDF</b> file or paste the JD as text.",
            parse_mode="HTML",
        )
        return PASTE_JD

    file_name = document.file_name or "jd"
    ext = os.path.splitext(file_name)[1].lower()

    if ext != ".pdf":
        await message.reply_text(
            "⚠️ For the Job Description, only <b>PDF</b> format is supported.\n"
            "You can also just <b>paste the JD text</b> directly.",
            parse_mode="HTML",
        )
        return PASTE_JD

    await message.reply_text("📥 Downloading JD PDF...")

    file = await document.get_file()
    user = _require_user(update)
    file_path = os.path.join(TEMP_DIR, f"jd_{user.id}.pdf")
    await file.download_to_drive(file_path)
    user_data["jd_file_path"] = file_path

    try:
        jd_text = parse_cv(file_path)
        if not jd_text.strip():
            await message.reply_text(
                "⚠️ Could not extract text from the JD PDF. "
                "It might be image-based or scanned. "
                "Please paste the JD text directly instead.",
            )
            return PASTE_JD

        word_count = len(jd_text.split())
        if word_count < 20:
            await message.reply_text(
                f"⚠️ Extracted text seems too short ({word_count} words). "
                "Please paste the JD text directly.",
            )
            return PASTE_JD

        logger.info(f"JD PDF extracted: {word_count} words")

        # Reuse the same JD analysis flow
        # Inject jd_text into user_data and trigger analysis
        user_data["jd_text"] = jd_text

    except Exception as e:
        logger.error(f"JD PDF parsing error: {e}")
        await message.reply_text(
            f"❌ Error reading JD PDF: {str(e)}\n\nPlease paste the JD text directly.",
        )
        return PASTE_JD

    # --- Proceed to analysis (same as handle_jd_input) ---
    processing_msg = await message.reply_text(
        f"✅ JD PDF extracted! ({word_count} words)\n\n"
        "⏳ <b>Analyzing your CV against the Job Description...</b>\n\n"
        "🔍 Extracting keywords...\n"
        "📊 Calculating ATS score...\n"
        "💡 Generating suggestions...\n"
        "📝 Creating optimized versions...\n\n"
        "This may take 15-30 seconds...",
        parse_mode="HTML",
    )

    try:
        cv_text = user_data["cv_text"]
        result = analyze_cv(cv_text, jd_text)
        user_data["analysis"] = result

        analysis_msg = _build_analysis_message(result)

        keyboard = [
            [InlineKeyboardButton("🎯 ATS-Optimized (Keyword Heavy)", callback_data="ver_ats_heavy")],
            [InlineKeyboardButton("⚖️ Balanced Professional", callback_data="ver_balanced")],
            [InlineKeyboardButton("📄 Concise 1-Page", callback_data="ver_concise")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text(
            analysis_msg,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return CHOOSE_VERSION

    except Exception as e:
        logger.exception("Gemini analysis error (JD PDF)")
        await processing_msg.edit_text(
            f"❌ <b>Analysis failed:</b> {str(e)}\n\n"
            f"Please try again with /start.",
            parse_mode="HTML",
        )
        return PASTE_JD


async def handle_jd_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle JD text input — analyze CV against JD."""
    user_data = _user_data(context)
    message = _require_message(update)
    jd_text = message.text

    if not jd_text or len(jd_text.strip()) < 20:
        await message.reply_text(
            "⚠️ Job Description seems too short. "
            "Please paste the complete JD text, or upload it as a PDF.",
        )
        return PASTE_JD

    user_data["jd_text"] = jd_text

    processing_msg = await message.reply_text(
        "⏳ <b>Analyzing your CV against the Job Description...</b>\n\n"
        "🔍 Extracting keywords...\n"
        "📊 Calculating ATS score...\n"
        "💡 Generating suggestions...\n"
        "📝 Creating optimized versions...\n\n"
        "This may take 15-30 seconds...",
        parse_mode="HTML",
    )

    try:
        cv_text = user_data["cv_text"]
        result = analyze_cv(cv_text, jd_text)
        user_data["analysis"] = result

        analysis_msg = _build_analysis_message(result)

        keyboard = [
            [InlineKeyboardButton("🎯 ATS-Optimized (Keyword Heavy)", callback_data="ver_ats_heavy")],
            [InlineKeyboardButton("⚖️ Balanced Professional", callback_data="ver_balanced")],
            [InlineKeyboardButton("📄 Concise 1-Page", callback_data="ver_concise")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text(
            analysis_msg,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return CHOOSE_VERSION

    except Exception as e:
        logger.exception("Gemini analysis error")
        await processing_msg.edit_text(
            f"❌ <b>Analysis failed:</b> {str(e)}\n\n"
            f"Please try again with /start or paste the JD again.",
            parse_mode="HTML",
        )
        return PASTE_JD


async def handle_version_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle optimized version selection."""
    user_data = _user_data(context)
    query = _require_callback_query(update)
    await query.answer()
    query_data = query.data
    message = query.message

    if message is None:
        raise ValueError("Template selection requires a callback message.")
    if query_data is None:
        raise ValueError("Version selection requires callback data.")

    version_map = {
        "ver_ats_heavy": ("ats_heavy", "🎯 ATS-Optimized"),
        "ver_balanced": ("balanced", "⚖️ Balanced"),
        "ver_concise": ("concise", "📄 Concise"),
    }

    version_key, version_label = version_map.get(query_data, ("balanced", "⚖️ Balanced"))
    user_data["chosen_version"] = version_key
    user_data["chosen_version_label"] = version_label

    keyboard = []
    for key, (label, _) in TEMPLATES.items():
        keyboard.append([InlineKeyboardButton(label, callback_data=f"tpl_{key}")])

    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        f"✅ Selected: <b>{version_label}</b>\n\n"
        f"🎨 <b>Choose a template format:</b>",
        parse_mode="HTML",
        reply_markup=reply_markup,
    )
    return CHOOSE_TEMPLATE


async def handle_template_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle template selection — generate CV and send file."""
    user_data = _user_data(context)
    query = _require_callback_query(update)
    await query.answer()
    query_data = query.data
    message = query.message

    if message is None:
        raise ValueError("Template selection requires a callback message.")
    if query_data is None:
        raise ValueError("Template selection requires callback data.")

    template_key = query_data.replace("tpl_", "")

    if template_key not in TEMPLATES:
        await query.edit_message_text("❌ Invalid template. Please try again with /start")
        return ConversationHandler.END

    template_label = TEMPLATES[template_key][0]

    await query.edit_message_text(
        f"⏳ <b>Generating your optimized CV...</b>\n\n"
        f"Version: {user_data.get('chosen_version_label', '')}\n"
        f"Template: {template_label}\n\n"
        f"Please wait...",
        parse_mode="HTML",
    )

    try:
        analysis = user_data["analysis"]
        version_key = user_data["chosen_version"]
        optimized_text = analysis["optimized_resumes"][version_key]
        version_report = analysis.get("version_reports", {}).get(version_key, {})

        file_path = generate_cv(
            optimized_text,
            template_key,
            version_name=version_key,
        )

        version_score = _safe_score(version_report.get("new_score"))
        if version_score is not None:
            new_score = version_score
            improvement = version_report.get("improvement_summary", "")
        else:
            jd_text = user_data["jd_text"]
            rescore_result = rescore_cv(optimized_text, jd_text)
            new_score = _safe_score(rescore_result.get("new_score"))
            improvement = rescore_result.get("improvement_summary", "")

        old_score = _safe_score(analysis.get("ats_score")) or 0.0
        old_emoji = _score_emoji(old_score)
        new_emoji = _score_emoji(new_score) if isinstance(new_score, (int, float)) else "⚪"

        new_emoji = _score_emoji(new_score)
        chat = _require_chat(update)

        with open(file_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat.id,
                document=f,
                filename=os.path.basename(file_path),
                caption=(
                    f"📄 <b>Your Optimized CV</b>\n\n"
                    f"📊 <b>Score Comparison:</b>\n"
                    f"   Before: {old_emoji} {_score_label(old_score)}\n"
                    f"   After:  {new_emoji} {_score_label(new_score)}\n\n"
                    f"💬 {improvement}"
                ),
                parse_mode="HTML",
            )

        keyboard = [
            [InlineKeyboardButton("🔄 Try Different Template", callback_data="action_template")],
            [InlineKeyboardButton("📝 Try Different Version", callback_data="action_version")],
            [InlineKeyboardButton("📋 New Job Description", callback_data="action_new_jd")],
            [InlineKeyboardButton("🏠 Start Over", callback_data="action_restart")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=chat.id,
            text=(
                "✅ <b>CV Generated Successfully!</b>\n\n"
                "What would you like to do next?\n"
                "Or send /end to finish the session."
            ),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

        try:
            os.remove(file_path)
        except OSError:
            pass

        return CHOOSE_TEMPLATE

    except Exception as e:
        logger.error(f"CV generation error: {e}")
        chat = _require_chat(update)
        await context.bot.send_message(
            chat_id=chat.id,
            text=(
                f"❌ <b>Error generating CV:</b> {str(e)}\n\n"
                f"Please try again with /start"
            ),
            parse_mode="HTML",
        )
        return ConversationHandler.END


async def handle_action_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle post-download action buttons."""
    user_data = _user_data(context)
    query = _require_callback_query(update)
    await query.answer()
    query_data = query.data

    if query_data is None:
        raise ValueError("Action buttons require callback data.")

    if query_data == "action_template":
        keyboard = []
        for key, (label, _) in TEMPLATES.items():
            keyboard.append([InlineKeyboardButton(label, callback_data=f"tpl_{key}")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🎨 <b>Choose a different template:</b>",
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return CHOOSE_TEMPLATE

    elif query_data == "action_version":
        keyboard = [
            [InlineKeyboardButton("🎯 ATS-Optimized (Keyword Heavy)", callback_data="ver_ats_heavy")],
            [InlineKeyboardButton("⚖️ Balanced Professional", callback_data="ver_balanced")],
            [InlineKeyboardButton("📄 Concise 1-Page", callback_data="ver_concise")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "📝 <b>Choose a different version:</b>",
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return CHOOSE_VERSION

    elif query_data == "action_new_jd":
        await query.edit_message_text(
            "📋 <b>Provide a new Job Description:</b>\n\n"
            "• 📝 Paste the JD text directly\n"
            "• 📄 Upload the JD as a PDF file",
            parse_mode="HTML",
        )
        return PASTE_JD

    elif query_data == "action_restart":
        _cleanup_user_files(user_data)
        user_data.clear()
        await query.edit_message_text(
            "🏠 <b>Starting over!</b>\n\n"
            "📎 Upload your CV file (PDF or DOCX):",
            parse_mode="HTML",
        )
        return UPLOAD_CV

    return CHOOSE_TEMPLATE


async def end_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /end — gracefully end session and clean up."""
    user_data = _user_data(context)
    _cleanup_user_files(user_data)
    user_data.clear()
    message = _require_message(update)
    await message.reply_text(
        "👋 <b>Session ended. Thanks for using ATS Resume Optimizer!</b>\n\n"
        "Your data has been cleared. Send /start whenever you're ready for another session.",
        parse_mode="HTML",
    )
    return ConversationHandler.END


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /cancel — exit conversation."""
    user_data = _user_data(context)
    _cleanup_user_files(user_data)
    user_data.clear()
    message = _require_message(update)
    await message.reply_text(
        "👋 <b>Session cancelled.</b> Send /start to begin again!",
        parse_mode="HTML",
    )
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """
🤖 <b>ATS Resume Optimizer — Help</b>

<b>Commands:</b>
/start — Begin a new session
/end — End current session & clear data
/cancel — Cancel current step
/help — Show this message

<b>How to use:</b>
1. Send /start
2. Upload your CV (PDF or DOCX)
3. Paste the Job Description — <i>or upload it as a PDF!</i>
4. View your ATS score & suggestions
5. Choose an optimized version
6. Pick a template format
7. Download your new CV!

<b>Job Description formats accepted:</b>
📝 Paste text directly
📄 Upload as a PDF file

<b>Templates available:</b>
📄 1-Page ATS — Compact, keyword-dense
📋 2-Page Detailed — Spacious, professional
🎨 Modern Sidebar — Two-column layout
📝 Classic Clean — Traditional format

<b>Optimized versions:</b>
🎯 ATS-Optimized — Maximum keyword matching
⚖️ Balanced — Keywords + readability
📄 Concise — Single-page impact
"""
    message = _require_message(update)
    await message.reply_text(help_text, parse_mode="HTML")


# ── Build the bot application ────────────────────────────────────────────────

def create_bot() -> Application:
    """Create and configure the Telegram bot application."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            UPLOAD_CV: [
                MessageHandler(filters.Document.ALL, handle_cv_upload),
            ],
            PASTE_JD: [
                # Accept PDF uploads as JD
                MessageHandler(filters.Document.ALL, handle_jd_pdf_upload),
                # Accept plain text as JD
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_jd_input),
            ],
            CHOOSE_VERSION: [
                CallbackQueryHandler(handle_version_choice, pattern=r"^ver_"),
            ],
            CHOOSE_TEMPLATE: [
                CallbackQueryHandler(handle_template_choice, pattern=r"^tpl_"),
                CallbackQueryHandler(handle_action_buttons, pattern=r"^action_"),
            ],
        },
        fallbacks=[
            CommandHandler("end", end_command),
            CommandHandler("cancel", cancel_command),
        ],
        allow_reentry=True,
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_command))
    # /end also works outside of an active conversation
    app.add_handler(CommandHandler("end", end_command))

    return app
