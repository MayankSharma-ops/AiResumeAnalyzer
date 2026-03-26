"""
Telegram Bot — conversational ATS analyzer with inline keyboards.

Flow:
  /start → Upload CV → Paste JD → Analysis (score + keywords + suggestions)
  → Choose optimized version → Choose template → Download DOCX + new score
"""

import asyncio
import os
import logging
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


# ── Bot Handlers ─────────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /start — welcome message."""
    # Clear any previous data
    user_data = _user_data(context)
    user_data.clear()
    message = _require_message(update)

    welcome = """
🤖 <b>ATS Resume Optimizer Bot</b>

I'll help you beat ATS systems and land interviews!

<b>Here's how it works:</b>
1️⃣ Upload your CV (PDF or DOCX)
2️⃣ Paste the Job Description
3️⃣ Get your ATS score & analysis
4️⃣ Download optimized CV in multiple templates

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
            f"📋 <b>Now paste the Job Description:</b>\n"
            f"(Copy the full JD text and send it here)",
            parse_mode="HTML",
        )
        return PASTE_JD

    except Exception as e:
        logger.error(f"CV parsing error: {e}")
        await message.reply_text(
            f"❌ Error parsing CV: {str(e)}\nPlease try another file.",
        )
        return UPLOAD_CV


async def handle_jd_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle JD text input — analyze CV against JD."""
    user_data = _user_data(context)
    message = _require_message(update)
    jd_text = message.text

    if not jd_text or len(jd_text.strip()) < 20:
        await message.reply_text(
            "⚠️ Job Description seems too short. "
            "Please paste the complete JD text.",
        )
        return PASTE_JD

    user_data["jd_text"] = jd_text

    # Processing message
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
        # Call Gemini
        cv_text = user_data["cv_text"]
        result = analyze_cv(cv_text, jd_text)
        user_data["analysis"] = result

        # Build and send analysis message
        analysis_msg = _build_analysis_message(result)

        # Version selection buttons
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

    # Template selection buttons
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
        # Get the optimized resume text
        analysis = user_data["analysis"]
        version_key = user_data["chosen_version"]
        optimized_text = analysis["optimized_resumes"][version_key]

        # Generate DOCX
        file_path = generate_cv(
            optimized_text,
            template_key,
            version_name=version_key,
        )

        # Re-score the optimized CV
        jd_text = user_data["jd_text"]
        rescore_result = rescore_cv(optimized_text, jd_text)
        new_score = rescore_result.get("new_score", "N/A")
        improvement = rescore_result.get("improvement_summary", "")

        old_score = analysis["ats_score"]
        old_emoji = _score_emoji(old_score)
        new_emoji = _score_emoji(new_score) if isinstance(new_score, (int, float)) else "⚪"

        chat = _require_chat(update)

        # Send the file
        with open(file_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat.id,
                document=f,
                filename=os.path.basename(file_path),
                caption=(
                    f"📄 <b>Your Optimized CV</b>\n\n"
                    f"📊 <b>Score Comparison:</b>\n"
                    f"   Before: {old_emoji} {old_score}/10\n"
                    f"   After:  {new_emoji} {new_score}/10\n\n"
                    f"💬 {improvement}"
                ),
                parse_mode="HTML",
            )

        # Action buttons
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
                "What would you like to do next?"
            ),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

        # Clean up temp file
        try:
            os.remove(file_path)
        except OSError:
            pass

        return CHOOSE_TEMPLATE  # Stay in this state for follow-up actions

    except Exception as e:
        logger.error(f"CV generation error: {e}")
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
        # Show template selection again
        keyboard = []
        for key, (label, _) in TEMPLATES.items():
            keyboard.append([InlineKeyboardButton(label, callback_data=f"tpl_{key}")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"🎨 <b>Choose a different template:</b>",
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
            f"📝 <b>Choose a different version:</b>",
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return CHOOSE_VERSION

    elif query_data == "action_new_jd":
        await query.edit_message_text(
            "📋 <b>Paste the new Job Description:</b>",
            parse_mode="HTML",
        )
        return PASTE_JD

    elif query_data == "action_restart":
        user_data.clear()
        await query.edit_message_text(
            "🏠 <b>Starting over!</b>\n\n"
            "📎 Upload your CV file (PDF or DOCX):",
            parse_mode="HTML",
        )
        return UPLOAD_CV

    return CHOOSE_TEMPLATE


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /cancel — exit conversation."""
    user_data = _user_data(context)
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
/cancel — Cancel current session
/help — Show this message

<b>How to use:</b>
1. Send /start
2. Upload your CV (PDF or DOCX)
3. Paste the Job Description
4. View your ATS score & suggestions
5. Choose an optimized version
6. Pick a template format
7. Download your new CV!

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

    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            UPLOAD_CV: [
                MessageHandler(filters.Document.ALL, handle_cv_upload),
            ],
            PASTE_JD: [
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
        fallbacks=[CommandHandler("cancel", cancel_command)],
        allow_reentry=True,
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_command))

    return app
