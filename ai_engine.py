"""
AI Engine — uses Google Gemini API for ATS scoring, keyword analysis, and CV optimization.
"""

import json
import re
from typing import Any
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
import logging

logger = logging.getLogger(__name__)


def _configure_gemini(api_key: str) -> None:
    configure_fn = getattr(genai, "configure", None)
    if not callable(configure_fn):
        raise ValueError("google.generativeai.configure is unavailable.")
    configure_fn(api_key=api_key)


_configure_gemini(GEMINI_API_KEY)


# ── Prompts ───────────────────────────────────────────────────────────────────

MASTER_PROMPT = """You are an ATS resume analyzer. Analyze the resume against the job description.

Resume:
<<RESUME_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>

Respond with ONLY a JSON object. No explanation, no markdown, no code fences. Start your response with { and end with }.

Required format:
{
"ats_score": 7.5,
"matched_keywords": ["python", "sql"],
"missing_keywords": ["docker", "aws"],
"suggestions": ["Add quantified achievements", "Include missing keywords"],
"optimized_resumes": {
"ats_heavy": "Full resume text optimized for ATS keyword matching",
"balanced": "Full resume text balancing keywords and readability",
"concise": "Condensed single-page resume with most impactful content"
}
}

Rules:
- ats_score is a number 0-10
- matched_keywords and missing_keywords are arrays of strings
- suggestions is an array of 5-8 strings
- optimized_resumes contains three string values
- Do NOT invent experience or qualifications
- Do NOT use markdown in resume text values
"""

RESCORE_PROMPT = """Compare this resume to the job description and return a JSON score.

Resume:
<<OPTIMIZED_RESUME>>

Job Description:
<<JOB_DESCRIPTION>>

Respond with ONLY this JSON (no markdown, no code fences):
{"new_score": 8.0, "improvement_summary": "Brief explanation here"}
"""


def _render_prompt(template: str, replacements: dict) -> str:
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini response contained no text.")
    return text


def _create_model(model_name: str) -> Any:
    model_cls = getattr(genai, "GenerativeModel", None)
    if model_cls is None:
        raise ValueError("google.generativeai.GenerativeModel unavailable.")
    return model_cls(model_name)


def _generation_config(**kwargs: object) -> Any:
    types_module = getattr(genai, "types", None)
    config_cls = getattr(types_module, "GenerationConfig", None)
    if config_cls is None:
        raise ValueError("GenerationConfig unavailable.")
    return config_cls(**kwargs)


def _extract_json(text: str) -> dict:
    """
    Robustly extract and parse JSON from Gemini response.
    Handles: markdown fences, leading/trailing text, malformed keys.
    """
    # ── DEBUG: log raw response so we can see exactly what Gemini sends ──────
    logger.info("=" * 60)
    logger.info("RAW GEMINI RESPONSE repr (first 1000 chars):")
    logger.info(repr(text[:1000]))
    logger.info("=" * 60)

    # Step 1: strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Step 2: find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found. Got: {repr(text[:300])}")
    text = text[start:end + 1]

    # Step 3: attempt direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parse failed: {e}")

    # Step 4: repair — newlines inside key names
    # Gemini sometimes emits:  {\n  "ats_score": ...}  where the \n is a
    # LITERAL backslash-n inside the JSON string rather than a real newline.
    # OR it emits real newlines between { and the first key that get absorbed.
    repaired = text

    # 4a: real newlines + whitespace before a quote → just a quote
    repaired = re.sub(r'\n\s*"', '"', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.warning(f"After newline repair: {e}")

    # 4b: literal \n (backslash + n) sequences before quotes
    repaired2 = repaired.replace('\\n', ' ')
    try:
        return json.loads(repaired2)
    except json.JSONDecodeError as e:
        logger.warning(f"After \\n replacement: {e}")

    # Step 5: last resort regex extraction
    logger.warning("Falling back to regex field extraction")
    return _regex_extract_fields(text)


def _regex_extract_fields(text: str) -> dict:
    """Last-resort extraction using regex when JSON is too malformed."""

    def extract_number(key: str) -> float:
        m = re.search(rf'["\']?{re.escape(key)}["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        return float(m.group(1)) if m else 0.0

    def extract_array(key: str) -> list:
        m = re.search(rf'["\']?{re.escape(key)}["\']?\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if not m:
            return []
        return re.findall(r'"([^"]*)"', m.group(1))

    def extract_nested_string(outer_key: str, inner_key: str) -> str:
        outer_m = re.search(
            rf'["\']?{re.escape(outer_key)}["\']?\s*:\s*\{{(.*?)\}}',
            text, re.DOTALL
        )
        if not outer_m:
            return ""
        inner_text = outer_m.group(1)
        inner_m = re.search(
            rf'["\']?{re.escape(inner_key)}["\']?\s*:\s*"((?:[^"\\]|\\.)*)"',
            inner_text, re.DOTALL
        )
        return inner_m.group(1) if inner_m else ""

    fallback = "Optimization unavailable — please try again."
    return {
        "ats_score": extract_number("ats_score"),
        "matched_keywords": extract_array("matched_keywords"),
        "missing_keywords": extract_array("missing_keywords"),
        "suggestions": extract_array("suggestions") or ["Review keyword alignment"],
        "optimized_resumes": {
            "ats_heavy": extract_nested_string("optimized_resumes", "ats_heavy") or fallback,
            "balanced": extract_nested_string("optimized_resumes", "balanced") or fallback,
            "concise": extract_nested_string("optimized_resumes", "concise") or fallback,
        }
    }


def _normalize_keys_deep(obj: Any) -> Any:
    """Recursively strip whitespace and surrounding quotes from all dict keys."""
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if isinstance(k, str):
                k = k.strip('\n\r\t \'"')  # strip whitespace AND quotes in one go
            clean[k] = _normalize_keys_deep(v)
        return clean
    elif isinstance(obj, list):
        return [_normalize_keys_deep(i) for i in obj]
    return obj


def _call_gemini(model: Any, prompt: str, temperature: float, max_tokens: int) -> str:
    """Call Gemini with JSON mime type, falling back to plain if unsupported."""
    for use_json_mime in [True, False]:
        try:
            kwargs: dict = {"temperature": temperature, "max_output_tokens": max_tokens}
            if use_json_mime:
                kwargs["response_mime_type"] = "application/json"
            gen_config = _generation_config(**kwargs)
            response = model.generate_content(prompt, generation_config=gen_config)
            return _response_text(response)
        except Exception as e:
            if use_json_mime:
                logger.warning(f"JSON mime call failed ({e}), retrying without mime type")
            else:
                raise
    raise RuntimeError("All Gemini call attempts failed")


def analyze_cv(resume_text: str, jd_text: str) -> dict:
    """Analyze resume against job description using Gemini."""
    model = _create_model(GEMINI_MODEL)
    prompt = _render_prompt(MASTER_PROMPT, {
        "<<RESUME_TEXT>>": resume_text,
        "<<JOB_DESCRIPTION>>": jd_text,
    })

    raw = _call_gemini(model, prompt, temperature=0.3, max_tokens=8192)
    result = _extract_json(raw)
    result = _normalize_keys_deep(result)

    logger.info(f"Parsed top-level keys: {list(result.keys())}")

    # Apply safe defaults so we never crash on missing keys
    result.setdefault("ats_score", 0)
    result.setdefault("matched_keywords", [])
    result.setdefault("missing_keywords", [])
    result.setdefault("suggestions", [])

    fallback = resume_text  # Use original as last resort
    opt = _normalize_keys_deep(result.get("optimized_resumes") or {})
    opt.setdefault("ats_heavy", fallback)
    opt.setdefault("balanced", fallback)
    opt.setdefault("concise", fallback)
    result["optimized_resumes"] = opt

    return result


def rescore_cv(optimized_resume: str, jd_text: str) -> dict:
    """Re-score an optimized resume against the JD."""
    model = _create_model(GEMINI_MODEL)
    prompt = _render_prompt(RESCORE_PROMPT, {
        "<<OPTIMIZED_RESUME>>": optimized_resume,
        "<<JOB_DESCRIPTION>>": jd_text,
    })

    try:
        raw = _call_gemini(model, prompt, temperature=0.2, max_tokens=1024)
        result = _extract_json(raw)
        return _normalize_keys_deep(result)
    except Exception as e:
        logger.warning(f"Rescore failed: {e}")
        return {"new_score": "N/A", "improvement_summary": "Score calculation unavailable."}