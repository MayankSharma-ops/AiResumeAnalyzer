"""
AI Engine — uses Google Gemini API for ATS scoring, keyword analysis, and CV optimization.
"""

import json
import re
from typing import Any
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL


def _configure_gemini(api_key: str) -> None:
    """Configure Gemini without relying on incomplete type stubs."""
    configure_fn = getattr(genai, "configure", None)
    if not callable(configure_fn):
        raise ValueError("google.generativeai.configure is unavailable.")
    configure_fn(api_key=api_key)


# Configure Gemini
_configure_gemini(GEMINI_API_KEY)


# ── Master Prompt ────────────────────────────────────────────────────────────

MASTER_PROMPT = """You are an advanced ATS (Applicant Tracking System) analyzer and resume optimization expert with 20+ years of HR and recruiting experience.

TASK:
Analyze the provided Resume against the Job Description. Perform ATS scoring, gap analysis, and generate optimized resumes.

INPUT:
Resume Text:
<<RESUME_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>

INSTRUCTIONS:

1. ATS SCORING:
- Give a score out of 10 (can use decimals like 6.5)
- Base it on:
  - Keyword match (50% weight)
  - Skills relevance (30% weight)
  - Experience alignment (20% weight)

2. KEYWORD ANALYSIS:
- Extract the most important keywords from the Job Description
- Identify which keywords are MATCHED in the resume
- Identify which keywords are MISSING from the resume

3. SUGGESTIONS:
- Provide 5-8 clear, actionable improvement suggestions
- Do NOT add fake experience or qualifications
- Focus on:
  - Better phrasing and action verbs
  - Adding missing keywords naturally into existing content
  - Quantifying achievements (e.g., "improved by 30%")
  - Section ordering and formatting tips

4. RESUME OPTIMIZATION:
Generate 3 improved resume versions:
- "ats_heavy": Keyword-optimized version that maximizes ATS compatibility
- "balanced": Professional version balancing keywords and readability
- "concise": Condensed 1-page version with the most impactful content

IMPORTANT RULES:
- Do NOT invent new projects, jobs, or experience
- Only enhance and rephrase existing content
- Maintain complete truthfulness
- Keep formatting clean with clear section headers

OUTPUT FORMAT (STRICT JSON ONLY — no markdown, no code fences):

{
  "ats_score": <number>,
  "matched_keywords": ["keyword1", "keyword2"],
  "missing_keywords": ["keyword1", "keyword2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "optimized_resumes": {
    "ats_heavy": "<full optimized resume text>",
    "balanced": "<full optimized resume text>",
    "concise": "<full optimized resume text>"
  }
}
"""

RESCORE_PROMPT = """You are an ATS (Applicant Tracking System) scoring engine.

Compare the optimized resume below with the job description and give an updated ATS score.

Optimized Resume:
<<OPTIMIZED_RESUME>>

Job Description:
<<JOB_DESCRIPTION>>

Return ONLY valid JSON (no markdown, no code fences):

{
  "new_score": <number out of 10>,
  "improvement_summary": "<1-2 sentence explanation of improvements>"
}
"""


def _clean_json_response(text: str) -> str:
    """Strip markdown code fences and whitespace from Gemini response."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_json_block(text: str) -> str | None:
    """Extract a JSON object block from an incoming text response."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def _normalize_json_keys(result: dict) -> dict:
    """Normalize dictionary keys so minor formatting issues don't break access."""
    normalized = {}
    for key, value in result.items():
        clean_key = key
        if isinstance(clean_key, str):
            clean_key = clean_key.strip()
            if clean_key.startswith('"') and clean_key.endswith('"'):
                clean_key = clean_key[1:-1].strip()
        normalized[clean_key] = value
    return normalized


def _response_text(response: object) -> str:
    """Extract text from a Gemini response with a helpful failure mode."""
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini response did not contain text output.")
    return text


def _create_model(model_name: str) -> Any:
    """Build a Gemini model without relying on incomplete type stubs."""
    model_cls = getattr(genai, "GenerativeModel", None)
    if model_cls is None:
        raise ValueError("google.generativeai.GenerativeModel is unavailable.")
    return model_cls(model_name)


def _generation_config(**kwargs: object) -> Any:
    """Build generation config without relying on incomplete type stubs."""
    types_module = getattr(genai, "types", None)
    config_cls = getattr(types_module, "GenerationConfig", None)
    if config_cls is None:
        raise ValueError("google.generativeai.types.GenerationConfig is unavailable.")
    return config_cls(**kwargs)


def _render_prompt(template: str, replacements: dict[str, str]) -> str:
    """Render prompts with explicit tokens so JSON braces remain literal."""
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def analyze_cv(resume_text: str, jd_text: str) -> dict:
    """
    Send resume + JD to Gemini and get ATS analysis.
    Returns dict with ats_score, matched_keywords, missing_keywords,
    suggestions, and optimized_resumes.
    """
    model = _create_model(GEMINI_MODEL)

    prompt = _render_prompt(
        MASTER_PROMPT,
        {
            "<<RESUME_TEXT>>": resume_text,
            "<<JOB_DESCRIPTION>>": jd_text,
        },
    )

    response = model.generate_content(
        prompt,
        generation_config=_generation_config(
            temperature=0.3,  # Lower = more deterministic
            max_output_tokens=8192,
        ),
    )

    raw = _clean_json_response(_response_text(response))

    # Try direct JSON parse first, then extraction-based fallback.
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        block = _extract_json_block(raw)
        if block:
            try:
                result = json.loads(block)
            except json.JSONDecodeError as second_error:
                raise ValueError(
                    f"Gemini returned invalid JSON block. "
                    f"Original raw: {raw[:500]}\n"  # first 500 chars
                    f"Parse error: {second_error}",
                )
        else:
            raise ValueError(
                f"Gemini did not return valid JSON. Raw response (first 500 chars):\n{raw[:500]}"
            )

    # Support the rare case where Gemini returns a string containing JSON.
    if isinstance(result, str):
        try:
            inner = json.loads(result)
            result = inner
        except json.JSONDecodeError:
            raise ValueError(
                "Gemini response was a JSON string, but inner payload is invalid JSON."
            )

    if not isinstance(result, dict):
        raise ValueError(
            f"Gemini response parsed to {type(result).__name__}, expected object."
        )

    result = _normalize_json_keys(result)

    # Validate required keys
    required_keys = ["ats_score", "matched_keywords", "missing_keywords", "suggestions", "optimized_resumes"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing key '{key}' in Gemini response.")

    return result


def rescore_cv(optimized_resume: str, jd_text: str) -> dict:
    """
    Re-score an optimized resume against the JD.
    Returns dict with new_score and improvement_summary.
    """
    model = _create_model(GEMINI_MODEL)

    prompt = _render_prompt(
        RESCORE_PROMPT,
        {
            "<<OPTIMIZED_RESUME>>": optimized_resume,
            "<<JOB_DESCRIPTION>>": jd_text,
        },
    )

    response = model.generate_content(
        prompt,
        generation_config=_generation_config(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    raw = _clean_json_response(_response_text(response))

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError("Gemini did not return valid JSON for re-scoring.")

    return result
