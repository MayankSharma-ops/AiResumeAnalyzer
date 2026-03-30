"""
AI engine for ATS scoring, keyword analysis, and resume optimization.
"""

import json
import logging
import math
import re
from typing import Any

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


def _configure_gemini(api_key: str) -> None:
    configure_fn = getattr(genai, "configure", None)
    if not callable(configure_fn):
        raise ValueError("google.generativeai.configure is unavailable.")
    configure_fn(api_key=api_key)


_configure_gemini(GEMINI_API_KEY)


ANALYSIS_PROMPT = """You are a strict ATS scoring engine.

Analyze the resume against the job description before any optimization.

Resume:
<<RESUME_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>

Return ONLY valid JSON.
No markdown.
No code fences.

{
  "ats_score": 2.5,
  "matched_keywords": ["word1", "word2", "word3"],
  "missing_keywords": ["word4", "word5", "word6"],
  "suggestions": ["tip1", "tip2", "tip3", "tip4", "tip5", "tip6"]
}

Rules:
- Score must be between 0 and 10.
- matched_keywords: top relevant JD terms already present in resume.
- missing_keywords: top relevant JD terms missing from resume.
- suggestions: exactly 6 concise, actionable improvements.
"""


TARGET_CONTEXT_PROMPT = """You are building a resume-tailoring brief from a resume and job description.

Resume:
<<RESUME_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>

Return ONLY valid JSON.
No markdown.
No code fences.

{
  "target_role": "Senior Data Analyst",
  "seniority": "mid-senior",
  "summary_focus": "analytics, stakeholder reporting, experimentation, and SQL-heavy decision support",
  "ats_focus": "Prioritize the repeated platform, tool, and domain terms from the JD.",
  "priority_keywords": ["SQL", "Python", "A/B testing"],
  "priority_skills": ["SQL", "Python", "Tableau"],
  "experience_focus": [
    "Highlight dashboards, reporting ownership, and business impact.",
    "Surface experimentation and cross-functional collaboration."
  ],
  "preferred_sections": ["PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "EXPERIENCE", "PROJECTS", "KEYWORDS"]
}

Rules:
- Use only information grounded in the provided resume and JD.
- Do not invent experience, projects, tools, achievements, or credentials.
- Keep each field concise and specific to this JD.
- priority_keywords: 10 to 15 JD terms or phrases.
- priority_skills: 6 to 10 key skills/tools.
- experience_focus: 3 to 5 rewrite priorities.
- preferred_sections: only standard resume section headings.
"""


OPTIMIZE_PROMPT = """You are an ATS resume optimizer.

Rewrite this resume to better match the job description.

Original Resume:
<<RESUME_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>

Keywords to include where truthful:
<<MISSING_KEYWORDS>>

Target Version:
<<VERSION_NAME>>

Version Rules:
<<VERSION_RULES>>

API Tailoring Context:
- Target Role: <<TARGET_ROLE>>
- Seniority: <<TARGET_SENIORITY>>
- Summary Focus: <<SUMMARY_FOCUS>>
- ATS Focus: <<ATS_FOCUS>>
- Priority Keywords: <<PRIORITY_KEYWORDS>>
- Priority Skills: <<PRIORITY_SKILLS>>
- Experience Focus:
<<EXPERIENCE_FOCUS>>
- Preferred Sections: <<PREFERRED_SECTIONS>>

Additional Guidance:
<<ADDITIONAL_GUIDANCE>>

Mandatory Rules:
1. Keep the candidate's real identity, contact details, companies, dates, and education.
2. Never invent job titles, companies, projects, certifications, dates, achievements, or years of experience.
3. Improve wording and structure using JD terminology where truthful.
4. Strengthen the summary and skills sections.
5. Preserve a realistic resume format with clear headings and bullet points.
6. Prefer these exact headings when relevant: PROFESSIONAL SUMMARY, TECHNICAL SKILLS, EXPERIENCE, EDUCATION, PROJECTS, CERTIFICATIONS, KEYWORDS.
7. Adapt the summary, skills ordering, keyword emphasis, and experience wording to this JD.
8. Use the API tailoring context above to change the resume context for the target role, but stay truthful to the source resume.
9. If the JD emphasizes domain-specific tools or terminology, surface the closest truthful evidence from the resume early.

Return ONLY the full resume text for this one version.
Use real line breaks.
Do NOT return JSON.
Do NOT use markdown code fences.
"""


VERSION_RULES = {
    "ats_heavy": (
        "Maximize keyword coverage. Add a strong summary, dense skills section, "
        "and a short keyword section near the end. Prioritize ATS matching over style."
    ),
    "balanced": (
        "Balance keyword coverage with professional readability. Make the resume sound natural, "
        "clear, and recruiter-friendly."
    ),
    "concise": (
        "Keep the resume tight and high impact. Prefer shorter bullets, reduced repetition, "
        "and a one-page style where possible."
    ),
}

OPTIMIZATION_TARGET_SCORE = 8.0
OPTIMIZATION_ATTEMPTS = 2

SECTION_HEADING_ALIASES = {
    "summary": "PROFESSIONAL SUMMARY",
    "professional summary": "PROFESSIONAL SUMMARY",
    "professional profile": "PROFESSIONAL SUMMARY",
    "profile": "PROFESSIONAL SUMMARY",
    "objective": "OBJECTIVE",
    "skills": "SKILLS",
    "key skills": "SKILLS",
    "core skills": "TECHNICAL SKILLS",
    "technical skills": "TECHNICAL SKILLS",
    "tools & technologies": "TECHNICAL SKILLS",
    "tools and technologies": "TECHNICAL SKILLS",
    "technical proficiency": "TECHNICAL SKILLS",
    "experience": "EXPERIENCE",
    "work experience": "EXPERIENCE",
    "professional experience": "EXPERIENCE",
    "employment history": "EXPERIENCE",
    "work history": "EXPERIENCE",
    "education": "EDUCATION",
    "projects": "PROJECTS",
    "certifications": "CERTIFICATIONS",
    "certificates": "CERTIFICATIONS",
    "awards": "AWARDS",
    "achievements": "ACHIEVEMENTS",
    "languages": "LANGUAGES",
    "keywords": "KEYWORDS",
    "relevant keywords": "KEYWORDS",
    "target role keywords": "KEYWORDS",
}


def _render_prompt(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def _create_model(model_name: str) -> Any:
    model_cls = getattr(genai, "GenerativeModel", None)
    if model_cls is None:
        raise ValueError("google.generativeai.GenerativeModel is unavailable.")
    return model_cls(model_name)


def _generation_config(**kwargs: object) -> Any:
    types_module = getattr(genai, "types", None)
    config_cls = getattr(types_module, "GenerationConfig", None)
    if config_cls is None:
        raise ValueError("google.generativeai.types.GenerationConfig is unavailable.")
    return config_cls(**kwargs)


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if candidates:
        collected: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    collected.append(part_text)
        if collected:
            return "\n".join(collected)

    raise ValueError("Gemini response contained no text.")


def _call_gemini(
    model: Any,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    json_output: bool,
) -> str:
    for use_json_mime in ([True, False] if json_output else [False]):
        try:
            kwargs: dict[str, object] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if use_json_mime:
                kwargs["response_mime_type"] = "application/json"
            response = model.generate_content(
                prompt,
                generation_config=_generation_config(**kwargs),
            )
            text = _response_text(response).strip()
            logger.info(
                "Gemini response received: json_output=%s mime=%s chars=%s",
                json_output,
                use_json_mime,
                len(text),
            )
            return text
        except Exception as exc:
            if use_json_mime:
                logger.warning("Gemini JSON mime failed (%s); retrying without mime", exc)
            else:
                raise
    raise RuntimeError("All Gemini call attempts failed.")


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_json_fragment(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. Got: {repr(cleaned[:300])}")

    end = cleaned.rfind("}")
    if end == -1 or end <= start:
        fragment = cleaned[start:]
        logger.warning("Gemini returned truncated JSON fragment: %s", repr(fragment[:300]))
        return fragment

    return cleaned[start:end + 1]


def _extract_partial_array(text: str, key: str) -> list[str]:
    pattern = rf'["\']?{re.escape(key)}["\']?\s*:\s*\[(.*?)(?:\]|$)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []

    values = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))
    result: list[str] = []
    for value in values:
        unescaped = value.replace('\\"', '"').replace("\\n", " ").strip()
        if unescaped:
            result.append(unescaped)
    return result


def _extract_partial_string(text: str, key: str) -> str:
    closed = re.search(
        rf'["\']?{re.escape(key)}["\']?\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        re.DOTALL,
    )
    if closed:
        return closed.group(1)

    partial = re.search(rf'["\']?{re.escape(key)}["\']?\s*:\s*"(.*)$', text, re.DOTALL)
    if not partial:
        return ""

    value = partial.group(1)
    value = value.replace("\\n", "\n").replace('\\"', '"')
    return value.strip()


def _regex_extract_fields(text: str) -> dict[str, Any]:
    def extract_number(key: str) -> float:
        match = re.search(
            rf'["\']?{re.escape(key)}["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            text,
        )
        return float(match.group(1)) if match else 0.0

    return {
        "ats_score": extract_number("ats_score"),
        "matched_keywords": _extract_partial_array(text, "matched_keywords"),
        "missing_keywords": _extract_partial_array(text, "missing_keywords"),
        "suggestions": _extract_partial_array(text, "suggestions"),
        "ats_heavy": _extract_partial_string(text, "ats_heavy"),
        "balanced": _extract_partial_string(text, "balanced"),
        "concise": _extract_partial_string(text, "concise"),
        "target_role": _extract_partial_string(text, "target_role"),
        "seniority": _extract_partial_string(text, "seniority"),
        "summary_focus": _extract_partial_string(text, "summary_focus"),
        "ats_focus": _extract_partial_string(text, "ats_focus"),
        "priority_keywords": _extract_partial_array(text, "priority_keywords"),
        "priority_skills": _extract_partial_array(text, "priority_skills"),
        "experience_focus": _extract_partial_array(text, "experience_focus"),
        "preferred_sections": _extract_partial_array(text, "preferred_sections"),
    }


def _extract_json(text: str) -> dict[str, Any]:
    fragment = _extract_json_fragment(text)

    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r",(\s*[}\]])", r"\1", fragment)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning("Falling back to regex JSON extraction")
    return _regex_extract_fields(fragment)


def _normalize_keys_deep(obj: Any) -> Any:
    if isinstance(obj, dict):
        normalized: dict[Any, Any] = {}
        for key, value in obj.items():
            clean_key = key.strip('\n\r\t \'"') if isinstance(key, str) else key
            normalized[clean_key] = _normalize_keys_deep(value)
        return normalized
    if isinstance(obj, list):
        return [_normalize_keys_deep(item) for item in obj]
    return obj


def _extract_jd_keywords(jd_text: str, limit: int = 35) -> list[str]:
    stop_words = {
        "the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "of",
        "with", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "we", "you", "your", "our", "their", "this", "that",
        "these", "those", "from", "into", "through", "will", "would", "could",
        "should", "can", "may", "might", "must", "required", "preferred",
        "responsibilities", "qualifications", "ability", "strong", "excellent",
        "good", "proven", "demonstrated", "experience", "team", "company",
        "role", "position", "job", "work", "working",
    }

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]{2,}\b", jd_text)
    keywords: list[str] = []
    seen: set[str] = set()

    for word in words:
        lowered = word.lower()
        if lowered in stop_words or lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(word)
        if len(keywords) >= limit:
            break

    return keywords


def _normalize_for_matching(text: str) -> str:
    normalized = text.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9+#./]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _keyword_in_resume(keyword: str, resume_normalized: str, resume_tokens: set[str]) -> bool:
    normalized_keyword = _normalize_for_matching(keyword)
    if not normalized_keyword:
        return False
    if f" {normalized_keyword} " in f" {resume_normalized} ":
        return True

    keyword_tokens = [token for token in normalized_keyword.split() if token]
    return bool(keyword_tokens) and all(token in resume_tokens for token in keyword_tokens)


def _split_keywords(resume_text: str, jd_keywords: list[str]) -> tuple[list[str], list[str]]:
    resume_normalized = _normalize_for_matching(resume_text)
    resume_tokens = set(resume_normalized.split())
    matched: list[str] = []
    missing: list[str] = []

    for keyword in jd_keywords:
        if _keyword_in_resume(keyword, resume_normalized, resume_tokens):
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing


def _local_ats_score(resume_text: str, jd_keywords: list[str]) -> float:
    if not jd_keywords:
        return 5.0

    matched, _ = _split_keywords(resume_text, jd_keywords)
    ratio = len(matched) / len(jd_keywords)
    score = round(min(10.0, max(1.0, ratio * 10)), 1)
    logger.info("Local ATS score: %s/%s keywords matched -> %s", len(matched), len(jd_keywords), score)
    return score


def _default_suggestions(missing_keywords: list[str]) -> list[str]:
    top_missing = ", ".join(missing_keywords[:5]) if missing_keywords else "key job requirements"
    return [
        "Add a stronger Professional Summary aligned to the target role.",
        f"Include missing keywords naturally in Skills and Experience sections, especially: {top_missing}.",
        "Rewrite experience bullets with stronger action verbs and clearer outcomes.",
        "Group technical skills into a dedicated section for better ATS matching.",
        "Quantify achievements with metrics wherever the resume already supports them.",
        "Reorder sections so the most relevant skills and experience appear earlier.",
    ]


def _clean_text_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _clean_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = _clean_text_value(item)
        if text:
            cleaned.append(text)
    return _dedupe_preserve_order(cleaned)[:limit]


def _sanitize_score(value: Any, fallback: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return round(fallback, 1)

    if not math.isfinite(score) or score < 0 or score > 10:
        return round(fallback, 1)

    return round(score, 1)


def _canonical_heading(line: str) -> str | None:
    normalized = re.sub(r"\s+", " ", line.strip().strip(":")).lower()
    return SECTION_HEADING_ALIASES.get(normalized)


def _split_resume_sections(resume_text: str) -> tuple[list[str], dict[str, str]]:
    order = ["HEADER"]
    sections: dict[str, str] = {"HEADER": ""}
    current_section = "HEADER"
    current_lines: list[str] = []

    for raw_line in resume_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        heading = _canonical_heading(raw_line)
        if heading:
            body = "\n".join(current_lines).strip()
            if body:
                existing = sections.get(current_section, "")
                sections[current_section] = "\n\n".join(part for part in [existing, body] if part).strip()
            current_section = heading
            if current_section not in sections:
                sections[current_section] = ""
                order.append(current_section)
            current_lines = []
            continue
        current_lines.append(raw_line.rstrip())

    tail = "\n".join(current_lines).strip()
    if tail:
        existing = sections.get(current_section, "")
        sections[current_section] = "\n\n".join(part for part in [existing, tail] if part).strip()

    return order, sections


def _render_resume_sections(order: list[str], sections: dict[str, str]) -> str:
    parts: list[str] = []
    for name in order:
        body = sections.get(name, "").strip()
        if not body:
            continue
        if name == "HEADER":
            parts.append(body)
        else:
            parts.append(name)
            parts.append(body)
    return "\n\n".join(parts).strip()


def _upsert_section(resume_text: str, heading: str, lines: list[str]) -> str:
    order, sections = _split_resume_sections(resume_text)
    normalized_lines = _dedupe_preserve_order(lines)
    if not normalized_lines:
        return _render_resume_sections(order, sections)

    existing_lines = [
        line.strip()
        for line in sections.get(heading, "").splitlines()
        if line.strip()
    ]
    merged_lines = _dedupe_preserve_order(existing_lines + normalized_lines)
    sections[heading] = "\n".join(merged_lines)

    if heading not in order:
        order.append(heading)

    return _render_resume_sections(order, sections)


def _format_prompt_list(items: list[str], fallback: str) -> str:
    return ", ".join(items) if items else fallback


def _format_prompt_lines(items: list[str], fallback: str) -> str:
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)


def _infer_target_role(jd_text: str) -> str:
    title_markers = (
        "engineer",
        "developer",
        "manager",
        "analyst",
        "designer",
        "architect",
        "scientist",
        "specialist",
        "consultant",
        "administrator",
        "lead",
        "director",
        "coordinator",
        "officer",
        "executive",
    )
    lines = [
        re.sub(r"\s+", " ", line.strip(" -*\t"))
        for line in jd_text.splitlines()
        if line.strip()
    ]

    for line in lines[:10]:
        lowered = line.lower().rstrip(":")
        if len(line.split()) <= 12 and any(marker in lowered for marker in title_markers):
            return line.rstrip(":")

    return "Target role from the job description"


def _default_target_context(jd_text: str, jd_keywords: list[str]) -> dict[str, Any]:
    top_terms = jd_keywords[:12]
    role = _infer_target_role(jd_text)

    return {
        "target_role": role,
        "seniority": "",
        "summary_focus": (
            f"Align the resume toward {role} requirements with emphasis on "
            f"{_format_prompt_list(top_terms[:5], 'the most relevant JD terms')}."
        ),
        "ats_focus": "Prioritize repeated JD tools, platforms, business domain terms, and measurable impact language.",
        "priority_keywords": top_terms,
        "priority_skills": top_terms[:8],
        "experience_focus": [
            "Lead with the most relevant accomplishments for this JD.",
            "Surface matching tools, platforms, and domain language earlier in the resume.",
            "Rewrite bullets to be clearer, impact-oriented, and closer to the JD wording where truthful.",
        ],
        "preferred_sections": [
            "PROFESSIONAL SUMMARY",
            "TECHNICAL SKILLS",
            "EXPERIENCE",
            "PROJECTS",
            "KEYWORDS",
        ],
    }


def _sanitize_target_context(
    context: dict[str, Any],
    *,
    jd_text: str,
    jd_keywords: list[str],
) -> dict[str, Any]:
    fallback = _default_target_context(jd_text, jd_keywords)

    preferred_sections = _clean_string_list(context.get("preferred_sections"), limit=6)
    preferred_sections = [section.upper() for section in preferred_sections] or fallback["preferred_sections"]

    return {
        "target_role": _clean_text_value(context.get("target_role")) or fallback["target_role"],
        "seniority": _clean_text_value(context.get("seniority")) or fallback["seniority"],
        "summary_focus": _clean_text_value(context.get("summary_focus")) or fallback["summary_focus"],
        "ats_focus": _clean_text_value(context.get("ats_focus")) or fallback["ats_focus"],
        "priority_keywords": _clean_string_list(context.get("priority_keywords"), limit=15) or fallback["priority_keywords"],
        "priority_skills": _clean_string_list(context.get("priority_skills"), limit=10) or fallback["priority_skills"],
        "experience_focus": _clean_string_list(context.get("experience_focus"), limit=5) or fallback["experience_focus"],
        "preferred_sections": preferred_sections,
    }


def _build_target_context(
    model: Any,
    *,
    resume_text: str,
    jd_text: str,
    jd_keywords: list[str],
) -> dict[str, Any]:
    prompt = _render_prompt(
        TARGET_CONTEXT_PROMPT,
        {
            "<<RESUME_TEXT>>": resume_text,
            "<<JOB_DESCRIPTION>>": jd_text,
        },
    )

    try:
        raw = _call_gemini(
            model,
            prompt,
            temperature=0.15,
            max_tokens=1024,
            json_output=True,
        )
        parsed = _normalize_keys_deep(_extract_json(raw))
    except Exception as exc:
        logger.warning("Gemini tailoring context failed (%s); using local fallback", exc)
        parsed = {}

    return _sanitize_target_context(parsed, jd_text=jd_text, jd_keywords=jd_keywords)


def _build_keyword_summary(
    jd_keywords: list[str],
    version_key: str,
    target_context: dict[str, Any],
) -> str:
    target_role = target_context.get("target_role") or "the target role"
    summary_focus = target_context.get("summary_focus") or ""
    prioritized_terms = target_context.get("priority_keywords", [])
    top_terms = ", ".join((prioritized_terms or jd_keywords)[:6]) or "the target role"

    if version_key == "ats_heavy":
        return (
            f"Resume tailored for {target_role} opportunities, emphasizing {summary_focus or top_terms}. "
            f"Keywords surfaced prominently include {top_terms}."
        )
    if version_key == "concise":
        return (
            f"Condensed resume aligned to {target_role} roles, prioritizing {summary_focus or top_terms} "
            "for faster ATS and recruiter scanning."
        )
    return (
        f"Resume aligned to {target_role} roles with focus on {summary_focus or top_terms}, "
        "balancing keyword visibility with professional readability."
    )


def _build_additional_guidance(
    *,
    target_context: dict[str, Any],
    remaining_missing: list[str],
    attempt_number: int,
    current_score: float | None,
) -> str:
    lines: list[str] = []

    if current_score is None:
        lines.append(
            f"Aim for a strong ATS match and try to reach at least {OPTIMIZATION_TARGET_SCORE:.1f}/10 "
            "locally where the resume truthfully supports it."
        )
    else:
        lines.append(
            f"Previous draft scored {current_score:.1f}/10 locally. Improve truthful keyword coverage "
            f"toward {OPTIMIZATION_TARGET_SCORE:.1f}/10 or higher."
        )

    if attempt_number > 1:
        lines.append("This is a refinement pass. Keep the resume realistic, but make the targeting more explicit.")

    if remaining_missing:
        lines.append(
            "Still-missing terms to surface where supported: "
            + _format_prompt_list(remaining_missing[:12], "none")
        )

    ats_focus = target_context.get("ats_focus")
    if ats_focus:
        lines.append(f"ATS emphasis: {ats_focus}")

    return "\n".join(lines)


def _enforce_improvement(
    candidate_text: str,
    original_text: str,
    jd_keywords: list[str],
    missing_keywords: list[str],
    version_key: str,
    target_context: dict[str, Any],
) -> str:
    base_text = candidate_text.strip() or original_text.strip()
    priority_keywords = target_context.get("priority_keywords", [])
    priority_skills = target_context.get("priority_skills", [])
    combined_keywords = _dedupe_preserve_order(priority_keywords + jd_keywords + missing_keywords)
    skills_keywords = _dedupe_preserve_order(priority_skills + combined_keywords)

    improved = _upsert_section(
        base_text,
        "PROFESSIONAL SUMMARY",
        [_build_keyword_summary(combined_keywords, version_key, target_context)],
    )
    improved = _upsert_section(
        improved,
        "TECHNICAL SKILLS",
        [", ".join(skills_keywords[:20])],
    )

    keyword_source = _dedupe_preserve_order(priority_keywords + combined_keywords)
    keyword_slice = keyword_source[:25] if version_key == "ats_heavy" else keyword_source[:15]
    improved = _upsert_section(
        improved,
        "KEYWORDS",
        [", ".join(keyword_slice)],
    )

    return improved.strip()


def _build_version_report(
    *,
    baseline_score: float,
    version_score: float,
    resume_text: str,
    jd_keywords: list[str],
) -> dict[str, Any]:
    baseline_score = _sanitize_score(baseline_score, baseline_score)
    version_score = _sanitize_score(version_score, baseline_score)
    matched, _ = _split_keywords(resume_text, jd_keywords)
    total = len(jd_keywords)
    percent = round((len(matched) / total) * 100) if total else 0
    delta = round(version_score - baseline_score, 1)

    if delta > 0:
        summary = (
            f"ATS keyword coverage improved by {delta:.1f} points. "
            f"Resume now matches {len(matched)} of {total} key JD terms ({percent}% coverage)."
        )
    elif delta == 0:
        summary = (
            f"Resume matches {len(matched)} of {total} key JD terms ({percent}% coverage). "
            "No additional score gain was available from the extracted keyword set."
        )
    else:
        summary = (
            f"Resume matches {len(matched)} of {total} key JD terms ({percent}% coverage). "
            "Optimization preserved structure, but keyword gain was limited."
        )

    return {
        "new_score": version_score,
        "improvement_summary": summary,
    }


def _sanitize_analysis(analysis: dict[str, Any], resume_text: str, jd_text: str) -> dict[str, Any]:
    jd_keywords = _extract_jd_keywords(jd_text)
    local_matched, local_missing = _split_keywords(resume_text, jd_keywords)
    local_score = _local_ats_score(resume_text, jd_keywords)

    score = _sanitize_score(analysis.get("ats_score"), local_score)

    matched_keywords = _dedupe_preserve_order([
        item for item in analysis.get("matched_keywords", []) if isinstance(item, str) and item.strip()
    ]) or local_matched[:12]

    missing_keywords = _dedupe_preserve_order([
        item for item in analysis.get("missing_keywords", []) if isinstance(item, str) and item.strip()
    ]) or local_missing[:15]

    suggestions = [
        item for item in analysis.get("suggestions", []) if isinstance(item, str) and item.strip()
    ]
    if len(suggestions) < 3:
        suggestions = _default_suggestions(missing_keywords)

    return {
        "ats_score": score,
        "matched_keywords": matched_keywords[:15],
        "missing_keywords": missing_keywords[:20],
        "suggestions": suggestions[:6],
    }


def _clean_resume_text(text: str) -> str:
    cleaned = _strip_code_fences(text)
    cleaned = cleaned.strip().strip('"')
    return cleaned.replace("\\n", "\n").replace("\\t", "\t").strip()


def _optimize_resume_version(
    model: Any,
    *,
    version_key: str,
    resume_text: str,
    jd_text: str,
    jd_keywords: list[str],
    missing_keywords: list[str],
    target_context: dict[str, Any],
) -> tuple[str, float]:
    best_text = _enforce_improvement(
        resume_text,
        resume_text,
        jd_keywords,
        missing_keywords,
        version_key,
        target_context,
    )
    best_score = _local_ats_score(best_text, jd_keywords)

    working_text = resume_text
    working_missing = _dedupe_preserve_order(target_context.get("priority_keywords", []) + missing_keywords)
    current_score: float | None = None

    for attempt_number in range(1, OPTIMIZATION_ATTEMPTS + 1):
        generated_text = _generate_resume_version(
            model,
            version_key=version_key,
            resume_text=working_text,
            jd_text=jd_text,
            missing_keywords=working_missing,
            target_context=target_context,
            additional_guidance=_build_additional_guidance(
                target_context=target_context,
                remaining_missing=working_missing,
                attempt_number=attempt_number,
                current_score=current_score,
            ),
        )
        improved_text = _enforce_improvement(
            generated_text,
            working_text,
            jd_keywords,
            working_missing,
            version_key,
            target_context,
        )
        version_score = _local_ats_score(improved_text, jd_keywords)

        if version_score >= best_score:
            best_text = improved_text
            best_score = version_score

        _, remaining_missing = _split_keywords(improved_text, jd_keywords)
        if version_score >= OPTIMIZATION_TARGET_SCORE or not remaining_missing:
            break

        working_text = improved_text
        working_missing = _dedupe_preserve_order(
            remaining_missing + target_context.get("priority_keywords", [])
        )
        current_score = version_score

    return best_text, best_score


def _generate_resume_version(
    model: Any,
    *,
    version_key: str,
    resume_text: str,
    jd_text: str,
    missing_keywords: list[str],
    target_context: dict[str, Any],
    additional_guidance: str,
) -> str:
    prompt = _render_prompt(
        OPTIMIZE_PROMPT,
        {
            "<<RESUME_TEXT>>": resume_text,
            "<<JOB_DESCRIPTION>>": jd_text,
            "<<MISSING_KEYWORDS>>": ", ".join(missing_keywords[:20]) or "Use the most relevant JD terms naturally.",
            "<<VERSION_NAME>>": version_key,
            "<<VERSION_RULES>>": VERSION_RULES[version_key],
            "<<TARGET_ROLE>>": target_context.get("target_role", "Target role from the JD"),
            "<<TARGET_SENIORITY>>": target_context.get("seniority", "") or "Not explicitly stated",
            "<<SUMMARY_FOCUS>>": target_context.get("summary_focus", ""),
            "<<ATS_FOCUS>>": target_context.get("ats_focus", ""),
            "<<PRIORITY_KEYWORDS>>": _format_prompt_list(
                target_context.get("priority_keywords", []),
                "Use the most relevant JD terms naturally.",
            ),
            "<<PRIORITY_SKILLS>>": _format_prompt_list(
                target_context.get("priority_skills", []),
                "Use the most relevant skills and tools from the JD.",
            ),
            "<<EXPERIENCE_FOCUS>>": _format_prompt_lines(
                target_context.get("experience_focus", []),
                "- Rewrite experience bullets to better reflect the JD where truthful.",
            ),
            "<<PREFERRED_SECTIONS>>": _format_prompt_list(
                target_context.get("preferred_sections", []),
                "PROFESSIONAL SUMMARY, TECHNICAL SKILLS, EXPERIENCE, EDUCATION, PROJECTS, KEYWORDS",
            ),
            "<<ADDITIONAL_GUIDANCE>>": additional_guidance or "No extra guidance.",
        },
    )

    try:
        raw = _call_gemini(
            model,
            prompt,
            temperature=0.25,
            max_tokens=4096,
            json_output=False,
        )
        cleaned = _clean_resume_text(raw)
        if cleaned:
            return cleaned
    except Exception as exc:
        logger.warning("Resume generation failed for %s (%s); using fallback", version_key, exc)

    return resume_text


def analyze_cv(resume_text: str, jd_text: str) -> dict[str, Any]:
    model = _create_model(GEMINI_MODEL)
    jd_keywords = _extract_jd_keywords(jd_text)
    baseline_score = _local_ats_score(resume_text, jd_keywords)

    analysis_prompt = _render_prompt(
        ANALYSIS_PROMPT,
        {
            "<<RESUME_TEXT>>": resume_text,
            "<<JOB_DESCRIPTION>>": jd_text,
        },
    )

    try:
        raw_analysis = _call_gemini(
            model,
            analysis_prompt,
            temperature=0.1,
            max_tokens=1024,
            json_output=True,
        )
        analysis = _normalize_keys_deep(_extract_json(raw_analysis))
    except Exception as exc:
        logger.warning("Gemini analysis parse failed (%s); using local fallback", exc)
        analysis = {}

    result = _sanitize_analysis(analysis, resume_text, jd_text)
    result["ats_score"] = _sanitize_score(baseline_score, baseline_score)
    target_context = _build_target_context(
        model,
        resume_text=resume_text,
        jd_text=jd_text,
        jd_keywords=jd_keywords,
    )

    optimized_resumes: dict[str, str] = {}
    version_reports: dict[str, dict[str, Any]] = {}

    for version_key in ("ats_heavy", "balanced", "concise"):
        improved_text, version_score = _optimize_resume_version(
            model,
            version_key=version_key,
            resume_text=resume_text,
            jd_text=jd_text,
            jd_keywords=jd_keywords,
            missing_keywords=result["missing_keywords"],
            target_context=target_context,
        )
        optimized_resumes[version_key] = improved_text
        version_reports[version_key] = _build_version_report(
            baseline_score=baseline_score,
            version_score=version_score,
            resume_text=improved_text,
            jd_keywords=jd_keywords,
        )

    result["optimized_resumes"] = optimized_resumes
    result["version_reports"] = version_reports
    result["target_context"] = target_context
    logger.info("Analysis complete with %s missing keywords", len(result["missing_keywords"]))
    return result


def rescore_cv(optimized_resume: str, jd_text: str) -> dict[str, Any]:
    jd_keywords = _extract_jd_keywords(jd_text)
    score = _sanitize_score(_local_ats_score(optimized_resume, jd_keywords), 0.0)
    matched, _ = _split_keywords(optimized_resume, jd_keywords)
    total = max(len(jd_keywords), 1)
    percent = round(len(matched) / total * 100)

    return {
        "new_score": score,
        "improvement_summary": (
            f"Optimized resume matches {len(matched)} of {len(jd_keywords)} "
            f"key JD terms ({percent}% keyword coverage)."
        ),
    }
