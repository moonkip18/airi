"""Local input-language detection.

Cheap, deterministic heuristics only — no LLM calls. This runs on every
incoming message, so it must be fast. It separates INPUT language (what the
user wrote) from OUTPUT language (what Airi decides to speak), which the
policy module owns.
"""
import re
from dataclasses import dataclass, field
from enum import Enum

# Persian/Farsi uses the Arabic script with Persian-specific additions.
_PERSIAN_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)
_LATIN_RANGE = (0x0041, 0x007A)

# Common colloquial Persian expressions seen in group chats. When one of these
# appears, Airi *recognized a funny Persian expression* — a deliberate-Persian
# trigger candidate (never an automatic language switch).
PERSIAN_EXPRESSIONS = [
    "بابا ول کن", "ول کن", "ایول", "خفن", "داداش", "بیخیال", "بی خیال",
    "چی میگی", "چی میگی؟", "نگو", "وای", "وییی", "خدایی", "باشه", "چی شده",
    "اهان", "الکی", "توپه", "مرسی", "لطفا", "چرا", "عجب", "چه خبر", "خوبی",
    "چطوری", "قشنگ", "خیلی خفن", "نمی‌دونم", "نمیدونم", "میمیرم", "خنده داره",
    "عشقه", "فدات", "جون", "جانم", "اوف", "اوف بابا", "خب", "چی",
]

# Common transliterated Persian (Finglish) tokens — lowercase matching.
FINGLISH_TOKENS = {
    "salam", "khubi", "khoobi", "khubam", "khoobam", "chetori", "chikar",
    "mikoni", "mikham", "dari", "midi", "bashe", "nakon", "dadash", "mishe",
    "nabashe", "eyval", "khafan", "alki", "vay", "mersi", "lotfan", "vaghti",
    "kheyli", "mamnoon", "midoni", "nemidoonam", "nisti", "bia", "berim",
    "fidat", "aziz", "chera", "chand", "age", "nago", "velkon", "baba",
    "inghadr", "kesafat", "halam", "kooft", "bikhial", "chi", "shod",
}


class InputLanguage(str, Enum):
    PERSIAN = "persian"
    ENGLISH = "english"
    MIXED = "mixed"          # substantial amounts of both scripts
    FINGLISH = "finglish"    # persian written in latin letters
    UNKNOWN = "unknown"


@dataclass
class DetectedLanguage:
    language: InputLanguage
    persian_script_ratio: float = 0.0
    latin_ratio: float = 0.0
    has_persian_script: bool = False
    has_transliterated_persian: bool = False
    matched_expression: str = ""
    details: dict = field(default_factory=dict)


def _script_ratios(text: str) -> tuple:
    """Return (persian_ratio, latin_ratio) over alphabetic characters."""
    total = 0
    persian = 0
    latin = 0
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _PERSIAN_RANGES):
            persian += 1
            total += 1
        elif _LATIN_RANGE[0] <= code <= _LATIN_RANGE[1]:
            latin += 1
            total += 1
    if total == 0:
        return 0.0, 0.0
    return persian / total, latin / total


def _contains_persian_expression(text: str) -> str:
    for expr in PERSIAN_EXPRESSIONS:
        if expr in text:
            return expr
    return ""


def _has_finglish(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in FINGLISH_TOKENS)
    # At least two transliterated tokens, or one in a short message.
    return hits >= 2 or (hits >= 1 and len(words) <= 4)


def detect_input_language(text: str) -> DetectedLanguage:
    """Detect the language of an incoming message. Purely local logic."""
    if not text or not text.strip():
        return DetectedLanguage(language=InputLanguage.UNKNOWN)

    persian_ratio, latin_ratio = _script_ratios(text)
    expression = _contains_persian_expression(text)
    finglish = _has_finglish(text)

    if persian_ratio > 0.15 and latin_ratio > 0.15:
        language = InputLanguage.MIXED
    elif persian_ratio >= 0.15:
        language = InputLanguage.PERSIAN
    elif latin_ratio > 0 and finglish:
        language = InputLanguage.FINGLISH
    elif latin_ratio > 0:
        language = InputLanguage.ENGLISH
    else:
        language = InputLanguage.UNKNOWN

    return DetectedLanguage(
        language=language,
        persian_script_ratio=round(persian_ratio, 3),
        latin_ratio=round(latin_ratio, 3),
        has_persian_script=persian_ratio >= 0.15,
        has_transliterated_persian=finglish,
        matched_expression=expression,
        details={"length": len(text)},
    )


def is_persian(detected: DetectedLanguage) -> bool:
    """True when the input involves Persian in any form (script, mixed, finglish)."""
    return detected.language in (
        InputLanguage.PERSIAN,
        InputLanguage.MIXED,
        InputLanguage.FINGLISH,
    )
