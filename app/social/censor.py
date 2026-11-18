"""Outgoing-message profanity filter.

She swears in her own voice; the platform bleeps the hard ones with
[filtered], streamer-style. The filter runs at send time on every outgoing
text — the LLM never self-censors, which is what keeps the swearing natural
instead of performed. Type "fuck", see "[filtered]".

Hard profanity only: mild swears (damn, hell, crap, ass, piss) pass through
untouched — censoring those would flatten her voice to nothing.
"""
import re

from app.config import settings

# Word-boundary anchored so innocent neighbors (class, cocktail, Dickens,
# shiitake) never match; stems with suffixes catch conjugations
# (fucking, fucked, bitches, assholes).
_PATTERN = re.compile(
    r"\b("
    r"motherfuck\w*|fuck\w*|f+u+c+k+\w*|"
    r"shit\w*|"
    r"bitch\w*|"
    r"assholes?|asshats?|"
    r"cunts?|bastards?|"
    r"dickheads?|dicks?|"
    r"cocks?|twats?|wankers?|pricks?|bollocks|"
    r"sluts?|whores?|slags?"
    r")\b",
    re.IGNORECASE,
)

_REPLACEMENT = "[filtered]"


def censor(text: str) -> str:
    """Bleep hard profanity in an outgoing message."""
    if not text or not settings.profanity_filter:
        return text
    return _PATTERN.sub(_REPLACEMENT, text)
