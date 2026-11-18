"""Message / event analysis: extract as much as reasonably possible from an
incoming message BEFORE any social decision. Runs locally and cheaply on every
message — the LLM is never required for these fields.

Extracts: message_id, speaker, timestamp, reply_target, mentioned_users,
possible_target, topic, subtopic, message_type, language, novelty, emotional
salience. Some fields come straight from Telegram metadata; others are cheap
heuristics. The result feeds attention, working memory, and the social
decision engine.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import re

from app.language import detect_input_language, DetectedLanguage, InputLanguage
from app.state.manager import extract_topic_keywords

# Message type classification
MSG_TEXT = "text"
MSG_LINK = "link"
MSG_QUESTION = "question"
MSG_STATEMENT = "statement"
MSG_SHORT_CHATTER = "short_chatter"
MSG_REACTION_LIKE = "reaction_like"  # "lol", "😂", "💀", "nice" etc.

# Cheap novelty/salience signals
_EXCLAMATION = set("!؟?")
_REACTION_TOKENS = {"lol", "lmao", "haha", "hehe", "😂", "💀", "😭", "🤣", "ok", "okay", "nice", "wow", "gg", "bro"}
_SHORT_TEXT_MAX = 8

_HUMOR_HINTS = ("lol", "lmao", "😂", "💀", "😭", "🤣", "xd", "joke", "funny", "rofl", "lmaooo")
_EMOTIONAL_HINTS_POS = ("love", "love it", "amazing", "insane", "wow", "wowww", "sick", "goated", "best", "❤", "عشقه")
_EMOTIONAL_HINTS_NEG = ("hate", "awful", "trash", "terrible", "worst", "anger", "mad", "annoying", "💀", "💢")

# Greetings are social bids, not chatter — word-token matched (substring
# matching would catch "hi" inside "this").
_GREETING_TOKENS = {
    "hi", "hii", "hiii", "hello", "helloo", "hellooo", "hey", "heyy", "yo",
    "salam", "salams", "سلام", "درود",
}


@dataclass
class MessageAnalysis:
    message_id: Optional[int] = None
    speaker_user_id: Optional[int] = None
    speaker_username: Optional[str] = None
    timestamp: Optional[datetime] = None
    reply_target_message_id: Optional[int] = None
    reply_target_user_id: Optional[int] = None
    mentioned_users: List[int] = field(default_factory=list)
    possible_target_user_id: Optional[int] = None  # best guess who it's aimed at
    topic: str = ""
    subtopic: str = ""
    keywords: List[str] = field(default_factory=list)
    message_type: str = MSG_TEXT
    language: InputLanguage = InputLanguage.UNKNOWN
    detected_language: Optional[DetectedLanguage] = None
    novelty: float = 0.5  # 0..1 how new/out-of-ordinary this is
    emotional_salience: float = 0.0  # 0..1 emotional charge
    humor_likelihood: float = 0.0  # 0..1
    is_question: bool = False
    is_greeting: bool = False
    # True = positive vibe, False = negative vibe, None = neutral/unknown.
    # Lets the reaction layer pick hearts vs the right kind of shock.
    positive: Optional[bool] = None
    text_length: int = 0
    raw_text: str = ""


def _is_reaction_like(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if len(t.split()) > 3:
        return False
    return t in _REACTION_TOKENS or any(c in text for c in ("😂", "💀", "😭", "🤣"))


def _is_greeting(text: str) -> bool:
    """A short message that is (mostly) a greeting."""
    t = (text or "").strip().lower()
    if not t or len(t.split()) > 3:
        return False
    words = re.findall(r"[\w\u0600-\u06FF]+", t)
    return any(w in _GREETING_TOKENS for w in words)


def _humor_score(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    for hint in _HUMOR_HINTS:
        if hint in t:
            score += 0.25
    if any(c in text or "" for c in ("😂", "💀", "😭", "🤣")):
        score += 0.4
    # Stickers/GIFs are usually memes — reaction-eligible, occasionally funny.
    if t.startswith(("[sticker]", "[gif]")):
        score += 0.25
    return min(1.0, score)


def _emotional_salience(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    for hint in _EMOTIONAL_HINTS_POS:
        if hint in t:
            score += 0.2
    for hint in _EMOTIONAL_HINTS_NEG:
        if hint in t:
            score += 0.2
    if any(c in (text or "") for c in _EXCLAMATION):
        score += 0.15
    if any(c in (text or "") for c in ("😂", "😭", "💀", "💢", "❤")):
        score += 0.2
    return min(1.0, score)


def analyze_message(
    text: str,
    message_id: Optional[int] = None,
    speaker_user_id: Optional[int] = None,
    speaker_username: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    reply_target_message_id: Optional[int] = None,
    reply_target_user_id: Optional[int] = None,
    mentioned_user_ids: Optional[List[int]] = None,
    current_topic: Optional[str] = None,
    recent_hashes: Optional[set] = None,
    recent_window: Optional[List[str]] = None,
) -> MessageAnalysis:
    """Deterministic analysis of one incoming message. No LLM calls."""
    text = text or ""
    recent_hashes = recent_hashes or set()
    recent_window = recent_window or []

    detected = detect_input_language(text)

    keywords = extract_topic_keywords(text, max_keywords=5)

    # Topic/subtopic: derive from keywords; the current topic is carried along
    # so callers can decide transitions. This layer only *names* the message's
    # own topic candidates.
    topic = keywords[0] if keywords else ""
    subtopic = " ".join(keywords[1:4]) if len(keywords) > 1 else ""

    # Message type
    if _is_reaction_like(text):
        mtype = MSG_REACTION_LIKE
    elif "http://" in text or "https://" in text or "www." in text:
        mtype = MSG_LINK
    elif text.strip().endswith(("?", "؟")):
        mtype = MSG_QUESTION
    elif len(text.split()) <= _SHORT_TEXT_MAX:
        mtype = MSG_SHORT_CHATTER
    else:
        mtype = MSG_STATEMENT

    # Novelty: how unlike the recent window this message is.
    novelty = 0.5
    simple_hash = " ".join(keywords) or text[:40].strip().lower()
    if simple_hash in recent_hashes:
        novelty = 0.1
    else:
        if mtype in (MSG_LINK, MSG_QUESTION):
            novelty += 0.15
        if len(text.split()) >= 10:
            novelty += 0.1
        if any(c in text for c in ("😂", "😭", "💀", "!!", "؟")):
            novelty += 0.1
        novelty = min(1.0, novelty)

    # Cheap sentiment polarity for reaction selection (hearts vs shock):
    # positive only when a positive hint matched and none negative did.
    t_lower = text.lower()
    pos_hit = any(h in t_lower for h in _EMOTIONAL_HINTS_POS)
    neg_hit = any(h in t_lower for h in _EMOTIONAL_HINTS_NEG)
    if pos_hit and not neg_hit:
        positive = True
    elif neg_hit and not pos_hit:
        positive = False
    else:
        positive = None

    return MessageAnalysis(
        message_id=message_id,
        speaker_user_id=speaker_user_id,
        speaker_username=speaker_username,
        timestamp=timestamp,
        reply_target_message_id=reply_target_message_id,
        reply_target_user_id=reply_target_user_id,
        mentioned_users=list(mentioned_user_ids or []),
        topic=topic,
        subtopic=subtopic,
        keywords=keywords,
        message_type=mtype,
        language=detected.language,
        detected_language=detected,
        novelty=novelty,
        emotional_salience=_emotional_salience(text),
        humor_likelihood=_humor_score(text),
        is_question=mtype == MSG_QUESTION,
        is_greeting=_is_greeting(text),
        positive=positive,
        text_length=len(text),
        raw_text=text,
    )


def recent_message_hashes(recent_texts: List[str]) -> set:
    """Build a set of simple hashes for the recent window (for novelty checks)."""
    hashes = set()
    for t in recent_texts:
        if not t:
            continue
        words = extract_topic_keywords(t, max_keywords=5)
        hashes.add(" ".join(words) or t[:40].strip().lower())
    return hashes