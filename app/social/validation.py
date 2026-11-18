"""Lightweight final response validation + anti-repetition.

Runs deterministically on every candidate response BEFORE it is sent. The
goal is to catch generic filler, contradictions with important memories,
topic derailment, repetition, and AI-sounding bloat — without an extra
expensive LLM call.

Also tracks recently used phrases/jokes/topics so Airi naturally avoids
repeating herself.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.config import settings
from app.database import RecentPhrase

logger = structlog.get_logger()

# Filler / generic AI phrases that make her sound like a chatbot.
_FILLER_PHRASES = [
    "as an ai", "as an artificial intelligence", "i'm an ai", "i am an ai",
    "as a language model", "as an assistant", "how can i help", "how may i help",
    "glad to help", "hope this helps", "let me know if you", "feel free to ask",
    "i'd be happy to help", "is there anything else", "what would you like to do today",
    "certainly", "absolutely correct", "that's a great question",
]

# Agreement-fishing: a message that is nothing but agreement tokens plus a
# confirmation fish ("yeahh am i right", "right??", "no?", "isn't it"). The
# engagement-maximizing tell — a real person agrees and moves on. Flat
# acknowledgments WITHOUT the fish ("yeah", "ikr", "fr") still pass.
_AGREEMENT_TOKENS = {
    "yeah", "yea", "yeahh", "yeaa", "yeahhh", "yes", "yep", "ya", "yah",
    "true", "fr", "frfr", "exactly", "lol", "lmao", "ikr", "i", "know",
    "right", "real", "facts", "totally", "same", "nah", "no", "literally",
    "omg", "bro", "man", "lolz", "rofl", "haha", "hahaha",
}
_FISH_TOKENS = {
    "am i right", "right", "no", "isn't it", "isnt it", "isn't", "isnt",
    "innit", "or what", "correct", "am i wrong",
}


def _is_agreement_fishing(text: str) -> bool:
    """True when the reply is nothing but agreement + a confirmation fish.

    Every token must be an agreement/fish token AND at least one fish token
    must be present. 'yeahh am i right', 'right??', 'no?', "isn't it lol"
    -> True. 'yeah', 'fr', 'exactly lmao' -> False (no fish, natural acks).
    Anything with real content ('yeah i get what you mean') -> False.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    normalized = stripped.lower().rstrip("!?. ").replace(",", " ")
    words = normalized.split()
    if not words:
        return False

    joined = " " + normalized + " "
    # Multi-word fish phrases are checked on the joined string, then their
    # non-vocabulary words are dropped so the all-tokens pass below only
    # sees genuine filler.
    multi_fish = (
        "am i right" in joined
        or "isn't it" in joined or "isnt it" in joined
        or "innit" in joined
        or "or what" in joined
    )
    words = [w for w in words if w not in ("am", "i", "it", "or", "what")]
    single_fish = (
        ("right" in words and stripped.rstrip().endswith("?"))
        or (words == ["no"] and stripped.rstrip().endswith("?"))
    )
    if not multi_fish and not single_fish:
        return False

    # Everything left in the message must be filler agreement/fish vocabulary.
    return all(w in _AGREEMENT_TOKENS or w in _FISH_TOKENS for w in words)


def _trim_casual(text: str, cap: int) -> str:
    """Cut an over-long casual reply down to the cap, human-style.

    Keeps complete leading sentences (or a clause boundary as fallback)
    rather than hard-chopping mid-word. Never returns something worse than
    the input: if nothing sensible fits under the cap, the last sentence
    under the cap wins; a single long sentence gets clause-trimmed.
    """
    text = (text or "").strip()
    if len(text) <= cap:
        return text

    import re as _re
    sentences = _re.split(r"(?<=[.!?…])\s+", text)
    # Keep leading sentences that fit under the cap.
    kept = []
    for s in sentences:
        candidate = (" ".join(kept + [s])).strip()
        if len(candidate) > cap:
            break
        kept.append(s)
    if kept:
        return " ".join(kept)

    # One long sentence: try a clause boundary (", but", " — ", "; ").
    for sep in ("; ", " — ", ", but ", ", and "):
        idx = text.find(sep, 0, cap)
        if idx > 20:
            return text[:idx].rstrip(",;") + ("..." if text[:idx].endswith((",", ";")) else "")

    # Last resort: word-boundary hard trim (no mid-word cuts).
    cut = text[:cap]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(",;:") + "..."

_GENERIC_HELLO = {"hi", "hello", "hey there", "how are you", "good morning", "good evening"}

# Generic phrases repeated verbatim across long windows are suspicious.
_MIN_UNIQUE_PHRASES = 3


class AntiRepetitionTracker:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(
        self,
        conversation_id: Optional[int],
        phrase_type: str,
        content: str,
        target_user_id: Optional[int] = None,
        topic: Optional[str] = None,
    ):
        content = (content or "").strip()
        if not content:
            return
        row = RecentPhrase(
            conversation_id=conversation_id,
            phrase_type=phrase_type,
            content=content[:300],
            target_user_id=target_user_id,
            topic=(topic or "")[:300] or None,
            created_at=datetime.utcnow(),
        )
        self.session.add(row)
        await self.session.flush()
        await self._prune(conversation_id)

    async def record_response(self, conversation_id: Optional[int], text: str, topic: str,
                              target_user_id: Optional[int] = None):
        await self.record(conversation_id, "response_pattern", text,
                          target_user_id=target_user_id, topic=topic)
        await self.record(conversation_id, "topic", topic)

    async def recent(self, conversation_id: Optional[int], phrase_type: str, limit: int = 10) -> List[RecentPhrase]:
        stmt = select(RecentPhrase).where(RecentPhrase.phrase_type == phrase_type)
        if conversation_id is not None:
            stmt = stmt.where(RecentPhrase.conversation_id == conversation_id)
        stmt = stmt.order_by(RecentPhrase.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def is_recent_duplicate(
        self,
        conversation_id: Optional[int],
        text: str,
        threshold: float = 0.85,
        target_user_id: Optional[int] = None,
        topic: Optional[str] = None,
    ) -> bool:
        """Contextual repetition check: is this *spammy* repetition?

        Humans reuse phrases across people and contexts, so a phrase is only
        treated as repetition when it repeats toward the SAME person in the
        SAME context (same target, similar topic) within the recent window.
        The same phrase to a different person is normal social behavior.
        """
        import difflib
        recent = await self.recent(conversation_id, "response_pattern",
                                   limit=settings.anti_repetition_window)
        text_l = (text or "").strip().lower()
        if not text_l:
            return False

        topic_words = set((topic or "").lower().split())
        for old in recent:
            # Different addressee -> reuse is natural, not spam.
            if old.target_user_id is not None and target_user_id is not None \
                    and old.target_user_id != target_user_id:
                continue
            # Different topic -> reuse is natural, not spam.
            if old.topic and topic_words and not (topic_words & set(old.topic.lower().split())):
                continue

            old_l = (old.content or "").strip().lower()
            if old_l == text_l:
                return True
            if len(old_l) > 3 and difflib.SequenceMatcher(None, old_l, text_l).ratio() >= threshold:
                return True
        return False

    async def _prune(self, conversation_id: Optional[int], keep: int = 60):
        stmt = select(RecentPhrase)
        if conversation_id is not None:
            stmt = stmt.where(RecentPhrase.conversation_id == conversation_id)
        stmt = stmt.order_by(RecentPhrase.created_at.desc())
        result = await self.session.execute(stmt)
        rows = result.scalars().all()
        if len(rows) > keep:
            for row in rows[keep:]:
                await self.session.delete(row)
        await self.session.flush()


class ResponseValidator:
    """Deterministic checks on a candidate response."""

    def __init__(self, session: AsyncSession, bot_username: str = "Airi"):
        self.session = session
        self.bot_username = bot_username
        self.anti_repetition = AntiRepetitionTracker(session)

    async def validate(
        self,
        response: str,
        *,
        current_topic: Optional[str] = None,
        current_mood: Optional[str] = None,
        is_reply_to_airi: bool = False,
        important_memories: Optional[List[Any]] = None,
        recent_topic_history: Optional[List[str]] = None,
        conversation_id: Optional[int] = None,
        target_user_id: Optional[int] = None,
        room_avg_words: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return {ok, issues, response} — mutating response only if fixable."""
        issues: List[str] = []
        response = (response or "").strip()

        if not response:
            return {"ok": False, "issues": ["empty_response"], "response": response}

        lowered = response.lower()

        # 1. Generic AI filler
        for phrase in _FILLER_PHRASES:
            if phrase in lowered:
                issues.append(f"ai_filler:{phrase}")
                break

        # 1b. Agreement-fishing ("yeahh am i right", "right??") — the
        #     engagement-maximizing tell. Only fatal when the WHOLE message
        #     is the fish; agreement with substance ("yeah that's true, the
        #     pacing is awful though") still passes.
        if _is_agreement_fishing(response):
            issues.append("agreement_fishing")

        # 2. Unnecessarily long
        if len(response) > settings.max_response_chars:
            issues.append(f"too_long:{len(response)}")

        # 2b. Casual-mode length gate. Her measured failure mode: essay-length
        #     replies in a room of one-liners (bot median 41 chars vs human
        #     14). Instead of dropping the reply (silence on a direct question
        #     is worse), trim it the way a person would: keep the leading
        #     sentences that fit, drop the essay. Long-form rooms (everyone
        #     writing paragraphs) are exempt.
        casual_cap = settings.casual_max_chars
        if casual_cap and len(response) > casual_cap:
            long_form_room = room_avg_words is not None and room_avg_words >= 25
            if not long_form_room:
                trimmed = _trim_casual(response, casual_cap)
                if trimmed != response:
                    issues.append(f"casual_trimmed:{len(response)}->{len(trimmed)}")
                    response = trimmed
                    lowered = response.lower()

        # 3. Repetition — contextual (same person + same context only). A
        #    direct message to Airi gets extra protection against accidental
        #    suppression: repetition there is never blocking on its own.
        if not is_reply_to_airi and await self.anti_repetition.is_recent_duplicate(
            conversation_id, response, target_user_id=target_user_id, topic=current_topic,
        ):
            issues.append("repetitive")

        # 4. Generic greeting when a reply isn't demanded
        if not is_reply_to_airi and lowered.strip().rstrip("!?.") in _GENERIC_HELLO:
            issues.append("generic_greeting")

        # 5. Memory contradiction (cheap substring pass over important memories)
        if important_memories:
            for mem in important_memories:
                content = (mem.content if hasattr(mem, "content") else str(mem))[:200]
                content_l = content.lower()
                # If Airi asserts a strong negative of an established important
                # fact, flag it (best-effort; never blocks genuine correction).
                strong_neg = ["doesn't like", "hates", "is not", "never", "wasn't"]
                if mem.importance >= 0.8 and any(n in lowered for n in strong_neg) \
                        and any(w in content_l for w in ("likes", "loves", "enjoys", "favorite")):
                    issues.append(f"contradicts_memory:{mem.content[:60]}")
                    break

        ok = len(issues) == 0
        if issues:
            logger.debug(
                "response_validation_issues",
                issues=issues,
                response_preview=response[:80],
            )
        return {"ok": ok, "issues": issues, "response": response}

    async def should_skip(self, validation: Dict[str, Any]) -> bool:
        """Whether to block a response entirely.

        Prefer ALLOW over silence: repetition alone never cancels a reply —
        humans reuse phrases. Hard-blocked: genuine AI filler and pure
        agreement-fishing (both are engagement tells no human produces
        consistently). Over-length casual replies are handled by trimming in
        validate(), not by blocking.
        """
        for issue in validation["issues"]:
            if issue.startswith("ai_filler:") or issue == "agreement_fishing":
                return True
        return False