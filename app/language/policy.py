"""Output-language decision policy.

Airi's OUTPUT language is English by default. Persian is a deliberate,
rare behavioral choice — never automatic mirroring. This module implements
that rule with application logic (no LLM calls needed for the common case):

  DEFAULT              -> english
  Persian PERMITTED    -> only under deliberate triggers:
                           * a genuinely funny Persian expression appeared
                             AND Airi is highly amused
                           * teasing someone she likes who speaks Persian
                           * an established recurring Persian joke
  Even then, Persian is usually a SHORT snippet inside an English response.
"""
import random
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel

import structlog
from app.config import settings
from app.language.detector import DetectedLanguage, is_persian

logger = structlog.get_logger()


class LanguageDecision(BaseModel):
    """Explicit, auditable output-language decision."""
    language: Literal["english", "persian"] = "english"
    reason: str = "default_english"
    allow_persian_snippet: bool = False
    snippet_max_words: int = 6
    input_language: str = "unknown"

    @property
    def is_persian(self) -> bool:
        return self.language == "persian"


class LanguagePolicy:
    """Decides Airi's output language BEFORE response generation.

    The decision is deliberately conservative: English is the overwhelming
    default. Persian requires a deliberate trigger AND favorable internal
    state, and is usually granted only as a short snippet allowance rather
    than a full switch.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()

    def decide(
        self,
        detected: DetectedLanguage,
        emotional_state: Dict[str, float],
        relationship: Dict[str, float],
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        recurring_jokes: Optional[List[Dict[str, Any]]] = None,
        recent_persian_outputs: int = 0,
    ) -> LanguageDecision:
        recent_messages = recent_messages or []
        recurring_jokes = recurring_jokes or []

        amusement = float(emotional_state.get("amusement", 0.0))
        affection = float(relationship.get("affection", 0.0))
        curiosity = float(emotional_state.get("curiosity", 0.0))
        annoyance = float(emotional_state.get("annoyance", 0.0))

        base = LanguageDecision(
            language="english",
            input_language=detected.language.value,
        )

        # Annoyance suppresses playfulness — Airi won't do cute Persian
        # code-switching while irritated.
        if annoyance >= 0.7:
            base.reason = "english_default_annoyed"
            return base

        # Anti-spam for Persian usage: if she used Persian recently, back off
        # so it stays noticeable and rare.
        if recent_persian_outputs >= 2:
            base.reason = "english_default_recent_persian_use"
            return base


        # --- Trigger 1: established recurring Persian joke -------------------
        joke = self._matching_recurring_joke(detected, recent_messages, recurring_jokes)
        if joke and amusement >= 0.4:
            base.language = "english"
            base.allow_persian_snippet = True
            base.snippet_max_words = 12
            base.reason = f"recurring_joke:{joke.get('phrase', '')[:40]}"
            logger.debug("language_decision_persian_snippet_allowed", reason=base.reason)
            return base

        # --- Trigger 2: funny Persian expression + high amusement ------------
        if detected.matched_expression and amusement >= settings.persian_amusement_threshold:
            roll = self._rng.random()
            if roll < settings.persian_snippet_probability:
                base.language = "english"
                base.allow_persian_snippet = True
                base.snippet_max_words = 6
                base.reason = f"funny_expression:{detected.matched_expression[:30]}"
                logger.debug("language_decision_persian_snippet_allowed", reason=base.reason)
                return base
            base.reason = "english_default_expression_not_that_funny"
            return base

        # --- Trigger 3: affectionate teasing of a close Persian speaker ------
        if is_persian(detected) and affection >= settings.persian_affection_threshold:
            teasing_context = self._teasing_context(recent_messages)
            if teasing_context and amusement >= settings.persian_amusement_threshold * 0.8:
                roll = self._rng.random()
                if roll < settings.persian_snippet_probability:
                    base.language = "english"
                    base.allow_persian_snippet = True
                    base.snippet_max_words = 5
                    base.reason = f"teasing_close_friend:{teasing_context[:30]}"
                    return base
            base.reason = "english_default_close_friend_no_trigger"
            return base

        # --- Trigger 4: rare full-Persian comedic outburst --------------------
        # Only when: Persian input, extreme amusement, high affection/curiosity,
        # and a random gate. This produces full Persian responses very rarely.
        if (
            is_persian(detected)
            and amusement >= 0.9
            and (affection >= settings.persian_affection_threshold or curiosity >= settings.persian_curiosity_threshold)
            and self._rng.random() < settings.persian_snippet_probability * 0.2
        ):
            base.language = "persian"
            base.allow_persian_snippet = True
            base.snippet_max_words = 20
            base.reason = "rare_full_persian_outburst"
            logger.info("language_decision_full_persian", reason=base.reason)
            return base

        base.reason = "english_default"
        return base

    def _matching_recurring_joke(
        self,
        detected: DetectedLanguage,
        recent_messages: List[Dict[str, Any]],
        recurring_jokes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not recurring_jokes:
            return None
        recent_text = " ".join(str(m.get("text", "")) for m in recent_messages[-8:])
        for joke in recurring_jokes:
            phrase = str(joke.get("phrase", ""))
            if not phrase:
                continue
            if joke.get("language") == "persian" and (phrase in recent_text or phrase in detected.matched_expression):
                return joke
        return None

    def _teasing_context(self, recent_messages: List[Dict[str, Any]]) -> str:
        """Cheap heuristic: does the recent conversation look like banter?"""
        banter_markers = ["lol", "lmao", "😂", "😭", "bro", "dude", "xDD", "برو گمشو", "خفن"]
        window = recent_messages[-5:] if recent_messages else []
        for msg in window:
            text = str(msg.get("text", "")).lower()
            if any(marker in text for marker in banter_markers):
                return str(msg.get("text", ""))[:60]
        return ""


language_policy = LanguagePolicy()
